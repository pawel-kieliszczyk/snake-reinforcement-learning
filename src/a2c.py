import os.path
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from game import Action
from game import Game


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__('a2c_model')
        self.all_possible_actions_in_game = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN] # ignoring Action.NONE

        self.conv1 = kl.Conv2D(32, 5, padding='same', input_shape=(Game.HEIGHT+2, Game.WIDTH+2, 3))
        self.bn1 = kl.BatchNormalization()
        self.activation1 = kl.Activation('relu')
        self.conv2 = kl.Conv2D(64, 5, padding='same')
        self.bn2 = kl.BatchNormalization()
        self.activation2 = kl.Activation('relu')
        self.pooling1 = kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout1 = kl.Dropout(0.25)
        self.conv3 = kl.Conv2D(128, 3, padding='same')
        self.bn3 = kl.BatchNormalization()
        self.activation3 = kl.Activation('relu')
        self.conv4 = kl.Conv2D(128, 3, padding='same')
        self.bn4 = kl.BatchNormalization()
        self.activation4 = kl.Activation('relu')
        self.pooling2 = kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.dropout2 = kl.Dropout(0.25)
        self.flatten = kl.Flatten()

        # actor branch
        self.actor_dense1 = kl.Dense(256)
        self.actor_bn1 = kl.BatchNormalization()
        self.actor_activation1 = kl.Activation('relu')
        self.actor_dropout1 = kl.Dropout(0.25)
        self.actor_dense2 = kl.Dense(128)
        self.actor_bn2 = kl.BatchNormalization()
        self.actor_activation2 = kl.Activation('relu')
        self.actor_dropout2 = kl.Dropout(0.25)
        self.actor_logits = kl.Dense(len(self.all_possible_actions_in_game), name='policy_logits')

        #critic branch
        self.critic_dense1 = kl.Dense(256)
        self.critic_bn1 = kl.BatchNormalization()
        self.critic_activation1 = kl.Activation('relu')
        self.critic_dropout1 = kl.Dropout(0.25)
        self.critic_dense2 = kl.Dense(128)
        self.critic_bn2 = kl.BatchNormalization()
        self.critic_activation2 = kl.Activation('relu')
        self.critic_dropout2 = kl.Dropout(0.25)
        self.critic_value = kl.Dense(1, name='value')

        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        activation1 = self.activation1(bn1)
        conv2 = self.conv2(activation1)
        bn2 = self.bn2(conv2)
        activation2 = self.activation2(bn2)
        pooling1 = self.pooling1(activation2)
        dropout1 = self.dropout1(pooling1)
        conv3 = self.conv3(dropout1)
        bn3 = self.bn3(conv3)
        activation3 = self.activation3(bn3)
        conv4 = self.conv4(activation3)
        bn4 = self.bn4(conv4)
        activation4 = self.activation4(bn4)
        pooling2 = self.pooling2(activation4)
        dropout2 = self.dropout2(pooling2)
        flatten = self.flatten(dropout2)

        # actor branch
        actor_dense1 = self.actor_dense1(flatten)
        actor_bn1 = self.actor_bn1(actor_dense1)
        actor_activation1 = self.actor_activation1(actor_bn1)
        actor_dropout1 = self.actor_dropout1(actor_activation1)
        actor_dense2 = self.actor_dense2(actor_dropout1)
        actor_bn2 = self.actor_bn2(actor_dense2)
        actor_activation2 = self.actor_activation2(actor_bn2)
        actor_dropout2 = self.actor_dropout2(actor_activation2)
        actor_logits = self.actor_logits(actor_dropout2)

        #critic branch
        critic_dense1 = self.critic_dense1(flatten)
        critic_bn1 = self.critic_bn1(critic_dense1)
        critic_activation1 = self.critic_activation1(critic_bn1)
        critic_dropout1 = self.critic_dropout1(critic_activation1)
        critic_dense2 = self.critic_dense2(critic_dropout1)
        critic_bn2 = self.critic_bn2(critic_dense2)
        critic_activation2 = self.critic_activation2(critic_bn2)
        critic_dropout2 = self.critic_dropout2(critic_activation2)
        critic_value = self.critic_value(critic_dropout2)

        return actor_logits, critic_value

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(),
            loss=[self._logits_loss, self._value_loss]
        )

    def train(self, env, batch_sz=32, updates=100):
        action_ids = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + (Game.HEIGHT+2, Game.WIDTH+2, 3))

        ep_rews = [0.0]
        next_obs = env.reset()
        for _ in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                action_ids[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step] = env.step(self._action_from_id(action_ids[step]))

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            acts_and_advs = np.concatenate([action_ids[:, None], advs[:, None]], axis=-1)
            self.model.train_on_batch(observations, [acts_and_advs, returns])

        return ep_rews

    def select_action(self, obs):
        action_id, _ = self.model.action_value(obs[None, :])
        return self._action_from_id(action_id)

    def save_model(self):
        self.model.save_weights('saved_model/weights', save_format='tf')

    def load_model_if_previously_saved(self, env):
        if os.path.exists('saved_model'):
            self.train(env, updates=1) # needed to initialize the model
            self.model.load_weights('saved_model/weights')

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        return self.params['value'] * kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        weighted_sparse_ce = kls.CategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.params['entropy'] * entropy_loss

    def _action_from_id(self, action_id):
        return self.model.all_possible_actions_in_game[action_id]
