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

        self.common_layers = []
        self.common_layers.append(kl.Conv2D(64, 3, padding='same', input_shape=(Game.HEIGHT+2, Game.WIDTH+2, 2)))
        #self.common_layers.append(kl.BatchNormalization())
        self.common_layers.append(kl.Activation('relu'))
        self.common_layers.append(kl.Conv2D(64, 3, padding='same'))
        #self.common_layers.append(kl.BatchNormalization())
        self.common_layers.append(kl.Activation('relu'))
        self.common_layers.append(kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #self.common_layers.append(kl.Dropout(0.25))
        self.common_layers.append(kl.Conv2D(128, 3, padding='same'))
        # self.common_layers.append(kl.BatchNormalization())
        self.common_layers.append(kl.Activation('relu'))
        self.common_layers.append(kl.Conv2D(128, 3, padding='same'))
        # self.common_layers.append(kl.BatchNormalization())
        self.common_layers.append(kl.Activation('relu'))
        self.common_layers.append(kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.common_layers.append(kl.Dropout(0.25))
        self.common_layers.append(kl.Flatten())

        # actor branch
        # final layer should output logits
        self.actor_layers = []
        #self.actor_layers.append(kl.Dense(256))
        self.actor_layers.append(kl.Dense(1024))
        #self.actor_layers.append(kl.BatchNormalization())
        self.actor_layers.append(kl.Activation('relu'))
        #self.actor_layers.append(kl.Dropout(0.25))
        #self.actor_layers.append(kl.Dense(64))
        self.actor_layers.append(kl.Dense(128))
        #self.actor_layers.append(kl.BatchNormalization())
        self.actor_layers.append(kl.Activation('relu'))
        #self.actor_layers.append(kl.Dropout(0.25))
        self.actor_layers.append(kl.Dense(len(self.all_possible_actions_in_game), name='policy_logits'))

        # critic branch
        # final layer should output a single value (state value / expected reward)
        self.critic_layers = []
        #self.critic_layers.append(kl.Dense(256))
        self.critic_layers.append(kl.Dense(1024))
        #self.critic_layers.append(kl.BatchNormalization())
        self.critic_layers.append(kl.Activation('relu'))
        #self.critic_layers.append(kl.Dropout(0.25))
        #self.critic_layers.append(kl.Dense(64))
        self.critic_layers.append(kl.Dense(128))
        #self.critic_layers.append(kl.BatchNormalization())
        self.critic_layers.append(kl.Activation('relu'))
        #self.critic_layers.append(kl.Dropout(0.25))
        self.critic_layers.append(kl.Dense(1, name='value'))

        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        output_from_common_layers = tf.convert_to_tensor(inputs)
        # forward pass through common layers
        for layer in self.common_layers:
            output_from_common_layers = layer(output_from_common_layers)

        actor_output = output_from_common_layers
        # forward pass through actor layers
        for layer in self.actor_layers:
            actor_output = layer(actor_output)

        critic_output = output_from_common_layers
        # forward pass through actor layers
        for layer in self.critic_layers:
            critic_output = layer(critic_output)
        return actor_output, critic_output

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.95,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=0.001),
            loss=[self._logits_loss, self._value_loss]
        )

    def train(self, env, batch_sz=32, updates=100):
        action_ids = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + (Game.HEIGHT+2, Game.WIDTH+2, 2))

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
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.params['entropy'] * entropy_loss

    def _action_from_id(self, action_id):
        return self.model.all_possible_actions_in_game[action_id]
