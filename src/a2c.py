import os.path
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

from game import Game, Action


class EntryLayer(tf.keras.Model):
    def __init__(self):
        super(EntryLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

    def call(self, input_tensor, training=False):
        return self.conv(input_tensor)


class RepeatedLayer(tf.keras.Model):
    def __init__(self):
        super(RepeatedLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')

    def call(self, input_tensor, training=False):
        return self.conv(input_tensor)


class PolicyHeadLayer(tf.keras.Model):
    def __init__(self, num_of_actions):
        super(PolicyHeadLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(2, (1, 1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_of_actions)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.flatten(x)
        return self.dense(x)


class ValueHeadLayer(tf.keras.Model):
    def __init__(self):
        super(ValueHeadLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


class SampleActionFromLogits(tf.keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class TopActionFromLogits(tf.keras.Model):
    def call(self, logits):
        return tf.argmax(logits, 1)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.all_possible_actions_in_game = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] # ignoring Action.NONE

        self.common_layers = []
        self.common_layers.append(EntryLayer())
        for _ in range(3):
            self.common_layers.append(RepeatedLayer())

        self.actor_head = PolicyHeadLayer(len(self.all_possible_actions_in_game))
        self.critic_head = ValueHeadLayer()

        self.sample_action_from_logits = SampleActionFromLogits()
        self.top_action_from_logits = TopActionFromLogits()

    def call(self, input_tensor, training=False):
        output_from_common_layers = tf.convert_to_tensor(input_tensor)

        for layer in self.common_layers:
            output_from_common_layers = layer(output_from_common_layers, training=training)

        actor_output = self.actor_head(output_from_common_layers, training=training)
        critic_output = self.critic_head(output_from_common_layers, training=training)

        return actor_output, critic_output

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.sample_action_from_logits.predict(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def top_action(self, obs):
        logits, _ = self.predict(obs)
        action = self.top_action_from_logits.predict(logits)
        return np.squeeze(action, axis=-1)

    def get_variables(self):
        return self.get_weights()

    def set_variables(self, variables):
        self.set_weights(variables)


class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=0.001),
            loss=[self._logits_loss, self._value_loss]
        )

    def initialize_model(self, env):
        self.model.action_value(env.cur_obs()[None, :])
        self.model.top_action(env.cur_obs()[None, :])

    def generate_experience_batch(self, env, batch_size):
        action_ids = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + (Game.HEIGHT+2, Game.WIDTH+2, 2))

        ep_rews = [0.0]
        next_obs = env.cur_obs()
        for step in range(batch_size):
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
        return observations, acts_and_advs, returns

    def select_top_action(self, obs):
        action_id = self.model.top_action(obs[None, :])
        return self._action_from_id(action_id)

    def save_model(self):
        self.model.save_weights('saved_model/weights', save_format='tf')

    def load_model_if_previously_saved(self):
        if os.path.exists('saved_model'):
            self.model.load_weights('saved_model/weights').expect_partial()

    def load_pretrained_model(self):
        if os.path.exists('pretrained_model'):
            self.model.load_weights('pretrained_model/weights').expect_partial()

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
