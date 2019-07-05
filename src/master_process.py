from mpi4py import MPI
import numpy as np

import mpi_helper

from game import Game
from a2c import Model, A2CAgent
from learning_environment import LearningEnvironment


class Master(object):
    def run(self, max_worker_id, epochs, worker_batch_size):
        model = Model()
        agent = A2CAgent(model)
        learning_environment = LearningEnvironment()

        agent.initialize_model(learning_environment)
        agent.load_model_if_previously_saved()

        saved_model_score = self._get_average_score(learning_environment, agent, 100)

        for ep in range(1, epochs+1):
            print("Epoch {}/{}".format(ep, epochs))

            if ep % 1000 == 0:
                current_score = self._get_average_score(learning_environment, agent, 100)
                if current_score > saved_model_score:
                    agent.save_model()
                    saved_model_score = current_score

            variables = model.get_variables()
            self._send_variables_to_workers(variables)

            all_observations, all_acts_and_advs, all_returns = self._receive_experience_from_worker(1, worker_batch_size)
            for i in range(2, max_worker_id+1):
                observations, acts_and_advs, returns = self._receive_experience_from_worker(i, worker_batch_size)

                all_observations = np.concatenate((all_observations, observations))
                all_acts_and_advs = np.concatenate((all_acts_and_advs, acts_and_advs))
                all_returns = np.concatenate((all_returns, returns))

            model.train_on_batch(all_observations, [all_acts_and_advs, all_returns])

        current_score = self._get_average_score(learning_environment, agent, 100)
        if current_score > saved_model_score:
            agent.save_model()

    def _send_variables_to_workers(self, variables):
        for v in variables:
            mpi_helper.broadcast(v)

    def _receive_experience_from_worker(self, worker_id, worker_batch_size):
        observations = np.empty((worker_batch_size, Game.HEIGHT+2, Game.WIDTH+2, 2), dtype=np.float64)
        acts_and_advs = np.empty((worker_batch_size, 2), dtype=np.float64)
        returns = np.empty(worker_batch_size, dtype=np.float64)

        mpi_helper.receive(observations, source=worker_id)
        mpi_helper.receive(acts_and_advs, source=worker_id)
        mpi_helper.receive(returns, source=worker_id)

        return observations, acts_and_advs, returns

    def _get_average_score(self, learning_environment, agent, num_of_games):
        print("Playing {} test games...".format(num_of_games))
        scores = []
        for _ in range(num_of_games):
            scores.append(self._play_test_game(learning_environment, agent))

        avg_score = int(sum(scores) / len(scores))
        print("Test games finished with average score: {}".format(avg_score))
        return avg_score

    def _play_test_game(self, learning_environment, agent):
        game = Game(select_random_snake_and_food_positions=True)
        steps_without_scoring = 0
        while steps_without_scoring < 200 and not game.is_finished():
            obs = learning_environment.build_observation(game)
            action = agent.select_top_action(obs)

            score_before = game.get_score()
            game.make_action(action)
            score_after = game.get_score()

            if score_after > score_before:
                steps_without_scoring = 0

            steps_without_scoring += 1

        return game.get_score()
