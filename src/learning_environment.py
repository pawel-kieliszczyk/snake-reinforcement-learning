import numpy as np

from game import Action
from game import Game


class LearningEnvironment(object):
    def __init__(self):
        self.game = Game(select_random_snake_and_food_positions=True)
        self.steps_without_scoring = 0
        self.recent_game_scores = []

    def reset(self):
        self.game = Game(select_random_snake_and_food_positions=True)
        self.steps_without_scoring = 0
        return self.build_observation(self.game)

    def step(self, action):
        score_before = self.game.get_score()
        self.game.make_action(action)
        score_after = self.game.get_score()

        observation = self.build_observation(self.game)
        reward = score_after - score_before
        done = self.game.is_finished()

        if reward == 0:
            self.steps_without_scoring += 1
        else:
            self.steps_without_scoring = 0

        if self.steps_without_scoring == 300:
            done = True

        if done:
            self.recent_game_scores.append(self.game.get_score())
            while len(self.recent_game_scores) > 1000:
                del self.recent_game_scores[0]

        return (observation, reward, done)

    def build_observation(self, g):
        observation = np.zeros((Game.HEIGHT + 2, Game.WIDTH + 2, 2))

        # border
        for i in range(Game.HEIGHT+2):
            observation[i, 0, 0] = 1.0
            observation[i, Game.WIDTH + 1, 0] = 1.0
        for i in range(Game.WIDTH+2):
            observation[0, i, 0] = 1.0
            observation[Game.HEIGHT + 1, i, 0] = 1.0

        # snake's head
        observation[g.snake[0][0]+1, g.snake[0][1]+1, 1] = 1.0
        # snake's tail
        for (r, c) in g.snake[1:]:
            observation[r+1, c+1, 0] = 1.0

        # food
        observation[g.food_at[0]+1, g.food_at[1]+1, 0] = -1.0

        return observation

    def get_average_score(self):
        if len(self.recent_game_scores) == 0:
            return 0

        return int(sum(self.recent_game_scores) / len(self.recent_game_scores))
