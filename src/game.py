from enum import Enum
import random


class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    NONE = 5

class Game(object):
    HEIGHT = 7#15
    WIDTH = 7#40

    class Direction(Enum):
        UP = 1
        DOWN = 2
        LEFT = 3
        RIGHT = 4

    def __init__(self, select_random_snake_and_food_positions):
        if select_random_snake_and_food_positions:
            self._generate_snake()
            self._generate_food()
        else:
            self.snake = [(4, 4), (4, 3), (4, 2)]
            self.food_at = (6, 10)

        #self.snake = [(4, 4), (4, 3), (4, 2)]
        #self._generate_snake()
        self.direction = Game.Direction.RIGHT
        #self.food_at = (6, 10)
        #self._generate_food()

        self.finished = False
        self.score = 0

    def get_height(self):
        return self.HEIGHT

    def get_width(self):
        return self.WIDTH

    def is_finished(self):
        return self.finished

    def get_score(self):
        return self.score

    def make_action(self, action):
        self._update_direction(action)

        head_at = self.snake[0]
        if self.direction == Game.Direction.UP:
            head_at = (head_at[0] - 1, head_at[1])
        elif self.direction == Game.Direction.DOWN:
            head_at = (head_at[0] + 1, head_at[1])
        elif self.direction == Game.Direction.LEFT:
            head_at = (head_at[0], head_at[1] - 1)
        else:
            head_at = (head_at[0], head_at[1] + 1)

        if head_at in self.snake[:-1]:
            self.finished = True
            return

        if head_at[0] == -1 or head_at[0] == self.get_height():
            self.finished = True
            return

        if head_at[1] == -1 or head_at[1] == self.get_width():
            self.finished = True
            return

        self.snake.insert(0, head_at)
        if head_at == self.food_at:
            self.score += 1
            self._generate_food()
        else:
            del self.snake[-1]


    def _update_direction(self, action):
        if action == Action.UP:
            self.direction = Game.Direction.UP
        elif action == Action.DOWN:
            self.direction = Game.Direction.DOWN
        elif action == Action.LEFT:
            self.direction = Game.Direction.LEFT
        elif action == Action.RIGHT:
            self.direction = Game.Direction.RIGHT

    def _generate_snake(self):
        head_at = (random.randint(0, self.get_height()-1), random.randint(0, self.get_width()-1))
        tail1_at = random.choice(
            [(head_at[0], head_at[1]-1),
            (head_at[0], head_at[1]+1),
            (head_at[0]-1, head_at[1]),
            (head_at[0]+1, head_at[1])])
        tail2_at = random.choice(
            [(tail1_at[0], tail1_at[1]-1),
            (tail1_at[0], tail1_at[1]+1),
            (tail1_at[0]-1, tail1_at[1]),
            (tail1_at[0]+1, tail1_at[1])])

        self.snake = [head_at, tail1_at, tail2_at]

        if not self._is_generated_snake_valid():
            self._generate_snake()

    def _generate_food(self):
        food = []
        while food == []:
            food = [(random.randint(0, self.get_height() - 1), random.randint(0, self.get_width() - 1))]
            if food[0] in self.snake:
                food = []
        self.food_at = food[0]

    def _is_generated_snake_valid(self):
        if self.snake[0] == self.snake[2]:
            return False

        if self._is_invalid(self.snake[1]):
            return False

        if self._is_invalid(self.snake[2]):
            return False

        return True

    def _is_invalid(self, pos):
        if pos[0] < 0 or pos[0] >= self.get_height():
            return True
        if pos[1] < 0 or pos[1] >= self.get_width():
            return True
        return False
