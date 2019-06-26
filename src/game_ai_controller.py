from time import sleep
import curses
import random

from game import Action
from game import Game
from a2c import Model
from a2c import A2CAgent
from learning_environment import LearningEnvironment


class GameAIController(object):
    def learn(self, stdscr):
        self._initialize_screen(stdscr)
        win = self._create_window()

        model = Model()
        agent = A2CAgent(model)
        learning_environment = LearningEnvironment()

        agent.load_model_if_previously_saved(learning_environment)

        for iter in range(200):
            if iter % 10 == 0:
                self._play_test_game(learning_environment, agent, win)

            agent.train(learning_environment)

        agent.save_model()

    def play(self, stdscr):
        self._initialize_screen(stdscr)
        win = self._create_window()

        model = Model()
        agent = A2CAgent(model)
        learning_environment = LearningEnvironment()
        agent.load_pretrained_model(learning_environment)

        self._play_test_game(learning_environment, agent, win)
        self._draw_game_over_text_and_wait_for_input(win)

    def _initialize_screen(self, stdscr):
        curses.curs_set(False)
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        stdscr.clear()

    def _create_window(self):
        height = Game.HEIGHT + 2  # adding 2 for border
        width = Game.WIDTH + 2  # adding 2 for border

        win = curses.newwin(height, width, 0, 0)
        win.attrset(curses.color_pair(4))

        return win

    def _play_test_game(self, learning_environment, agent, win):
        game = Game(select_random_snake_and_food_positions=True)
        steps_without_scoring = 0
        while steps_without_scoring < 200 and not game.is_finished():
            obs = learning_environment.build_observation(game)
            action = agent.select_action(obs)

            # simple guard to play longer
            action = self._action_if_not_dying_or_random_action(game, action)

            score_before = game.get_score()
            game.make_action(action)
            score_after = game.get_score()

            if score_after > score_before:
                steps_without_scoring = 0

            self._draw_game(win, game)
            steps_without_scoring += 1

    def _action_if_not_dying_or_random_action(self, game, default_action):
        if self._action_not_dying(game, default_action):
            return default_action

        good_actions = []
        if self._is_free(game, (game.snake[0][0]-1, game.snake[0][1])):
            good_actions.append(Action.UP)
        if self._is_free(game, (game.snake[0][0]+1, game.snake[0][1])):
            good_actions.append(Action.DOWN)
        if self._is_free(game, (game.snake[0][0], game.snake[0][1]-1)):
            good_actions.append(Action.LEFT)
        if self._is_free(game, (game.snake[0][0], game.snake[0][1]+1)):
            good_actions.append(Action.RIGHT)

        if not good_actions:
            return default_action

        return random.choice(good_actions)

    def _action_not_dying(self, game, action):
        x = game.snake[0][0]
        y = game.snake[0][1]

        if action == Action.UP:
            x -= 1
        if action == Action.DOWN:
            x += 1
        if action == Action.LEFT:
            y -= 1
        if action == Action.RIGHT:
            y += 1

        new_head_at = (x, y)
        return self._is_free(game, new_head_at)

    def _is_free(self, game, loc):
        if loc[0] < 0 or loc[0] >= game.get_height():
            return False
        if loc[1] < 0 or loc[1] >= game.get_width():
            return False

        if loc in game.snake:
            return False

        return True

    def _draw_game(self, win, g):
        win.clear()

        # draw border and print score
        win.border(0)
        win.addstr(0, int(g.get_width() / 2 - 2), ' SNAKE ', curses.color_pair(1))
        win.addstr(int(g.get_height()) + 1, 1, 'Pts:' + str(g.get_score()), curses.color_pair(1))

        # draw snake
        for p in g.snake:
            win.addstr(p[0] + 1, p[1] + 1, '#', curses.color_pair(3))
        win.addstr(g.snake[0][0] + 1, g.snake[0][1] + 1, '#', curses.color_pair(2))

        # draw food
        win.addstr(g.food_at[0] + 1, g.food_at[1] + 1, '*', curses.color_pair(5) | curses.A_BOLD)

        win.refresh()
        sleep(0.1)

    def _draw_game_over_text_and_wait_for_input(self, win):
        win.addstr(Game.HEIGHT / 2, Game.WIDTH / 2 - 3, 'GAME OVER', curses.color_pair(4) | curses.A_BOLD)
        win.addstr(Game.HEIGHT / 2 + 1, Game.WIDTH / 2 - 6, 'PRESS SPACE KEY', curses.color_pair(4) | curses.A_BOLD)

        win.nodelay(False)

        key = win.getch()
        while key != ord(' '):
            key = win.getch()
