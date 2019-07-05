from time import sleep
import curses

from game import Game
from a2c import Model, A2CAgent
from learning_environment import LearningEnvironment


class GameController(object):
    def __init__(self, stdscr):
        self.stdscr = stdscr

    def play(self, agent, learning_environment):
        self._initialize_screen(self.stdscr)
        win = self._create_window()
        self._play_test_game(learning_environment, agent, win)

    def _initialize_screen(self, stdscr):
        curses.curs_set(False)
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE,curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        stdscr.clear()

    def _create_window(self):
        height = Game.HEIGHT + 2 # adding 2 for border
        width = Game.WIDTH + 2 # adding 2 for border

        win = curses.newwin(height, width, 0, 0)
        win.attrset(curses.color_pair(4))

        return win

    def _play_test_game(self, learning_environment, agent, win):
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

            self._draw_game(win, game)

        sleep(1)
        self._draw_game_over_text_and_wait_for_input(win)

    def _draw_game(self, win, game):
        win.clear()

        # draw border and print score
        win.border(0)
        win.addstr(0, int(game.get_width() / 2 - 4), ' SNAKE AI ', curses.color_pair(1))
        score_str = ' ' + str(game.get_score()) + ' '
        win.addstr(int(game.get_height() + 1), int(game.get_width() - len(score_str) + 1), score_str, curses.color_pair(1))

        # draw food
        win.addstr(game.food_at[0] + 1, game.food_at[1] + 1, '*', curses.color_pair(5) | curses.A_BOLD)

        # draw snake
        for p in game.snake:
            win.addstr(p[0] + 1, p[1] + 1, '#', curses.color_pair(3))
        win.addstr(game.snake[0][0] + 1, game.snake[0][1] + 1, '#', curses.color_pair(2))

        win.refresh()
        sleep(0.1)

    def _draw_game_over_text_and_wait_for_input(self, win):
        win.addstr(Game.HEIGHT / 2 - 1, Game.WIDTH / 2 - 3, 'GAME OVER', curses.color_pair(4) | curses.A_BOLD)
        win.addstr(Game.HEIGHT / 2 + 1, Game.WIDTH / 2 - 1, 'PRESS', curses.color_pair(4) | curses.A_BOLD)
        win.addstr(Game.HEIGHT / 2 + 2, Game.WIDTH / 2 - 3, 'SPACE KEY', curses.color_pair(4) | curses.A_BOLD)

        win.nodelay(False)

        key = win.getch()
        while key != ord(' '):
            key = win.getch()
