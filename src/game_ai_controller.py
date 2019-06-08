from time import sleep
import curses

from game import Action
from game import Game
from a2c import Model
from a2c import A2CAgent
from learning_environment import LearningEnvironment


class GameAIController(object):
    def learn(self, stdscr):
        # init screen
        curses.curs_set(False)
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        stdscr.clear()

        # create window
        height = Game.HEIGHT + 2  # adding 2 for border
        width = Game.WIDTH + 2  # adding 2 for border
        win = curses.newwin(height, width, 0, 0)
        win.attrset(curses.color_pair(4))

        model = Model()
        agent = A2CAgent(model)
        learning_environment = LearningEnvironment()

        agent.load_model_if_previously_saved(learning_environment)

        for _ in xrange(100):
            agent.train(learning_environment)
            self._play_test_game(learning_environment, agent, win)

        agent.save_model()

    def _play_test_game(self, learning_environment, agent, win):
        game = Game(select_random_snake_and_food_positions=True)
        steps = 0
        while steps < 100 and not game.is_finished():
            obs = learning_environment.build_observation(game)
            action = agent.select_action(obs)
            game.make_action(action)

            self._draw_game(win, game, learning_environment.get_number_of_played_games())
            steps += 1

    def _draw_game(self, win, g, games_played):
        win.clear()

        # draw border and print score
        win.border(False)
        win.addstr(0, g.get_width() / 2 - 2, ' SNAKE ', curses.color_pair(1))
        win.addstr(0, 2, ' Score: ' + str(g.get_score()) + ' ', curses.color_pair(1))
        win.addstr(g.get_height()+1, 2, ' Games played: ' + str(games_played) + ' ', curses.color_pair(1))

        # draw snake
        for p in g.snake:
            win.addstr(p[0] + 1, p[1] + 1, '#', curses.color_pair(3))
        win.addstr(g.snake[0][0] + 1, g.snake[0][1] + 1, '#', curses.color_pair(2))

        # draw food
        win.addstr(g.food_at[0] + 1, g.food_at[1] + 1, '*', curses.color_pair(5) | curses.A_BOLD)

        win.refresh()
        sleep(0.33)
