import curses

from game import Game
from game import Action


class GameController(object):
    def play_game(self, stdscr):
        curses.curs_set(False)
        curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        stdscr.clear()

        g = Game(select_random_snake_and_food_positions=False)

        height = g.get_height() + 2  # adding 2 for border
        width = g.get_width() + 2  # adding 2 for border
        win = curses.newwin(height, width, 0, 0)
        win.attrset(curses.color_pair(4))
        win.keypad(True)
        win.nodelay(True)
        win.timeout(150)

        while not g.is_finished():
            self._draw_game(win, g)
            action = self._read_action_from_input(win)
            g.make_action(action)

        win.addstr(g.get_height() / 2 - 1, g.get_width() / 2 - 3, 'GAME OVER', curses.color_pair(4) | curses.A_BOLD)
        win.addstr(g.get_height() / 2, g.get_width() / 2 - 6, 'PRESS SPACE KEY', curses.color_pair(4) | curses.A_BOLD)

        win.nodelay(False)

        key = win.getch()
        while key != ord(' '):
            key = win.getch()

    def _read_action_from_input(self, win):
        key = win.getch()

        action = Action.NONE
        if key == curses.KEY_UP:
            action = Action.UP
        elif key == curses.KEY_DOWN:
            action = Action.DOWN
        elif key == curses.KEY_LEFT:
            action = Action.LEFT
        elif key == curses.KEY_RIGHT:
            action = Action.RIGHT

        return action

    def _draw_game(self, win, g):
        win.clear()

        # draw border and print score
        win.border(False)
        win.addstr(0, g.get_width() / 2 - 2, ' SNAKE ', curses.color_pair(1))
        win.addstr(0, 2, ' Score: ' + str(g.get_score()) + ' ', curses.color_pair(1))

        # draw snake
        for p in g.snake:
            win.addch(p[0] + 1, p[1] + 1, '#', curses.color_pair(3))
        win.addch(g.snake[0][0] + 1, g.snake[0][1] + 1, '#', curses.color_pair(2))

        # draw food
        win.addch(g.food_at[0] + 1, g.food_at[1] + 1, '*', curses.color_pair(5) | curses.A_BOLD)
