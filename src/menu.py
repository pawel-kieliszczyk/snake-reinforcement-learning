import curses

from game import Game
from game_ai_controller import GameAIController
from game_controller import GameController


class Menu(object):
    def show(self, stdscr):
        while True:
            curses.curs_set(False)
            curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
            stdscr.clear()

            win = curses.newwin(10, 40, 0, 0)
            win.attrset(curses.color_pair(4))
            win.keypad(True)
            win.nodelay(False)

            win.addstr(0, 0, 'Welcome to Snake game!', curses.color_pair(3))
            win.addstr(1, 0, 'Press:', curses.color_pair(3))
            win.addstr(2, 0, '\'1\' (to play the game)', curses.color_pair(3))
            win.addstr(3, 0, '\'2\' (to watch AI learning to play)', curses.color_pair(3))
            win.addstr(4, 0, '\'3\' (to watch trained AI playing)', curses.color_pair(3))
            win.addstr(5, 0, '\'4\' (to exit)', curses.color_pair(3))

            key = win.getch()

            while key not in [ord('1'), ord('2'), ord('3'), ord('4')]:
                key = win.getch()

            win.clear()
            win.refresh()

            if key == ord('1'):
                game_controller = GameController()
                game_controller.play_game(stdscr)
            elif key == ord('2'):
                game_ai_controller = GameAIController()
                game_ai_controller.learn(stdscr)
            elif key == ord('3'):
                game_ai_controller = GameAIController()
                game_ai_controller.play(stdscr)
                pass
            elif key == ord('4'):
                break
