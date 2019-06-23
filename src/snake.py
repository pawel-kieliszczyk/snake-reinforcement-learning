import curses
import os

from menu import Menu


def main(stdscr):
    menu = Menu()
    menu.show(stdscr)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # will suppress tensorflow warnings
curses.wrapper(main)
