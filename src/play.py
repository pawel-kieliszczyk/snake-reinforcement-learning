import curses
from menu import Menu


def main(stdscr):
    menu = Menu()
    menu.show(stdscr)


curses.wrapper(main)
