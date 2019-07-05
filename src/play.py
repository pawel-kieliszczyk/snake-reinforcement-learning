import os
import curses

from a2c import Model, A2CAgent
from learning_environment import LearningEnvironment
from game_controller import GameController


def main(stdscr):
    model = Model()
    agent = A2CAgent(model)
    learning_environment = LearningEnvironment()

    agent.initialize_model(learning_environment)
    agent.load_model_if_previously_saved()

    game_controller = GameController(stdscr)
    game_controller.play(agent, learning_environment)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress TensorFlow warnings
curses.wrapper(main)

