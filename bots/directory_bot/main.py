import os
from random import choice
from src.definitions import Engine, GameBoard, GameMove

ROOT_DIR = os.path.join("bots", "directory_bot")


class ChessBot(Engine):
    def __call__(self, board: GameBoard) -> GameMove:
        return choice(list(board.legal_moves))


if "__main__" in __name__:
    # normal relative file paths won't work when we run your code
    # instead, make you paths relative to the ROOT_DIR, as shown using ROOT_DIR
    with open(os.path.join(ROOT_DIR, "test_file.txt")) as file:
        print(file.read(), end="")
