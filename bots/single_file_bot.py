from random import choice
from src.definitions import Engine, GameBoard, GameMove


class ChessBot(Engine):
    def __call__(self, board: GameBoard) -> GameMove:
        return choice(list(board.legal_moves))
