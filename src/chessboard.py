from __future__ import annotations

from typing import Optional
from chess import Board
from torch import Tensor

from src.definitions import GameBoard


class BullyCoverMembersException(Exception):
    """
    Exception raised when cover members need to get off their ass.
    """

    def __init__(self, msg=""):
        self.msg = msg


class ChessBoard(Board, GameBoard):
    def winner(self) -> Optional[bool]:
        outcome = self.outcome()
        if outcome is None:
            return None
        return outcome.winner

    def get_state(self) -> Tensor:
        raise BullyCoverMembersException(
            "It is so much more fun if you implement this functionality yourself! :)"
        )
