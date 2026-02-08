from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator, Optional
from torch import Tensor

GameMove = Any


class GameBoard(ABC):
    """
    A standardisation of two-player turn based games.
    """

    legal_moves: Generator
    move_stack: list[GameMove]
    turn: bool

    def __init__(self) -> None:
        """
        Initialisation of a position amounts to randomly selecting an element \
        from the set of all possible opening positions. For games like \
        chess, go, and shogi there is only one opening position, and as \
        such, these games will always initialise to the same positions.
        """
        pass

    @abstractmethod
    def copy(self) -> GameBoard:
        """This is a simple copy"""
        pass

    @abstractmethod
    def push(self, move) -> None:
        """
        Makes a move, or more formally, pushes a move onto the move stack.
        """
        pass

    @abstractmethod
    def pop(self) -> GameMove:
        """
        Reverts the most recent move by popping a move from the move stack.
        """
        pass

    @abstractmethod
    def san(self, move: GameMove) -> str:
        """
        The name is short for Standard Algebraic Notation.
        In general, this is a human readable version of a move.
        """
        pass

    @abstractmethod
    def parse_san(self, move: str) -> GameMove:
        """
        The inverse of the `san` method. Takes a string in the \
        Standard Algebraic Notation of the given board, and \
        outputs the `GameMove` that corresponds to it.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Checks if the game is over according to the rules.
        """
        pass

    @abstractmethod
    def ply(self) -> int:
        """
        Number of moves on the move stack. This is different than the
        """
        pass

    @abstractmethod
    def winner(self) -> Optional[bool]:
        """
        Returns `True` if the first player (the player that starts) \
        has won, and False if they lost. Additionally, it returns a \
        None if there is a tie, or the game has not ended.
        """
        pass

    @abstractmethod
    def get_state(self) -> Tensor:
        """
        Returns some representation of the state of the board as a \
        Tensor. The shape of this representation will be consistent \
        across all elements of the state space.
        """
        pass


class Engine(ABC):
    """
    An engine is a function which returns a move given a position.

    Engines, unlike evaluation functions, give the chosen move for the \
    side at play, and are therefore turn dependent. The assumption may \
    be made that an Engine object will never receive an invalid position, \
    or an impossible position given the rules of the game. It will also \
    never receive a position in which there are no legal moves.
    """

    name = ""

    def __init__(self, opening_position: GameBoard) -> None:
        """
        Note that the engine will be initialised with an opening \
        position of the game. This argument can be used to determine \
        the shape of elements in the state space, or learn other things \
        about the environment.
        """
        super().__init__()
        # `name` is the only reserved attribute for an Engine

    @abstractmethod
    def __call__(self, board: GameBoard) -> GameMove:
        pass

    def update(self, game: GameBoard) -> None:
        """
        A method called before the model is removed from working memory.
        Use this method to save anything you need to allow the model to \
        get back to the state it was next time it gets loaded. 

        The `game` argument is the final position of the game. 
        You can use this to see how the game went and learn from it.
        """
        pass
