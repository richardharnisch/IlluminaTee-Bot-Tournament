from tqdm import trange

from src.engine_container import EngineContainer
from src.definitions import GameBoard
from src.chessboard import ChessBoard
from src.load_engine import load_engine


def outcome_str(board: GameBoard) -> str:
    return {False: "Black won.", True: "White won.", None: "Draw."}[board.winner()]


def outcome_float(board: GameBoard) -> float:
    return {False: 0.0, True: 1.0, None: 0.5}[board.winner()]


def watch_engines(
    board: GameBoard,
    engine_one: EngineContainer,
    engine_two: EngineContainer,
    verbose: bool = True,
) -> float:
    while not board.is_game_over():
        if verbose:
            print(board)
            print("-" * 15)
        move = engine_one(board.copy()) if board.turn else engine_two(board.copy())
        try:
            board.push(move)
        except AssertionError:
            print(
                "Illegal move made by engine:",
                engine_one.name if board.turn else engine_two.name,
            )
            break

    if verbose:
        print(board)
        print("End of game.", outcome_str(board))

    engine_one.reset()
    engine_two.reset()
    return outcome_float(board)


def play_engine(
    engine: EngineContainer,
    board: GameBoard,
    human: bool,
) -> None:
    while not board.is_game_over():
        print(board)
        if board.turn == human:
            print("input move: ", end="")
            san_move = input()
            move = board.parse_san(san_move)
            if move not in board.legal_moves:
                print("Illegal move, try again.")
                continue
        else:
            print("Engine move: ", end="", flush=True)
            move = engine(board.copy())
            print(board.san(move))

        board.push(move)

    print(board)
    print("End of game.", outcome_str(board))
    engine.reset()


def one_vs_one(
    engine_one: EngineContainer, engine_two: EngineContainer, n_games: int = 100
) -> float:
    performance = 0

    for _ in trange(n_games):
        performance += watch_engines(ChessBoard(), engine_one, engine_two, False)

    performance /= n_games
    print(f"engine_one had a win rate of {performance} with white.")
    return performance


if "__main__" in __name__:
    board = ChessBoard()
    engine_one = load_engine("directory_bot")
    engine_two = load_engine("single_file_bot")
    engine_one = EngineContainer(engine_one, 300)
    engine_two = EngineContainer(engine_two, 300)

    human_play = False
    if human_play:
        play_engine(engine_two, board, False)
    else:
        print("running match")
        one_vs_one(engine_two, engine_one, 4)
        one_vs_one(engine_one, engine_two, 4)
        # watch_engines(board, engine_one=engine_one, engine_two=engine_two)
