import sys
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
    live: bool = False,
) -> float:
    """
    Play a single game between two engines.
    Prints to the stdout if verbose is set to True.
    When live is True, shows an updating board display with move counter and clocks.
    """
    lines_on_screen = 0
    last_san = ""

    def render(final=False):
        nonlocal lines_on_screen
        # Move cursor up to overwrite previous display
        if lines_on_screen > 0:
            sys.stdout.write(f"\033[{lines_on_screen}A\r")

        ply = board.ply()

        # Header line
        header = f"  {engine_one.name} (W) vs {engine_two.name} (B)"
        if final:
            result_code = {True: "1-0", False: "0-1", None: "1/2-1/2"}[board.winner()]
            header += f" \u2014 {result_code}"

        # Status line
        parts = [f"Ply {ply}"]
        if last_san:
            parts.append(last_san)
        parts.append(f"W: {engine_one.remaining_time:.1f}s")
        parts.append(f"B: {engine_two.remaining_time:.1f}s")
        if final:
            parts.append(outcome_str(board))
        status = "  " + " | ".join(parts)

        # Board lines (indented)
        board_str = str(board)
        board_lines = ["  " + line for line in board_str.split("\n")]

        all_lines = [header, status] + board_lines
        for line in all_lines:
            sys.stdout.write(f"\033[2K{line}\n")
        sys.stdout.flush()

        lines_on_screen = len(all_lines)

    if live:
        render()
    elif verbose:
        print(f"Playing game: '{engine_one.name}' vs. '{engine_two.name}'")

    try:
        while not board.is_game_over():
            if verbose and not live:
                print(board)
                print("-" * 15)

            board_copy = board.copy()
            current_turn = board.turn
            move = engine_one(board_copy) if current_turn else engine_two(board_copy)

            # Compute SAN before pushing
            try:
                ply = board.ply()
                move_num = ply // 2 + 1
                san_str = board.san(move)
                if current_turn:  # white's move
                    last_san = f"{move_num}. {san_str}"
                else:
                    last_san = f"{move_num}... {san_str}"
            except (ValueError, AssertionError):
                last_san = str(move)

            try:
                board.push(move)
            except AssertionError:
                print(
                    "Illegal move made by engine:",
                    engine_one.name if board.turn else engine_two.name,
                )
                break

            if live:
                render()
    except Exception:
        # Ensure cursor is past the live display before exception propagates
        if live and lines_on_screen > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        raise

    if live:
        render(final=True)
        print()  # blank line separator
    elif verbose:
        print(board)
        print("End of game.", outcome_str(board))

    engine_one.reset(board.copy())
    engine_two.reset(board.copy())
    return outcome_float(board)


def play_engine(
    engine: EngineContainer,
    board: GameBoard,
    human: bool,
) -> None:
    """
    Play a game between a human and an engine, using stdin and stdout.
    The `human` argument decides which side the human plays.
    """
    while not board.is_game_over():
        print(board)
        if board.turn == human:
            print("input move: ", end="")
            san_move = input()
            try:
                move = board.parse_san(san_move)
            except Exception:
                print("Illegal move, try again.")
                continue
        else:
            print("Engine move: ", end="", flush=True)
            move = engine(board.copy())
            print(board.san(move))

        board.push(move)

    print(board)
    print("End of game.", outcome_str(board))
    engine.reset(board.copy())


def one_vs_one(
    engine_one: EngineContainer,
    engine_two: EngineContainer,
    n_games: int = 100,
) -> float:
    """
    Plays several games between two engines and reports the outcome.
    """
    performance = 0

    for _ in trange(n_games):
        performance += watch_engines(
            ChessBoard(),
            engine_one,
            engine_two,
            False,
        )

    performance /= n_games
    print(
        f"Engine: '{engine_one.name}' had a point win rate",
        f"of {performance:.4f} over '{engine_two.name}' with white.",
    )
    return performance


if "__main__" in __name__:
    board = ChessBoard()
    engine_one = load_engine("directory_bot")
    engine_two = load_engine("single_file_bot")
    engine_one = EngineContainer(engine_one, 300)
    engine_two = EngineContainer(engine_two, 300)

    human_play = True
    if human_play:
        play_engine(engine_one, board, False)
    else:
        print("running match")
        one_vs_one(engine_two, engine_one, 4)
        one_vs_one(engine_one, engine_two, 4)
        # watch_engines(board, engine_one=engine_one, engine_two=engine_two)
