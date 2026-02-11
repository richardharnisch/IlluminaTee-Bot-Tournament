import argparse
import json
import math
import os
import sys
from datetime import datetime

from src.chessboard import ChessBoard
from src.engine_container import EngineContainer, TimeoutException
from src.load_engine import load_engine
from src.play import watch_engines, outcome_str
from src.swiss import PlayerRecord, swiss_pairings, compute_buchholz

BOT_DIR = "bots"


def discover_bots() -> list[str]:
    """Scans the bots/ directory for valid bots (mirrors load_engine logic)."""
    bots: list[str] = []
    for entry in sorted(os.listdir(BOT_DIR)):
        path = os.path.join(BOT_DIR, entry)
        if os.path.isdir(path):
            if os.path.isfile(os.path.join(path, "main.py")):
                bots.append(entry)
        elif entry.endswith(".py") and not entry.startswith("_"):
            bots.append(entry.removesuffix(".py"))
    return bots


class Tournament:
    def __init__(
        self,
        bot_names: list[str],
        num_rounds: int | None = None,
        seconds: float = 60.0,
        seconds_increment: float = 0.0,
        games_per_match: int = 2,
        verbose: bool = False,
    ):
        self.bot_names = bot_names
        self.num_rounds = num_rounds or max(1, math.ceil(math.log2(len(bot_names))))
        self.seconds = seconds
        self.seconds_increment = seconds_increment
        self.games_per_match = games_per_match
        self.verbose = verbose
        self.standings: dict[str, PlayerRecord] = {
            name: PlayerRecord(name=name) for name in bot_names
        }
        self.round_results: list[list[dict]] = []

    def run(self) -> None:
        print(f"Swiss Tournament: {len(self.bot_names)} bots, {self.num_rounds} rounds")
        print(f"Time control: {self.seconds}s + {self.seconds_increment}s increment")
        print(f"Games per match: {self.games_per_match}")
        print()

        for round_num in range(1, self.num_rounds + 1):
            print(f"=== Round {round_num}/{self.num_rounds} ===")
            self._run_round(round_num)
            self.print_standings()
            print()

        print("=== Final Standings ===")
        self.print_standings()

    def _run_round(self, round_num: int) -> None:
        pairings = swiss_pairings(self.standings, round_num)
        round_games: list[dict] = []

        for white_name, black_name in pairings:
            if black_name is None:
                self._handle_bye(white_name)
                round_games.append({
                    "white": white_name,
                    "black": None,
                    "result": "bye",
                    "score": 1.0,
                })
                continue

            match_results = self._play_match(white_name, black_name)
            round_games.extend(match_results)

        self.round_results.append(round_games)

    def _play_match(self, player_a: str, player_b: str) -> list[dict]:
        """Plays games_per_match games between two players, alternating colors."""
        results: list[dict] = []

        for game_num in range(self.games_per_match):
            if game_num % 2 == 0:
                white_name, black_name = player_a, player_b
            else:
                white_name, black_name = player_b, player_a

            use_live = sys.stdout.isatty()

            if not use_live:
                print(f"  {white_name} (W) vs {black_name} (B) ... ", end="", flush=True)

            try:
                white_class = load_engine(white_name)
                black_class = load_engine(black_name)
                white_engine = EngineContainer(white_class, self.seconds, self.seconds_increment)
                black_engine = EngineContainer(black_class, self.seconds, self.seconds_increment)
                board = ChessBoard()
                score = watch_engines(
                    board, white_engine, black_engine, self.verbose, live=use_live,
                )
            except TimeoutException as e:
                if use_live:
                    print(f"  Timeout: {e.msg}")
                else:
                    print(f"Timeout: {e.msg}")
                # Determine who timed out from the message
                if white_name in e.msg:
                    score = 0.0  # black wins
                else:
                    score = 1.0  # white wins
            except Exception as e:
                if use_live:
                    print(f"  Error: {e}")
                else:
                    print(f"Error: {e}")
                # Default: treat as draw on crash
                score = 0.5

            result_str = {1.0: "1-0", 0.0: "0-1", 0.5: "1/2-1/2"}.get(score, "1/2-1/2")
            if not self.verbose and not use_live:
                print(result_str)

            self._update_standings(white_name, black_name, score)
            results.append({
                "white": white_name,
                "black": black_name,
                "result": result_str,
                "score": score,
            })

        return results

    def _update_standings(self, white: str, black: str, score: float) -> None:
        w = self.standings[white]
        b = self.standings[black]

        if score == 1.0:
            w.score += 1.0
            w.wins += 1
            b.losses += 1
        elif score == 0.0:
            b.score += 1.0
            b.wins += 1
            w.losses += 1
        else:
            w.score += 0.5
            w.draws += 1
            b.score += 0.5
            b.draws += 1

        w.opponents.append(black)
        b.opponents.append(white)

    def _handle_bye(self, player: str) -> None:
        print(f"  {player} receives a bye (+1.0)")
        record = self.standings[player]
        record.score += 1.0
        record.wins += 1

    def print_standings(self) -> None:
        buchholz = compute_buchholz(self.standings)
        sorted_players = sorted(
            self.standings.values(),
            key=lambda p: (-p.score, -buchholz[p.name], p.name),
        )

        header = f"{'Rank':<5} {'Name':<25} {'Score':>6} {'W-D-L':>7} {'Buchholz':>9}"
        print(header)
        print("-" * len(header))
        for i, p in enumerate(sorted_players, 1):
            wdl = f"{p.wins}-{p.draws}-{p.losses}"
            print(f"{i:<5} {p.name:<25} {p.score:>6.1f} {wdl:>7} {buchholz[p.name]:>9.1f}")

    def save_results(self, filepath: str) -> None:
        buchholz = compute_buchholz(self.standings)
        sorted_players = sorted(
            self.standings.values(),
            key=lambda p: (-p.score, -buchholz[p.name], p.name),
        )

        data = {
            "tournament": {
                "date": datetime.now().isoformat(),
                "num_rounds": self.num_rounds,
                "seconds": self.seconds,
                "seconds_increment": self.seconds_increment,
                "games_per_match": self.games_per_match,
                "bots": self.bot_names,
            },
            "standings": [
                {
                    "rank": i,
                    "name": p.name,
                    "score": p.score,
                    "wins": p.wins,
                    "draws": p.draws,
                    "losses": p.losses,
                    "buchholz": buchholz[p.name],
                    "opponents": p.opponents,
                }
                for i, p in enumerate(sorted_players, 1)
            ],
            "rounds": [
                {"round": r + 1, "games": games}
                for r, games in enumerate(self.round_results)
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a local Swiss chess bot tournament"
    )
    parser.add_argument(
        "--list-bots", action="store_true",
        help="List available bots and exit",
    )
    parser.add_argument(
        "-r", "--rounds", type=int, default=None,
        help="Number of rounds (default: ceil(log2(num_bots)))",
    )
    parser.add_argument(
        "-t", "--time", type=float, default=60.0,
        help="Time control in seconds (default: 60)",
    )
    parser.add_argument(
        "-i", "--increment", type=float, default=0.0,
        help="Time increment in seconds (default: 0)",
    )
    parser.add_argument(
        "-b", "--bots", nargs="+", default=None,
        help="Bot names to include (default: all discovered bots)",
    )
    parser.add_argument(
        "-g", "--games-per-match", type=int, default=2,
        help="Games per match pairing (default: 2)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="tournament_results.json",
        help="Output JSON file path (default: tournament_results.json)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print board states during games",
    )
    args = parser.parse_args()

    available = discover_bots()

    if args.list_bots:
        print("Available bots:")
        for bot in available:
            print(f"  - {bot}")
        return

    bot_names = args.bots if args.bots else available

    if len(bot_names) < 2:
        print("Error: Need at least 2 bots for a tournament.", file=sys.stderr)
        sys.exit(1)

    for name in bot_names:
        if name not in available:
            print(f"Error: Bot '{name}' not found in {BOT_DIR}/", file=sys.stderr)
            sys.exit(1)

    tournament = Tournament(
        bot_names=bot_names,
        num_rounds=args.rounds,
        seconds=args.time,
        seconds_increment=args.increment,
        games_per_match=args.games_per_match,
        verbose=args.verbose,
    )
    tournament.run()
    tournament.save_results(args.output)


if "__main__" in __name__:
    main()
