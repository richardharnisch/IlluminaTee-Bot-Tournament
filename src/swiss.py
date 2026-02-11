from dataclasses import dataclass, field


@dataclass
class PlayerRecord:
    name: str
    score: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    opponents: list[str] = field(default_factory=list)


def swiss_pairings(
    standings: dict[str, PlayerRecord], round_num: int
) -> list[tuple[str, str | None]]:
    """
    Generates Swiss-style pairings: sort by score descending, then pair
    adjacent players while avoiding rematches. If the player count is odd,
    the lowest-ranked unpaired player receives a bye (None).
    """
    players = sorted(
        standings.values(), key=lambda p: (-p.score, p.name)
    )
    names = [p.name for p in players]
    paired: set[str] = set()
    pairings: list[tuple[str, str | None]] = []

    for i, name in enumerate(names):
        if name in paired:
            continue
        # Find the best available opponent (next in ranking, no rematch)
        opponent = None
        for j in range(i + 1, len(names)):
            candidate = names[j]
            if candidate in paired:
                continue
            if candidate in standings[name].opponents:
                continue
            opponent = candidate
            break

        if opponent is None:
            # No valid opponent found — try anyone unpaired (allow rematch)
            for j in range(i + 1, len(names)):
                candidate = names[j]
                if candidate not in paired:
                    opponent = candidate
                    break

        if opponent is not None:
            pairings.append((name, opponent))
            paired.add(name)
            paired.add(opponent)
        else:
            # This player gets a bye
            pairings.append((name, None))
            paired.add(name)

    return pairings


def compute_buchholz(standings: dict[str, PlayerRecord]) -> dict[str, float]:
    """
    Computes Buchholz tiebreak scores: the sum of each opponent's score.
    """
    buchholz: dict[str, float] = {}
    for name, record in standings.items():
        buchholz[name] = sum(
            standings[opp].score for opp in record.opponents if opp in standings
        )
    return buchholz
