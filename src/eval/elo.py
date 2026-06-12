"""
eval/elo.py — Lightweight rating tracker for agents.

For the three-player project, Elo is mostly kept as an optional utility.
The main evaluation output uses win counts for P1, P2, P3, and draws.
"""

DEFAULT_RATING = 1000
K_FACTOR = 32


class EloTracker:
    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}

    def _get(self, name: str) -> float:
        return self.ratings.setdefault(name, DEFAULT_RATING)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_pair(self, name_a: str, name_b: str, score_a: float) -> None:
        """Update A vs B with score_a = 1 win, 0.5 draw, 0 loss."""
        ra = self._get(name_a)
        rb = self._get(name_b)
        ea = self._expected(ra, rb)
        eb = 1.0 - ea
        score_b = 1.0 - score_a
        self.ratings[name_a] = ra + K_FACTOR * (score_a - ea)
        self.ratings[name_b] = rb + K_FACTOR * (score_b - eb)

    def update_three_player_result(self, names: list[str], winner: str | None) -> None:
        """Approximate three-player Elo by updating all pairwise matchups."""
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if winner is None:
                    self.update_pair(a, b, 0.5)
                elif winner == a:
                    self.update_pair(a, b, 1.0)
                elif winner == b:
                    self.update_pair(a, b, 0.0)
                else:
                    # If a third player won, these two both lost relative to winner,
                    # so treat their pairwise result as a draw.
                    self.update_pair(a, b, 0.5)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def __str__(self) -> str:
        lines = ["Elo leaderboard:", "-" * 28]
        for rank, (name, rating) in enumerate(self.leaderboard(), 1):
            lines.append(f"  {rank}. {name:<18} {rating:.0f}")
        return "\n".join(lines)
