"""
eval/elo.py - Elo rating tracker for agents

Usage:
    tracker = EloTracker()
    tracker.update("Q-Learning", "Random", winner="Q-Learning")
    print(tracker)
"""

DEFAULT_RATING = 1000
K_FACTOR       = 32

class EloTracker:
    """
    Tracks Elo ratings for all agents across tournament games

    Compares expected vs actual results i.e. beating a stron opp gains more 
    than beating a weak one.
    """

    def __init__(self) -> None:
        '''Maps agent name --> current Elo rating'''
        self.ratings: dict[str, float] = {}

    def _get(self, name: str) -> float:
        '''Return existing rating or initialise to default'''
        return self.ratings.setdefault(name, DEFAULT_RATING)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        '''Probability that A wins'''
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, name_a: str, name_b: str, winner: str | None) -> None:
        '''Update ratings after game'''
        ra = self._get(name_a)
        rb = self._get(name_b)

        # Expected scores
        ea = self._expected(ra, rb)
        eb = 1.0 - ea

        # Actual -> 1 win, 0 loss, 0.5 draw
        if winner == name_a:
            sa, sb = 1.0, 0.0
        elif winner == name_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[name_a] = ra + K_FACTOR * (sa - ea)
        self.ratings[name_b] = rb + K_FACTOR * (sb - eb)

    def update_match(self, name_a: str, name_b: str, wins_a: int, wins_b: int, draws: int) -> None:
        '''update after multi-game match result'''
        for _ in range(wins_a):
            self.update(name_a, name_b, winner=name_a)
        for _ in range(wins_b):
            self.update(name_a, name_b, winner=name_b)
        for _ in range(draws):
            self.update(name_a, name_b, winner=None)

    def leaderboard(self) -> list[tuple[str, float]]:
        '''Sort agents from highest to lowest'''
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def __str__(self) -> str:
        '''print leaderboard'''
        lines = ["Elo leaderboard:", "-" * 28]
        for rank, (name, rating) in enumerate(self.leaderboard(), 1):
            lines.append(f"  {rank}. {name:<16} {rating:.0f}")
        return "\n".join(lines)
