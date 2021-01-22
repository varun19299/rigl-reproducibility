from dataclasses import dataclass


@dataclass
class SmoothenValue(object):
    """
    Create a exponentially smooth moving average for a value (loss, etc) using `beta`.
    """

    beta: float = 0.9
    n: int = 0

    def __post_init__(self):
        self._mov_avg = 0.0

    def add_value(self, val: float) -> None:
        """
        Add `val` to calculate updated smoothed value.

        :param val: value to add
        :type val: float
        """
        self.n += 1
        self._mov_avg = self.beta * self._mov_avg + (1 - self.beta) * val
        self.smooth = self._mov_avg / (1 - self.beta ** self.n)


@dataclass
class AverageValue(object):
    """
    Create a simple running average for a value (loss, etc).
    """
    n: int = 0

    def __post_init__(self):
        self._mov_avg = 0.0

    def add_value(self, val: float):
        """
        Add `val` to calculate updated smoothed value.

        :param val: value to add
        :type val: float
        """
        self._mov_avg = (val + self.n * self._mov_avg) / (self.n + 1)
        self.smooth = self._mov_avg
        self.n += 1
