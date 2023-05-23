#!/usr/bin/env python3
"""Normal"""


class Normal:
    """Representation of Normal Distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal Distribution"""
        if data is None:
            self.mean = float(mean)
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float(
                (sum((x - self.mean) ** 2 for x in data) / len(data))
                ** 0.5
            )
