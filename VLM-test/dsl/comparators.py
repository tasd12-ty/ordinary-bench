"""
基于容差的序关系比较代数。

Comparator Algebra:
    - a <_tau b  iff  a < b * (1 - tau)
    - a ~=_tau b iff  |a - b| <= tau * max(a, b)
    - a >_tau b  iff  a > b * (1 + tau)
"""

from enum import Enum
from dataclasses import dataclass


class Comparator(Enum):
    LT = "<"
    APPROX = "~="
    GT = ">"

    def __str__(self) -> str:
        return self.value

    @property
    def ordinal(self) -> int:
        return {Comparator.LT: 0, Comparator.APPROX: 1, Comparator.GT: 2}[self]

    def flip(self) -> "Comparator":
        if self == Comparator.LT:
            return Comparator.GT
        elif self == Comparator.GT:
            return Comparator.LT
        return Comparator.APPROX

    @classmethod
    def from_string(cls, s: str) -> "Comparator":
        s = s.strip()
        mapping = {
            "<": cls.LT, "~=": cls.APPROX, "≈": cls.APPROX,
            "=": cls.APPROX, ">": cls.GT,
            "lt": cls.LT, "eq": cls.APPROX, "approx": cls.APPROX, "gt": cls.GT,
        }
        if s.lower() in mapping:
            return mapping[s.lower()]
        raise ValueError(f"Unknown comparator: {s}")


def compare(a: float, b: float, tau: float = 0.10) -> Comparator:
    if tau <= 0 or tau >= 1:
        raise ValueError(f"Tolerance tau must be in (0, 1), got {tau}")
    if a < 0 or b < 0:
        raise ValueError(f"Values must be non-negative, got a={a}, b={b}")
    if a == 0 and b == 0:
        return Comparator.APPROX
    if a == 0:
        return Comparator.LT
    if b == 0:
        return Comparator.GT

    max_val = max(a, b)
    threshold = tau * max_val
    diff = a - b

    if abs(diff) <= threshold:
        return Comparator.APPROX
    elif diff < 0:
        return Comparator.LT
    else:
        return Comparator.GT


