"""DSL module for ordinal spatial relations."""

from .comparators import Comparator, compare
from .predicates import (
  MetricType, QRRConstraint, TRRConstraint,
  compute_qrr, compute_trr,
  extract_all_qrr, extract_all_trr,
)
