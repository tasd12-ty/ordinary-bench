"""
Question enumeration and batch splitting.

Enumerates all QRR and TRR questions from GT constraints,
assigns question IDs, and splits into configurable batches.
"""

from itertools import combinations, permutations
from typing import Dict, List

from dsl.predicates import (
    MetricType, compute_qrr, compute_trr, _is_boundary, METRIC_FUNCTIONS,
)


def enumerate_qrr(
    objects: Dict[str, Dict],
    tau: float = 0.10,
    metric: MetricType = MetricType.DIST_3D,
) -> List[dict]:
    """
    Enumerate all disjoint QRR questions with GT answers.

    For N objects, disjoint QRR count = C(N,2) * C(N-2,2) / 2
    (divide by 2 because (pair1,pair2) and (pair2,pair1) are the same question)

    Returns list of question dicts.
    """
    obj_ids = sorted(objects.keys())
    pairs = list(combinations(obj_ids, 2))
    metric_func = METRIC_FUNCTIONS[metric]
    questions = []
    qid = 0

    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1:]:
            # Disjoint check
            if set(pair1) & set(pair2):
                continue
            # Skip boundary cases
            m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
            m2 = metric_func(objects[pair2[0]], objects[pair2[1]])
            if _is_boundary(m1, m2, tau):
                continue

            constraint = compute_qrr(objects, pair1, pair2, metric, tau)
            qid += 1
            questions.append({
                "qid": f"qrr_{qid:04d}",
                "type": "qrr",
                "pair1": list(constraint.pair1),
                "pair2": list(constraint.pair2),
                "metric": str(metric),
                "gt_comparator": str(constraint.comparator),
            })

    return questions


def enumerate_trr(
    objects: Dict[str, Dict],
    use_3d: bool = True,
) -> List[dict]:
    """
    Enumerate all TRR questions with GT answers.

    For N objects, TRR count = P(N,3) = N*(N-1)*(N-2)

    Returns list of question dicts.
    """
    obj_ids = sorted(objects.keys())
    questions = []
    qid = 0

    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple
        constraint = compute_trr(objects, target, ref1, ref2, use_3d)
        qid += 1
        questions.append({
            "qid": f"trr_{qid:04d}",
            "type": "trr",
            "target": constraint.target,
            "ref1": constraint.ref1,
            "ref2": constraint.ref2,
            "gt_hour": constraint.hour,
            "gt_quadrant": constraint.quadrant,
            "gt_angle_deg": round(constraint.angle_deg, 2),
        })

    return questions


def make_batches(questions: List[dict], batch_size: int) -> List[dict]:
    """
    Split questions into batches.

    Returns list of batch dicts with batch_id and questions.
    """
    batches = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        batches.append({
            "batch_id": len(batches),
            "n_questions": len(chunk),
            "questions": chunk,
        })
    return batches


def question_counts(n_objects: int) -> dict:
    """
    Calculate expected question counts for N objects.

    QRR (disjoint): C(N,2) pairs, then choose disjoint pair from remaining.
    TRR: P(N,3) = N*(N-1)*(N-2)
    """
    n_pairs = n_objects * (n_objects - 1) // 2
    # Disjoint QRR: pairs from C(N,2), only those sharing no objects
    qrr_count = 0
    ids = list(range(n_objects))
    pairs = list(combinations(ids, 2))
    for i, p1 in enumerate(pairs):
        for p2 in pairs[i + 1:]:
            if not (set(p1) & set(p2)):
                qrr_count += 1

    trr_count = n_objects * (n_objects - 1) * (n_objects - 2)

    return {
        "n_objects": n_objects,
        "n_qrr": qrr_count,
        "n_trr": trr_count,
        "total": qrr_count + trr_count,
    }
