"""
GT constraint extraction from data-gen scene JSON.

Parses scene JSON objects into the format expected by dsl/predicates.py,
then extracts all QRR and TRR ground-truth constraints.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from dsl.predicates import (
    MetricType, extract_all_qrr, extract_all_trr,
)


def parse_objects(scene: dict) -> Dict[str, Dict]:
    """
    Convert scene JSON objects to the dict format for DSL predicates.

    Scene JSON format (from data-gen):
      {"id": "obj_0", "shape": "sphere", "size": "large",
       "material": "rubber", "color": "brown",
       "3d_coords": [-1.14, 0.45, 0.0],
       "pixel_coords": [287, 150, 12.42],
       "rotation": 105.1}

    Returns:
      {obj_id: {"position_3d": [...], "position_2d": [...], "depth": ..., "size": ..., ...}}
    """
    objects = {}
    for obj in scene.get("objects", []):
        obj_id = obj["id"]
        coords_3d = obj.get("3d_coords", [0, 0, 0])
        pixel = obj.get("pixel_coords", [0, 0, 0])

        objects[obj_id] = {
            "id": obj_id,
            "shape": obj.get("shape", ""),
            "color": obj.get("color", ""),
            "size": obj.get("size", "medium"),
            "material": obj.get("material", ""),
            "position_3d": coords_3d,
            "3d_coords": coords_3d,
            "position_2d": pixel[:2],
            "pixel_coords": pixel,
            "depth": pixel[2] if len(pixel) > 2 else 0.0,
            "rotation": obj.get("rotation", 0.0),
        }
    return objects


def object_description(obj: dict) -> str:
    """Build a human-readable description like 'large brown rubber sphere'."""
    parts = []
    if obj.get("size"):
        parts.append(str(obj["size"]))
    if obj.get("color"):
        parts.append(obj["color"])
    if obj.get("material"):
        parts.append(obj["material"])
    if obj.get("shape"):
        parts.append(obj["shape"])
    return " ".join(parts)


def extract_gt(scene: dict, tau: float = 0.10) -> dict:
    """
    Extract all GT constraints from a scene.

    Returns:
      {"qrr": [QRRConstraint.to_dict(), ...], "trr": [TRRConstraint.to_dict(), ...]}
    """
    objects = parse_objects(scene)
    if len(objects) < 2:
        return {"qrr": [], "trr": []}

    # QRR: only dist3D (disjoint pairs)
    qrr_constraints = extract_all_qrr(
        objects, MetricType.DIST_3D, tau=tau, disjoint_only=True
    )
    qrr_list = [c.to_dict() for c in qrr_constraints]

    # TRR: use 3D coordinates (view-invariant)
    trr_constraints = extract_all_trr(objects, use_3d=True)
    trr_list = [c.to_dict() for c in trr_constraints]

    return {"qrr": qrr_list, "trr": trr_list}


def load_scene(path: str) -> dict:
    """Load a scene JSON file."""
    with open(path) as f:
        return json.load(f)
