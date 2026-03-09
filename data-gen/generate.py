#!/usr/bin/env python3
"""
ordinary-bench data generation entry point.

Usage:
    python generate.py                            # use config.toml (70 scenes)
    python generate.py --preset test              # 1 scene/split, fast test
    python generate.py --config my.toml           # custom config
    python generate.py --preset test --dry-run    # just print config
    python generate.py --workers 4                # parallel Blender processes
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:
  try:
    import tomli as tomllib
  except ModuleNotFoundError:
    tomllib = None

import pipeline

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# test preset: 1 scene per object count, low quality, for quick validation
PRESETS = {
  "test": {
    "splits": {
      "n04": {"n_scenes": 1, "min_objects": 4, "max_objects": 4},
      "n05": {"n_scenes": 1, "min_objects": 5, "max_objects": 5},
      "n06": {"n_scenes": 1, "min_objects": 6, "max_objects": 6},
      "n07": {"n_scenes": 1, "min_objects": 7, "max_objects": 7},
      "n08": {"n_scenes": 1, "min_objects": 8, "max_objects": 8},
      "n09": {"n_scenes": 1, "min_objects": 9, "max_objects": 9},
      "n10": {"n_scenes": 1, "min_objects": 10, "max_objects": 10},
    },
    "rendering": {"samples": 64},
  },
}

DEFAULT_CONFIG = {
  "blender": {
    "executable": "blender",
    "use_gpu": False,
  },
  "rendering": {
    "width": 480,
    "height": 320,
    "samples": 256,
    "n_views": 4,
    "camera_distance": 12.0,
    "elevation": 30.0,
    "azimuth_start": 45.0,
  },
  "objects": {
    "min_count": 4,
    "max_count": 10,
    "min_dist": 0.25,
    "margin": 0.4,
  },
  "output": {
    "dir": "./output",
    "seed": 42,
  },
  "splits": {},
}


def deep_merge(base, override):
  """Recursively merge override into base (override wins)."""
  result = dict(base)
  for k, v in override.items():
    if k in result and isinstance(result[k], dict) and isinstance(v, dict):
      result[k] = deep_merge(result[k], v)
    else:
      result[k] = v
  return result


def load_config(config_path, preset, cli_args):
  """Load config from TOML file, apply preset, then CLI overrides."""
  cfg = dict(DEFAULT_CONFIG)

  # Layer 1: TOML file
  if config_path:
    p = Path(config_path)
    if not p.exists():
      logger.error(f"Config file not found: {p}")
      sys.exit(1)
    if tomllib is None:
      logger.error(
        "TOML support requires Python 3.11+ or 'pip install tomli'"
      )
      sys.exit(1)
    with open(p, "rb") as f:
      file_cfg = tomllib.load(f)
    cfg = deep_merge(cfg, file_cfg)

  # Layer 2: preset (replaces splits entirely)
  if preset:
    if preset not in PRESETS:
      logger.error(f"Unknown preset: {preset}. Choose from: {list(PRESETS)}")
      sys.exit(1)
    cfg = deep_merge(cfg, PRESETS[preset])

  # Layer 3: CLI overrides
  if cli_args.blender:
    cfg["blender"]["executable"] = cli_args.blender
  if cli_args.output_dir:
    cfg["output"]["dir"] = cli_args.output_dir
  if cli_args.gpu is not None:
    cfg["blender"]["use_gpu"] = cli_args.gpu

  return cfg


def create_directories(cfg):
  """Create output directory structure."""
  output = Path(cfg["output"]["dir"])
  for sub in ["images/single_view", "images/multi_view", "scenes", "splits"]:
    (output / sub).mkdir(parents=True, exist_ok=True)


def _run_split(args_tuple):
  """Worker function for parallel execution."""
  split_name, split_cfg, cfg = args_tuple
  return split_name, pipeline.build_split(split_name, split_cfg, cfg)


def main():
  parser = argparse.ArgumentParser(
    description="Generate ordinary-bench dataset (scenes + rendered images)"
  )
  parser.add_argument(
    "--config", "-c", default=None,
    help="Path to TOML config file (default: config.toml in script dir)",
  )
  parser.add_argument(
    "--preset", "-p", choices=list(PRESETS.keys()), default=None,
    help="Size preset (overrides split counts)",
  )
  parser.add_argument(
    "--blender", default=None,
    help="Blender executable path (overrides config)",
  )
  parser.add_argument(
    "--output-dir", "-o", default=None,
    help="Output directory (overrides config)",
  )
  parser.add_argument(
    "--gpu", action="store_true", default=None,
    help="Enable GPU rendering",
  )
  parser.add_argument(
    "--workers", "-w", type=int, default=1,
    help="Number of parallel Blender processes (default: 1)",
  )
  parser.add_argument(
    "--start-idx", type=int, default=0,
    help="Scene index offset for incremental generation (e.g. 10 to add scenes 10-19)",
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="Print resolved config and exit",
  )

  args = parser.parse_args()

  # If no explicit config given, try config.toml in script directory
  config_path = args.config
  if config_path is None:
    default_toml = Path(__file__).resolve().parent / "config.toml"
    if default_toml.exists():
      config_path = str(default_toml)

  cfg = load_config(config_path, args.preset, args)

  # Inject start_idx into all splits for incremental generation
  if args.start_idx > 0:
    for split_cfg in cfg["splits"].values():
      split_cfg["start_idx"] = args.start_idx

  if not cfg["splits"]:
    logger.error("No splits configured. Check config.toml or use --preset.")
    sys.exit(1)

  if args.dry_run:
    print(json.dumps(cfg, indent=2))
    return

  # Validate Blender executable
  blender = cfg["blender"]["executable"]
  if blender == "blender":
    logger.warning(
      "Using default 'blender' command. "
      "Set blender.executable in config.toml or use --blender."
    )

  n_splits = len(cfg["splits"])
  total_scenes = sum(s["n_scenes"] for s in cfg["splits"].values())
  workers = min(args.workers, n_splits)

  logger.info("=" * 50)
  logger.info("ORDINARY-BENCH Data Generation")
  logger.info("=" * 50)
  logger.info(f"Output:  {cfg['output']['dir']}")
  logger.info(f"Blender: {blender}")
  logger.info(f"Splits:  {n_splits} ({total_scenes} scenes total)")
  logger.info(f"Workers: {workers}")

  create_directories(cfg)

  all_stats = {}

  if workers <= 1:
    # Sequential
    for split_name, split_cfg in cfg["splits"].items():
      logger.info(f"\n--- Split: {split_name} ---")
      stats = pipeline.build_split(split_name, split_cfg, cfg)
      all_stats[split_name] = stats
  else:
    # Parallel
    tasks = [
      (name, scfg, cfg) for name, scfg in cfg["splits"].items()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
      futures = {pool.submit(_run_split, t): t[0] for t in tasks}
      for future in as_completed(futures):
        name = futures[future]
        try:
          _, stats = future.result()
          all_stats[name] = stats
          logger.info(f"Split '{name}' done: {stats['n_scenes']} scenes")
        except Exception as e:
          logger.error(f"Split '{name}' failed: {e}")
          raise

  pipeline.save_dataset_info(cfg, all_stats)

  # Summary
  total_scenes = sum(s["n_scenes"] for s in all_stats.values())
  total_images = sum(
    s["n_single_view_images"] + s["n_multi_view_images"]
    for s in all_stats.values()
  )
  print(f"\nDone! {total_scenes} scenes, {total_images} images")
  print(f"Output: {cfg['output']['dir']}")


if __name__ == "__main__":
  main()
