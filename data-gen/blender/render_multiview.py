# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Extended for multi-view rendering support

"""
Multi-view rendering script for ORDINAL-SPATIAL benchmark.

Renders the same scene from multiple camera viewpoints for multi-view
spatial reasoning evaluation.

Usage:
    blender --background --python render_multiview.py -- [arguments]

Example:
    blender --background --python render_multiview.py -- \
        --num_images 10 \
        --n_views 4 \
        --output_dir ../output/multiview/
"""

from __future__ import print_function
import math
import sys
import random
import argparse
import json
import os
from datetime import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

INSIDE_BLENDER = True
try:
    import bpy
    import bpy_extras
    from mathutils import Vector, Euler
    BLENDER_VERSION = bpy.app.version
    IS_BLENDER_280_OR_LATER = BLENDER_VERSION >= (2, 80, 0)
except ImportError:
    INSIDE_BLENDER = False
    IS_BLENDER_280_OR_LATER = False
    BLENDER_VERSION = (0, 0, 0)

if INSIDE_BLENDER:
    try:
        import utils
    except ImportError:
        # Try to add the script directory to sys.path and retry.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            import utils
        except ImportError:
            print("\nERROR: Cannot import utils.py")
            print(f"Tried adding script dir to sys.path: {script_dir}")
            print("Add image_generation to Blender's Python path.")
            sys.exit(1)


@dataclass
class CameraConfig:
    """Configuration for a single camera viewpoint."""
    camera_id: str
    azimuth: float      # Azimuth angle in degrees (0 = +X direction)
    elevation: float    # Elevation angle in degrees (0 = horizontal)
    distance: float     # Distance from scene center
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_cartesian(self) -> Tuple[float, float, float]:
        """
        Convert spherical coordinates to Cartesian.

        Returns:
            (x, y, z) camera position
        """
        azimuth_rad = math.radians(self.azimuth)
        elevation_rad = math.radians(self.elevation)

        x = self.distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = self.distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = self.distance * math.sin(elevation_rad)

        return (
            x + self.look_at[0],
            y + self.look_at[1],
            z + self.look_at[2]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        pos = self.to_cartesian()
        return {
            "camera_id": self.camera_id,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "position": list(pos),
            "look_at": list(self.look_at)
        }


@dataclass
class MultiViewConfig:
    """Configuration for multi-view rendering."""
    n_views: int = 4
    camera_distance: float = 12.0
    elevation: float = 30.0
    azimuth_start: float = 45.0  # Start at 45 degrees for better coverage
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def generate_cameras(self) -> List[CameraConfig]:
        """
        Generate camera configurations for all viewpoints.

        Returns:
            List of CameraConfig objects
        """
        cameras = []
        azimuth_step = 360.0 / self.n_views

        for i in range(self.n_views):
            azimuth = self.azimuth_start + i * azimuth_step
            # Normalize to [0, 360)
            azimuth = azimuth % 360.0

            cameras.append(CameraConfig(
                camera_id=f"view_{i}",
                azimuth=azimuth,
                elevation=self.elevation,
                distance=self.camera_distance,
                look_at=self.look_at
            ))

        return cameras


def get_object_by_name(name: str, alternative_names: Optional[List[str]] = None):
    """Get Blender object by name with fallbacks."""
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    if alternative_names:
        for alt_name in alternative_names:
            if alt_name in bpy.data.objects:
                return bpy.data.objects[alt_name]
    raise KeyError(f"Object not found: {name}")


def set_camera_position(camera_config: CameraConfig) -> None:
    """
    Set Blender camera position and orientation.

    Args:
        camera_config: Camera configuration with position parameters
    """
    camera = bpy.data.objects['Camera']
    position = camera_config.to_cartesian()
    look_at = camera_config.look_at

    # Set position
    camera.location = position

    # Calculate direction and set rotation
    direction = Vector(look_at) - Vector(position)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def compute_pixel_coords_for_view(camera, objects_3d: List[Dict]) -> List[Dict]:
    """
    Compute pixel coordinates for all objects from current camera view.

    Args:
        camera: Blender camera object
        objects_3d: List of object dictionaries with 3d_coords

    Returns:
        List of objects with updated pixel_coords
    """
    updated_objects = []
    for obj in objects_3d:
        obj_copy = obj.copy()
        coords_3d = obj['3d_coords']
        pixel_coords = utils.get_camera_coords(camera, Vector(coords_3d))
        obj_copy['pixel_coords'] = pixel_coords
        updated_objects.append(obj_copy)
    return updated_objects


def compute_directions_for_view(camera) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute cardinal directions relative to current camera view.

    Args:
        camera: Blender camera object

    Returns:
        Dictionary mapping direction names to vectors
    """
    # Create temporary plane to get ground normal
    if IS_BLENDER_280_OR_LATER:
        bpy.ops.mesh.primitive_plane_add(size=10)
    else:
        bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object
    plane_normal = plane.data.vertices[0].normal

    if IS_BLENDER_280_OR_LATER:
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    else:
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete temporary plane
    utils.delete_object(plane)

    return {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
        'above': tuple(plane_up),
        'below': tuple(-plane_up)
    }


def render_single_view(
    camera_config: CameraConfig,
    output_image: str,
    objects_3d: List[Dict],
    args
) -> Dict[str, Any]:
    """
    Render scene from a single camera viewpoint.

    Args:
        camera_config: Camera configuration
        output_image: Output image path
        objects_3d: List of objects with 3D coordinates
        args: Command line arguments

    Returns:
        View metadata dictionary
    """
    # Set camera position
    set_camera_position(camera_config)

    camera = bpy.data.objects['Camera']

    # Compute view-specific data
    objects_with_pixels = compute_pixel_coords_for_view(camera, objects_3d)
    directions = compute_directions_for_view(camera)

    # Set output path and render
    bpy.context.scene.render.filepath = output_image
    bpy.ops.render.render(write_still=True)

    # Build view metadata
    view_data = {
        "view_id": camera_config.camera_id,
        "image_path": os.path.basename(output_image),
        "camera": camera_config.to_dict(),
        "directions": directions,
        "objects": objects_with_pixels
    }

    return view_data


def render_multiview_scene(
    args,
    num_objects: int,
    output_index: int,
    output_split: str,
    output_dir: str,
    mv_config: MultiViewConfig
) -> Dict[str, Any]:
    """
    Render a complete multi-view scene.

    Args:
        args: Command line arguments
        num_objects: Number of objects to place
        output_index: Scene index
        output_split: Dataset split name
        output_dir: Output directory for this scene
        mv_config: Multi-view configuration

    Returns:
        Complete scene metadata
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load base scene
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render settings
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100

    # GPU settings
    if args.use_gpu == 1:
        if IS_BLENDER_280_OR_LATER:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            for compute_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
                try:
                    cycles_prefs.compute_device_type = compute_type
                    for device in cycles_prefs.devices:
                        device.use = True
                    break
                except:
                    continue
        bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.blur_glossy = 2.0

    if IS_BLENDER_280_OR_LATER:
        bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    # Initialize scene structure
    scene_id = f"{output_split}_{output_index:06d}"
    scene_struct = {
        "scene_id": scene_id,
        "split": output_split,
        "image_index": output_index,
        "n_objects": num_objects,
        "objects": [],
        "world_constraints": {},
        "views": []
    }

    # Add random jitter to lights
    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    lamp_key = get_object_by_name('Lamp_Key', ['Light_Key', 'Key', 'KeyLight'])
    lamp_back = get_object_by_name('Lamp_Back', ['Light_Back', 'Back', 'BackLight'])
    lamp_fill = get_object_by_name('Lamp_Fill', ['Light_Fill', 'Fill', 'FillLight'])

    if args.key_light_jitter > 0:
        for i in range(3):
            lamp_key.location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            lamp_back.location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            lamp_fill.location[i] += rand(args.fill_light_jitter)

    # Place objects (using first camera position for initial setup)
    cameras = mv_config.generate_cameras()
    set_camera_position(cameras[0])

    # Create temporary plane for direction computation
    if IS_BLENDER_280_OR_LATER:
        bpy.ops.mesh.primitive_plane_add(size=10)
    else:
        bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal

    if IS_BLENDER_280_OR_LATER:
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    else:
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()

    # Store directions for object placement
    temp_directions = {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
    }

    utils.delete_object(plane)

    # Add objects
    objects_3d, blender_objects = add_random_objects(
        temp_directions, num_objects, args, camera
    )

    scene_struct["objects"] = objects_3d

    # Render from each viewpoint
    for cam_config in cameras:
        img_path = os.path.join(output_dir, f"{cam_config.camera_id}.png")

        view_data = render_single_view(
            cam_config,
            img_path,
            objects_3d,
            args
        )

        scene_struct["views"].append(view_data)

    # Save scene metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    # Also render single-view image (view_0) to single_view directory
    if args.output_single_view_dir:
        single_view_dir = args.output_single_view_dir
        os.makedirs(single_view_dir, exist_ok=True)
        single_img_path = os.path.join(single_view_dir, f"{scene_id}.png")

        # Copy view_0 image or re-render
        import shutil
        view0_path = os.path.join(output_dir, "view_0.png")
        if os.path.exists(view0_path):
            shutil.copy(view0_path, single_img_path)

    return scene_struct


def add_random_objects(directions, num_objects, args, camera, _retry_count=0):
    """
    Add random objects to the scene.
    Adapted from render_images.py with retry limit.

    Placement area scales with number of objects to accommodate dense scenes.
    """
    MAX_RETRIES = 100  # Increased for dense scenes
    if _retry_count >= MAX_RETRIES:
        raise RuntimeError(f"Failed to place objects after {MAX_RETRIES} attempts")

    # Scale placement area based on number of objects
    # Base area: 3.0 for up to 6 objects
    # For more objects, scale up to accommodate density
    if num_objects <= 6:
        placement_range = 3.0
    elif num_objects <= 10:
        placement_range = 3.5
    else:
        # For 11-15 objects, use larger area
        placement_range = 4.0

    # Adjust spacing for dense scenes
    effective_min_dist = args.min_dist
    effective_margin = args.margin
    if num_objects > 10:
        # Reduce spacing requirements for very dense scenes
        effective_min_dist = max(0.15, args.min_dist * 0.7)
        effective_margin = max(0.25, args.margin * 0.7)

    # Load properties
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = list(properties['sizes'].items())

    positions = []
    objects = []
    blender_objects = []

    for i in range(num_objects):
        size_name, r = random.choice(size_mapping)

        # Try to place object
        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(
                    directions, num_objects, args, camera,
                    _retry_count=_retry_count + 1
                )

            x = random.uniform(-placement_range, placement_range)
            y = random.uniform(-placement_range, placement_range)

            # Check distances
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < effective_min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = directions[direction_name]
                    margin_val = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin_val < effective_margin:
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # Choose random color and shape
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        if obj_name == 'Cube':
            r /= math.sqrt(2)

        theta = 360.0 * random.random()

        # Add object
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Add material
        mat_name, mat_name_out = random.choice(material_mapping)
        utils.add_material(mat_name, Color=rgba)

        # Record object data
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            "id": f"obj_{i}",
            "shape": obj_name_out,
            "size": size_name,
            "material": mat_name_out,
            "3d_coords": tuple(obj.location),
            "rotation": theta,
            "pixel_coords": pixel_coords,
            "color": color_name,
        })

    return objects, blender_objects


def main(args):
    """Main entry point for multi-view rendering."""
    print(f"Starting multi-view rendering: {args.num_images} scenes, {args.n_views} views each")

    # Create multi-view config
    mv_config = MultiViewConfig(
        n_views=args.n_views,
        camera_distance=args.camera_distance,
        elevation=args.elevation,
        azimuth_start=args.azimuth_start
    )

    # Create output directories
    multiview_dir = os.path.join(args.output_dir, "multi_view")
    single_view_dir = os.path.join(args.output_dir, "single_view")
    os.makedirs(multiview_dir, exist_ok=True)
    os.makedirs(single_view_dir, exist_ok=True)

    args.output_single_view_dir = single_view_dir

    all_scenes = []
    successful = 0
    failed = 0

    for i in range(args.num_images):
        scene_id = f"{args.split}_{(i + args.start_idx):06d}"
        scene_output_dir = os.path.join(multiview_dir, scene_id)

        num_objects = random.randint(args.min_objects, args.max_objects)

        try:
            scene_struct = render_multiview_scene(
                args,
                num_objects=num_objects,
                output_index=(i + args.start_idx),
                output_split=args.split,
                output_dir=scene_output_dir,
                mv_config=mv_config
            )
            all_scenes.append(scene_struct)
            successful += 1
            print(f"  [{successful}/{args.num_images}] Rendered {scene_id}")

        except Exception as e:
            print(f"  [ERROR] Failed to render {scene_id}: {e}")
            failed += 1
            continue

    print(f"\nRendering complete: {successful} successful, {failed} failed")

    # Save combined scenes file
    output_file = os.path.join(args.output_dir, f"{args.split}_scenes.json")
    output_data = {
        "info": {
            "date": dt.today().strftime("%Y-%m-%d"),
            "split": args.split,
            "n_views": args.n_views,
            "camera_config": asdict(mv_config)
        },
        "scenes": all_scenes
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved scenes to {output_file}")


# Argument parser
parser = argparse.ArgumentParser(description="Multi-view scene rendering")

# Input options
parser.add_argument('--base_scene_blendfile', default='assets/base_scene_v5.blend')
parser.add_argument('--properties_json', default='assets/properties.json')
parser.add_argument('--shape_dir', default='assets/shapes_v5')
parser.add_argument('--material_dir', default='assets/materials_v5')

# Object settings
parser.add_argument('--min_objects', default=4, type=int)
parser.add_argument('--max_objects', default=10, type=int)
parser.add_argument('--min_dist', default=0.25, type=float)
parser.add_argument('--margin', default=0.4, type=float)
parser.add_argument('--max_retries', default=50, type=int)

# Multi-view settings
parser.add_argument('--n_views', default=4, type=int,
    help="Number of camera viewpoints")
parser.add_argument('--camera_distance', default=12.0, type=float,
    help="Camera distance from scene center")
parser.add_argument('--elevation', default=30.0, type=float,
    help="Camera elevation angle in degrees")
parser.add_argument('--azimuth_start', default=45.0, type=float,
    help="Starting azimuth angle in degrees")

# Output settings
parser.add_argument('--output_dir', default='../output/multiview/',
    help="Output directory for rendered scenes")
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--split', default='train')

# Render settings
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=320, type=int)
parser.add_argument('--render_num_samples', default=256, type=int)
parser.add_argument('--render_max_bounces', default=8, type=int)
parser.add_argument('--key_light_jitter', default=1.0, type=float)
parser.add_argument('--fill_light_jitter', default=1.0, type=float)
parser.add_argument('--back_light_jitter', default=1.0, type=float)


if __name__ == '__main__':
    if INSIDE_BLENDER:
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('Run from Blender:')
        print('  blender --background --python render_multiview.py -- [args]')
        print()
        print('For help:')
        print('  python render_multiview.py --help')
