# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Modified for Blender 5.0 compatibility

import sys, random, os
import bpy, bpy_extras


"""
Some utility functions for interacting with Blender
Compatible with Blender 2.80+ / 4.x / 5.0
"""

# Blender version check
BLENDER_VERSION = bpy.app.version
IS_BLENDER_280_OR_LATER = BLENDER_VERSION >= (2, 80, 0)


def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
  """ Delete a specified blender object """
  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0
    for o in bpy.data.objects:
      o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.delete()
  else:
    # Blender 2.79 and earlier
    for o in bpy.data.objects:
      o.select = False
    obj.select = True
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
  """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def set_layer(obj, layer_idx):
  """
  Move an object to a particular layer.
  In Blender 2.80+, layers are replaced by collections.
  layer_idx 0 = visible (Scene Collection), layer_idx > 0 = hidden collection
  """
  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0: Use collections instead of layers
    # Unlink from all current collections
    for col in obj.users_collection:
      col.objects.unlink(obj)

    if layer_idx == 0:
      # Link to scene collection (visible)
      bpy.context.scene.collection.objects.link(obj)
    else:
      # Create or get a hidden collection
      hidden_col_name = f"HiddenLayer_{layer_idx}"
      if hidden_col_name not in bpy.data.collections:
        hidden_col = bpy.data.collections.new(hidden_col_name)
        bpy.context.scene.collection.children.link(hidden_col)
        # Exclude from view layer to hide
        bpy.context.view_layer.layer_collection.children[hidden_col_name].exclude = True
      else:
        hidden_col = bpy.data.collections[hidden_col_name]
      hidden_col.objects.link(obj)
  else:
    # Blender 2.79 and earlier
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
      obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  # First figure out how many of this object are already in the scene so we can
  # give the new object a unique name
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
  bpy.ops.wm.append(filename=filename)

  # Give it a new name to avoid conflicts
  new_name = '%s_%d' % (name, count)
  bpy.data.objects[name].name = new_name

  # Set the new object as active, then rotate, scale, and translate it
  x, y = loc
  obj = bpy.data.objects[new_name]

  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
  else:
    # Blender 2.79 and earlier
    bpy.context.scene.objects.active = obj

  bpy.context.object.rotation_euler[2] = theta
  bpy.ops.transform.resize(value=(scale, scale, scale))
  # v5 shapes have origin at bottom, so z=0 places them on ground
  bpy.ops.transform.translate(value=(x, y, 0))


def load_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn, 'NodeTree', name)
    bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
  """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  # Figure out how many materials are already in the scene
  mat_count = len(bpy.data.materials)

  if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0: Use bpy.data.materials.new() instead of bpy.ops.material.new()
    mat = bpy.data.materials.new(name='Material_%d' % mat_count)
    mat.use_nodes = True
  else:
    # Blender 2.79 and earlier
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

  # Attach the new material to the active object
  # Make sure it doesn't already have materials
  obj = bpy.context.active_object
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # Find the output node of the new material
  # In Blender 5.0, node names may be localized, so use type instead
  output_node = None
  for n in mat.node_tree.nodes:
    if n.type == 'OUTPUT_MATERIAL':
      output_node = n
      break

  if output_node is None:
    raise RuntimeError("Could not find Material Output node in material")

  # Add a new GroupNode to the node tree of the active material,
  # and copy the node tree from the preloaded node group to the
  # new group node. This copying seems to happen by-value, so
  # we can create multiple materials of the same type without them
  # clobbering each other
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
  )

