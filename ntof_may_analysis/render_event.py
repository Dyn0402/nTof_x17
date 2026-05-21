#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on May 21 5:43 PM 2026
Created in PyCharm
Created as nTof_x17/render_event.py

@author: Dylan Neff, dylan
"""

import bpy
import mathutils
import json
import os

# --- CONFIGURATION ---
EVENT_DIR = "/home/dylan/PycharmProjects/nTof_x17/ntof_may_analysis/output/run_51/hv_scan_drift_600_resist_530/event_display"

GEO_FILE = os.path.join(EVENT_DIR, "detector_geo_100.glb")
TRACK_FILE = os.path.join(EVENT_DIR, "track_100.json")
RENDER_OUTPUT = os.path.join(EVENT_DIR, "cinematic_render.png")

SCENE_CENTER_BLN = (0, 100, 50)


def lab_to_blender(lab_x, lab_y, lab_z):
    """Convert lab coordinates to Blender world coordinates."""
    return (lab_x, -lab_z, lab_y)


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'

    new_world = bpy.data.worlds.new("DarkWorld")
    new_world.use_nodes = True
    bg_node = new_world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.002, 0.004, 0.012, 1.0)
        bg_node.inputs[1].default_value = 0.0
    scene.world = new_world


def add_lights():
    target = mathutils.Vector(SCENE_CENTER_BLN)

    # Key light
    bpy.ops.object.light_add(type='SUN', location=(400, -400, 600))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.data.color = (1.0, 0.95, 0.88)
    sun.data.use_shadow = False  # ELIMINATE SHADOWS
    direction = target - sun.location
    sun.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # Soft fill
    bpy.ops.object.light_add(type='AREA', location=(-500, 100, 150))
    fill = bpy.context.object
    fill.data.energy = 1000.0
    fill.data.size = 400.0
    fill.data.color = (0.4, 0.6, 1.0)
    fill.data.use_shadow = False  # ELIMINATE SHADOWS


def create_glass_material(name, color, alpha=0.1):
    """Helper to create highly transparent EEVEE materials."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    mat.use_transparent_shadow = True  # Don't cast opaque shadows through transparent faces

    principled = mat.node_tree.nodes.get("Principled BSDF")
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = 0.1
    principled.inputs['Alpha'].default_value = alpha
    return mat


def import_and_style_geometry():
    if not os.path.exists(GEO_FILE):
        print(f"Warning: Could not find {GEO_FILE}")
        return

    bpy.ops.import_scene.gltf(filepath=GEO_FILE)

    # 1. Define Distinct Materials (RGBA)
    mat_mm = create_glass_material("Mat_MM", (0.05, 0.2, 0.6, 1.0), alpha=0.12)    # Blue
    mat_strip = create_glass_material("Mat_Strip", (0.6, 0.3, 0.05, 1.0), alpha=0.12)  # Orange
    mat_large = create_glass_material("Mat_Large", (0.6, 0.05, 0.05, 1.0), alpha=0.10)  # Red

    # Capsule: solid-looking dark carbon grey — needs higher alpha to be visible on black bg
    mat_capsule = bpy.data.materials.new(name="Mat_Capsule")
    mat_capsule.use_nodes = True
    mat_capsule.blend_method = 'OPAQUE'   # fully opaque — carbon is not transparent
    cap_p = mat_capsule.node_tree.nodes.get("Principled BSDF")
    cap_p.inputs['Base Color'].default_value = (0.08, 0.08, 0.08, 1.0)  # very dark graphite
    cap_p.inputs['Metallic'].default_value = 0.0
    cap_p.inputs['Roughness'].default_value = 0.6   # matte carbon surface

    # 2. Identify objects based on their physical bounding box width (Blender X / Lab X)
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            dim_x = obj.dimensions.x

            # Use X-dimension mapping to assign colors
            if dim_x > 350:
                mat_to_apply = mat_mm  # MM Drift is 400mm wide
            elif 250 < dim_x < 350:
                mat_to_apply = mat_large  # Large Scintillator is 300mm wide
            elif dim_x < 30:
                mat_to_apply = mat_strip  # Scintillator strips are 25mm wide
            elif 30 <= dim_x <= 50:
                mat_to_apply = mat_capsule  # Capsule is 40mm wide (20mm radius)
            else:
                mat_to_apply = mat_mm  # Fallback

            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat_to_apply)
            else:
                obj.data.materials[0] = mat_to_apply


def import_and_style_tracks():
    if not os.path.exists(TRACK_FILE):
        print(f"Warning: Could not find {TRACK_FILE}")
        return

    with open(TRACK_FILE, 'r') as f:
        data = json.load(f)

    for track_name, coords in data.items():
        curve_data = bpy.data.curves.new(name=track_name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 1.0  # THINNER TRACK (1 mm radius)
        curve_data.use_fill_caps = True

        polyline = curve_data.splines.new('POLY')
        polyline.points.add(len(coords) - 1)

        for i, coord in enumerate(coords):
            bx, by, bz = lab_to_blender(coord[0], coord[1], coord[2])
            polyline.points[i].co = (bx, by, bz, 1)

        track_obj = bpy.data.objects.new(track_name, curve_data)
        bpy.context.collection.objects.link(track_obj)

        glow_mat = bpy.data.materials.new(name="TrackGlow")
        glow_mat.use_nodes = True
        principled = glow_mat.node_tree.nodes.get("Principled BSDF")
        principled.inputs['Emission Color'].default_value = (0.1, 1.0, 0.2, 1.0)
        principled.inputs['Emission Strength'].default_value = 3.0

        track_obj.data.materials.append(glow_mat)


def setup_camera_and_render():
    cam_loc = (380, -430, 220)
    target = mathutils.Vector(SCENE_CENTER_BLN)

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam

    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 26

    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.filepath = RENDER_OUTPUT

    print(f"Rendering to {RENDER_OUTPUT}...")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    setup_scene()
    add_lights()
    import_and_style_geometry()
    import_and_style_tracks()
    setup_camera_and_render()