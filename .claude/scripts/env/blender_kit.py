"""Scene-authoring kit that runs *inside* Blender.

Load it once per Blender session, then call its functions from subsequent
`execute_blender_code` calls:

    exec(open('/abs/path/.claude/scripts/env/blender_kit.py').read())
    mrs_reset_scene()
    add_table('table', size=(0.90, 1.20, 0.025), top_z=0.63)

Sending this file through `exec` rather than pasting its contents into every
tool call keeps the socket payloads small and means the helpers cannot drift
between calls.

Why a tagging convention at all
-------------------------------
Blender knows nothing about mass, friction or degrees of freedom, and a mesh
alone does not say whether a box is a table (welded), an envelope (free) or a
drawer (one prismatic joint). Every object therefore carries `mrs_*` custom
properties, which `export_scene_graph()` writes out and `build_env.py` turns
into a `SceneSpec`. An untagged object is exported as static decor, which is
the safe default: it will be visible and collidable but nothing will try to
actuate it.

Units are metres. The kit sets the scene's unit scale on reset and refuses to
export if it has been changed, because a scene authored in centimetres compiles
into a MuJoCo model that looks correct and behaves nothing like it.
"""

import json
import math
import os

import bpy
import bmesh  # noqa: F401  (imported for interactive use)
from mathutils import Vector

MRS_KIT_VERSION = 1

# Recognised values for the `mrs_role` custom property.
ROLES = (
    "static",     # welded to the world: tables, bins, walls, fixtures
    "free",       # six-DoF rigid body: the things the robot manipulates
    "hinged",     # one revolute DoF relative to its parent
    "sliding",    # one prismatic DoF relative to its parent
    "mocap",      # kinematically scripted, unaffected by contact
    "robot_mount",  # empty/marker giving the arm's base pose
    "camera",     # marker for a review or policy camera
    "decor",      # visual only, no collision
    "ignore",     # skipped entirely by the exporter
)

COLLISION_SHAPES = ("box", "cylinder", "sphere", "capsule", "mesh", "none")


# ---------------------------------------------------------------------------
# Session setup
# ---------------------------------------------------------------------------


def mrs_reset_scene(clear=True, unit_scale=1.0):
    """Start from an empty metric scene."""
    if clear:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras):
            for block in list(collection):
                if block.users == 0:
                    collection.remove(block)

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = unit_scale
    scene.unit_settings.length_unit = "METERS"
    return {"cleared": bool(clear), "unit_scale": scene.unit_settings.scale_length}


def tag(obj, role="static", **properties):
    """Attach the `mrs_*` metadata the exporter reads.

    Any keyword becomes `mrs_<key>`. The ones `build_env.py` understands:

        mass, density, friction (3 floats), condim, solref (2 floats),
        collision ('box'|'cylinder'|'sphere'|'capsule'|'mesh'|'none'),
        joint_axis (3 floats), joint_range (2 floats), joint_damping,
        joint_stiffness, actuator ('position'|'velocity'|'intvelocity'|'motor'),
        actuator_kp, actuator_kv, actuator_ctrlrange (2 floats),
        parent_body (name), tags (comma-separated), spawn_x/spawn_y/spawn_yaw
        (2 floats each), camera_role ('scene'|'wrist'|'inspection'), fovy
    """
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
    obj["mrs_role"] = role
    for key, value in properties.items():
        if isinstance(value, (tuple, list)):
            value = [float(v) for v in value]
        obj[f"mrs_{key}"] = value
    return obj


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _finish(obj, name, colour, role, properties):
    obj.name = name
    if colour is not None:
        set_colour(obj, colour)
    tag(obj, role=role, **properties)
    return obj


def add_box(name, size, location=(0, 0, 0), rotation=(0, 0, 0), colour=None, role="static", **kw):
    """`size` is the full extent in metres (not MuJoCo half-extents).

    `dimensions` is set explicitly and the view layer flushed, rather than
    trusting `scale` alone. Setting scale on an object whose mesh datablock was
    just recycled — which happens after `objects.remove` — can leave
    `dimensions` reporting the unit cube, and the exporter reads `dimensions`.
    A body that silently compiles as a 1 m cube is very hard to spot in a
    render and hangs an enormous inertia off whatever it is attached to.
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.scale = (size[0], size[1], size[2])
    bpy.context.view_layer.update()
    obj.dimensions = (size[0], size[1], size[2])
    bpy.context.view_layer.update()
    if max(abs(obj.dimensions[i] - size[i]) for i in range(3)) > 1e-6:
        raise RuntimeError(
            f"{name}: dimensions came out {tuple(obj.dimensions)}, expected {tuple(size)}"
        )
    return _finish(obj, name, colour, role, kw)


def add_cylinder(name, radius, depth, location=(0, 0, 0), rotation=(0, 0, 0),
                 colour=None, role="static", **kw):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                        location=location, rotation=rotation, vertices=32)
    return _finish(bpy.context.active_object, name, colour, role, kw)


def add_sphere(name, radius, location=(0, 0, 0), colour=None, role="static", **kw):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=24, ring_count=16)
    return _finish(bpy.context.active_object, name, colour, role, kw)


def add_plane(name, size, location=(0, 0, 0), colour=None, role="static", **kw):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.scale = (size[0], size[1], 1.0)
    return _finish(obj, name, colour, role, kw)


def add_empty(name, location=(0, 0, 0), rotation=(0, 0, 0), role="robot_mount", **kw):
    bpy.ops.object.empty_add(type="ARROWS", location=location, rotation=rotation, radius=0.12)
    obj = bpy.context.active_object
    obj.name = name
    tag(obj, role=role, **kw)
    return obj


# ---------------------------------------------------------------------------
# Composite fixtures
# ---------------------------------------------------------------------------


def add_table(name="table", size=(0.90, 1.20, 0.05), top_z=0.63, center=(0.0, 0.0),
              colour=(0.60, 0.48, 0.34, 1.0), legs=True):
    """A table whose *top surface* lands exactly on `top_z`.

    Authoring by surface height rather than centre height removes the most
    common off-by-a-half-thickness error in these scenes.
    """
    thickness = size[2]
    top = add_box(name, (size[0], size[1], thickness),
                  location=(center[0], center[1], top_z - thickness / 2.0),
                  colour=colour, role="static", tags="surface")
    created = [top]
    if legs:
        for sx in (-1, 1):
            for sy in (-1, 1):
                leg = add_box(
                    f"{name}_leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                    (0.06, 0.06, top_z - thickness),
                    location=(center[0] + sx * (size[0] / 2 - 0.07),
                              center[1] + sy * (size[1] / 2 - 0.07),
                              (top_z - thickness) / 2.0),
                    colour=(0.26, 0.27, 0.30, 1.0), role="decor")
                created.append(leg)
    return created


def add_bin(name, inner=(0.16, 0.16), depth=0.06, wall=0.006, location=(0, 0, 0.63),
            colour=(0.30, 0.38, 0.52, 1.0)):
    """An open-topped tray: floor plus four walls, sitting on `location[2]`."""
    half_x, half_y = inner[0] / 2.0, inner[1] / 2.0
    x, y, z = location
    parts = [add_box(name, (inner[0], inner[1], wall),
                     location=(x, y, z + wall / 2.0), colour=colour,
                     role="static", tags=f"bin,{name}")]
    for sx, sy, label in ((1, 0, "px"), (-1, 0, "nx"), (0, 1, "py"), (0, -1, "ny")):
        parts.append(add_box(
            f"{name}_wall_{label}",
            (wall if sx else inner[0], inner[1] if sx else wall, depth),
            location=(x + sx * (half_x + wall / 2.0), y + sy * (half_y + wall / 2.0), z + depth / 2.0),
            colour=colour, role="static", tags="bin_wall"))
    return parts


def add_conveyor_marker(name, origin, direction="+y", length=0.5, width=0.2,
                        roller_radius=0.03, speed=0.1, spacing=None, **kw):
    """A placeholder that becomes a powered roller bed in MuJoCo.

    Only a visual proxy is drawn in Blender — one flat slab plus arrows — since
    modelling sixteen individual rollers by hand is exactly the work the
    `roller_conveyor` macro exists to avoid. The exporter reads the parameters
    off this object and emits the macro.
    """
    vector = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}[direction]
    size = (length if abs(vector[0]) else width, length if abs(vector[1]) else width, roller_radius * 2)
    obj = add_box(name, size, location=(origin[0], origin[1], origin[2]),
                  colour=(0.45, 0.47, 0.52, 1.0), role="ignore")
    obj["mrs_role"] = "ignore"
    obj["mrs_dynamic"] = "roller_conveyor"
    obj["mrs_direction"] = direction
    obj["mrs_length"] = float(length)
    obj["mrs_width"] = float(width)
    obj["mrs_roller_radius"] = float(roller_radius)
    obj["mrs_speed"] = float(speed)
    if spacing is not None:
        obj["mrs_spacing"] = float(spacing)
    for key, value in kw.items():
        obj[f"mrs_{key}"] = value
    return obj


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------


def set_colour(obj, rgba):
    """Assign a flat viewport+render colour. Materials are not exported to
    MuJoCo as textures; only the base colour survives."""
    rgba = tuple(rgba) + (1.0,) * (4 - len(rgba))
    material = bpy.data.materials.get(f"mrs_{obj.name}") or bpy.data.materials.new(f"mrs_{obj.name}")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
    material.diffuse_color = rgba
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj["mrs_rgba"] = list(rgba)
    return material


# ---------------------------------------------------------------------------
# Review rendering
# ---------------------------------------------------------------------------

REVIEW_VIEWS = {
    "front":    ((0.0, -1.60, 1.15), (0.0, 0.0, 0.72)),
    "three_q":  ((1.25, -1.25, 1.35), (0.0, 0.0, 0.72)),
    "top":      ((0.0, 0.001, 2.10), (0.0, 0.0, 0.66)),
    "left":     ((-1.65, 0.0, 1.10), (0.0, 0.0, 0.72)),
    "right":    ((1.65, 0.0, 1.10), (0.0, 0.0, 0.72)),
    "robot_eye": ((0.90, 0.0, 1.05), (-0.10, 0.0, 0.70)),
}


def _aim(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_views(out_dir, views=None, resolution=640, samples=16, engine=None, prefix="view"):
    """Render the scene from several fixed viewpoints and return the paths.

    Multiple angles are the point. A single hero render hides the two errors
    that matter most in a robot scene — objects floating a centimetre above the
    surface they should rest on, and objects intersecting each other — because
    both are invisible from a viewpoint that happens to look along the gap.
    """
    views = views or REVIEW_VIEWS
    os.makedirs(out_dir, exist_ok=True)
    scene = bpy.context.scene

    if engine is None:
        available = {item.identifier for item in
                     bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
            if candidate in available:
                engine = candidate
                break
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = samples

    _ensure_review_light()

    camera_data = bpy.data.cameras.new("mrs_review_cam")
    camera = bpy.data.objects.new("mrs_review_cam", camera_data)
    scene.collection.objects.link(camera)
    previous_camera = scene.camera
    scene.camera = camera

    written = []
    try:
        for name, (position, target) in views.items():
            camera.location = position
            _aim(camera, target)
            path = os.path.join(out_dir, f"{prefix}_{name}.png")
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
            written.append(path)
    finally:
        scene.camera = previous_camera
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data, do_unlink=True)

    return written


def _ensure_review_light():
    if any(o.type == "LIGHT" for o in bpy.context.scene.objects):
        return
    data = bpy.data.lights.new("mrs_review_key", type="SUN")
    data.energy = 3.0
    light = bpy.data.objects.new("mrs_review_key", data)
    light.location = (1.5, -1.5, 3.0)
    _aim(light, (0.0, 0.0, 0.7))
    bpy.context.scene.collection.objects.link(light)


# ---------------------------------------------------------------------------
# Measurement — answers the questions a render cannot
# ---------------------------------------------------------------------------


def world_bounds(obj):
    """Axis-aligned world-space bounds of an object, as (min, max)."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    low = [min(c[i] for c in corners) for i in range(3)]
    high = [max(c[i] for c in corners) for i in range(3)]
    return low, high


def measure(names=None, precision=4):
    """Report position and extent of every tagged object.

    Renders show you that something is wrong; this tells you by how much.
    """
    report = []
    for obj in _exportable_objects():
        if names is not None and obj.name not in names:
            continue
        low, high = world_bounds(obj)
        report.append({
            "name": obj.name,
            "role": obj.get("mrs_role", "static"),
            "location": [round(v, precision) for v in obj.matrix_world.translation],
            "size": [round(high[i] - low[i], precision) for i in range(3)],
            "min": [round(v, precision) for v in low],
            "max": [round(v, precision) for v in high],
        })
    return report


def check_overlaps(tolerance=1e-4, ignore_roles=("decor", "ignore", "camera", "robot_mount")):
    """Axis-aligned bounding-box overlaps between tagged objects.

    A cheap proxy for the interpenetration that makes MuJoCo diverge on the
    first step. It over-reports (two touching boxes share a face, and a
    concave bin legitimately overlaps its own walls' boxes) so treat hits as
    things to look at, not as errors. `validate_env.py` does the exact test on
    the compiled model.
    """
    objects = [o for o in _exportable_objects() if o.get("mrs_role", "static") not in ignore_roles]
    bounds = {o.name: world_bounds(o) for o in objects}
    hits = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            (a_lo, a_hi), (b_lo, b_hi) = bounds[a.name], bounds[b.name]
            depth = [min(a_hi[k], b_hi[k]) - max(a_lo[k], b_lo[k]) for k in range(3)]
            if all(d > tolerance for d in depth):
                hits.append({"a": a.name, "b": b.name,
                             "overlap": [round(d, 4) for d in depth]})
    return hits


def check_resting(surface_z, tolerance=0.002, roles=("free",)):
    """Objects whose underside is not within `tolerance` of `surface_z`.

    Catches the classic 'looks fine, is floating 8 mm above the table' error,
    which in MuJoCo becomes a part that drops and bounces the moment the
    episode starts.
    """
    floating = []
    for obj in _exportable_objects():
        if obj.get("mrs_role", "static") not in roles:
            continue
        low, _ = world_bounds(obj)
        gap = low[2] - surface_z
        if abs(gap) > tolerance:
            floating.append({"name": obj.name, "gap": round(gap, 4)})
    return floating


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _exportable_objects():
    return [o for o in bpy.context.scene.objects if o.get("mrs_role", "static") != "ignore"
            or o.get("mrs_dynamic")]


def _props(obj):
    out = {}
    for key in obj.keys():
        if not key.startswith("mrs_"):
            continue
        value = obj[key]
        try:
            value = list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value
        except TypeError:
            pass
        out[key[4:]] = value
    return out


def export_scene_graph(path, mesh_dir=None, mesh_roles=("free", "static", "hinged", "sliding")):
    """Write `scene_graph.json`, exporting an OBJ for every mesh-collision object.

    The graph is the hand-off: it records world transforms, world-space extents
    and every `mrs_*` property, and nothing about Blender. `build_env.py`
    consumes it without importing bpy, so the conversion is testable off-line.
    """
    scene = bpy.context.scene
    if abs(scene.unit_settings.scale_length - 1.0) > 1e-9:
        raise RuntimeError(
            f"Scene unit scale is {scene.unit_settings.scale_length}, not 1.0. "
            f"Author in metres or the exported dimensions will be wrong."
        )

    path = os.path.abspath(path)
    mesh_dir = mesh_dir or os.path.join(os.path.dirname(path), "assets")
    os.makedirs(mesh_dir, exist_ok=True)

    entries = []
    for obj in _exportable_objects():
        props = _props(obj)
        role = props.get("role", "static")
        translation, quaternion, scale = obj.matrix_world.decompose()
        low, high = world_bounds(obj)

        entry = {
            "name": obj.name,
            "type": obj.type,
            "role": role,
            "pos": [round(v, 6) for v in translation],
            "quat": [round(quaternion.w, 6), round(quaternion.x, 6),
                     round(quaternion.y, 6), round(quaternion.z, 6)],
            "scale": [round(v, 6) for v in scale],
            # `dimensions` is the object's own extent along its local axes and
            # is what becomes the MuJoCo geom size. `extent` is the world
            # axis-aligned bounding box, which differs the moment an object is
            # rotated and is only used for the overlap and clearance checks.
            "dimensions": [round(v, 6) for v in obj.dimensions] if obj.type == "MESH" else [0.0, 0.0, 0.0],
            "extent": [round(high[i] - low[i], 6) for i in range(3)],
            "bounds_min": [round(v, 6) for v in low],
            "bounds_max": [round(v, 6) for v in high],
            "parent": obj.parent.name if obj.parent else None,
            "props": props,
        }

        if props.get("collision") == "mesh" and obj.type == "MESH" and role in mesh_roles:
            entry["mesh_file"] = _export_obj(obj, mesh_dir)

        entries.append(entry)

    payload = {
        "kit_version": MRS_KIT_VERSION,
        "blender": bpy.app.version_string,
        "unit_scale": scene.unit_settings.scale_length,
        "frame_range": [scene.frame_start, scene.frame_end, scene.render.fps],
        "objects": entries,
        "animation": _export_animation(),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return {"path": path, "objects": len(entries), "meshes": sum("mesh_file" in e for e in entries),
            "animated": len(payload["animation"])}


def _export_obj(obj, mesh_dir):
    """Write the object's mesh in its OWN frame, with scale baked in.

    `wm.obj_export` writes world-space vertices. MuJoCo then applies the body
    transform on top, so exporting as-is places the mesh at twice its position
    and rotates it twice. Neutralise translation and rotation for the duration
    of the export but keep scale, since the MuJoCo mesh asset is registered at
    scale (1, 1, 1).
    """
    from mathutils import Matrix

    filename = f"{obj.name}.obj"
    filepath = os.path.join(mesh_dir, filename)

    original = obj.matrix_world.copy()
    _, _, scale = original.decompose()
    try:
        obj.matrix_world = Matrix.Diagonal(scale).to_4x4()
        bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True,
                              apply_modifiers=True, export_materials=False,
                              global_scale=1.0, forward_axis="Y", up_axis="Z")
        obj.select_set(False)
    finally:
        obj.matrix_world = original
        bpy.context.view_layer.update()

    return filename


def _export_animation(step=2):
    """Bake keyframed object motion into sample tables.

    Blender's F-curves do not survive the trip to MuJoCo, so anything the
    author animated is sampled at a fixed rate and replayed by the `baked`
    driver against a mocap body.
    """
    scene = bpy.context.scene
    fps = scene.render.fps or 24
    baked = {}

    for obj in _exportable_objects():
        if obj.animation_data is None or obj.animation_data.action is None:
            continue
        samples = []
        for frame in range(scene.frame_start, scene.frame_end + 1, step):
            scene.frame_set(frame)
            translation, quaternion, _ = obj.matrix_world.decompose()
            samples.append([
                round((frame - scene.frame_start) / fps, 5),
                round(translation.x, 6), round(translation.y, 6), round(translation.z, 6),
                round(quaternion.w, 6), round(quaternion.x, 6),
                round(quaternion.y, 6), round(quaternion.z, 6),
            ])
        if len(samples) >= 2:
            baked[obj.name] = samples

    scene.frame_set(scene.frame_start)
    return baked


def summary():
    """One-line-per-object overview, cheap to print after every edit."""
    lines = []
    for obj in _exportable_objects():
        low, high = world_bounds(obj)
        size = [round(high[i] - low[i], 3) for i in range(3)]
        lines.append(f"{obj.get('mrs_role', 'static'):<11} {obj.name:<26} "
                     f"pos={[round(v, 3) for v in obj.matrix_world.translation]} size={size}")
    return "\n".join(sorted(lines))


print(f"mrs blender kit v{MRS_KIT_VERSION} loaded — "
      f"{len([o for o in bpy.context.scene.objects])} objects in scene")
