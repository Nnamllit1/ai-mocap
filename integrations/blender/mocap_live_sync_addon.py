from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


bl_info = {
    "name": "Mocap Live Sync (OSC)",
    "author": "Mocap Web Portal",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Mocap Live",
    "description": "Creates a COCO-17 armature and syncs live OSC mocap data.",
    "category": "Animation",
}


COCO_JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

JOINT_NAME_SET = set(COCO_JOINTS)
BONE_EDGES = [(COCO_JOINTS[a], COCO_JOINTS[b]) for a, b in COCO_EDGES]

COLLECTION_NAME = "MOCAP_Live"
ARMATURE_NAME = "MOCAP_Skeleton"
JOINT_PREFIX = "MOCAP_J_"
BONE_PREFIX = "MOCAP_B_"

AXIS_PRESET_PASSTHROUGH = "passthrough"
AXIS_PRESET_Z_UP = "blender_z_up"
AXIS_PRESET_Z_UP_NEG_X = "blender_z_up_neg_x"
AXIS_PRESET_Y_UP = "blender_y_up"

AXIS_PRESET_LABELS = {
    AXIS_PRESET_PASSTHROUGH: "Passthrough (x, y, z)",
    AXIS_PRESET_Z_UP: "Blender Z-up (x, z, -y)",
    AXIS_PRESET_Z_UP_NEG_X: "Blender Z-up (-x, z, -y)",
    AXIS_PRESET_Y_UP: "Y-up style (x, -z, y)",
}

AXIS_PRESET_ITEMS = [
    (
        AXIS_PRESET_PASSTHROUGH,
        AXIS_PRESET_LABELS[AXIS_PRESET_PASSTHROUGH],
        "Do not remap axes.",
    ),
    (
        AXIS_PRESET_Z_UP,
        AXIS_PRESET_LABELS[AXIS_PRESET_Z_UP],
        "Most common CV-to-Blender mapping.",
    ),
    (
        AXIS_PRESET_Z_UP_NEG_X,
        AXIS_PRESET_LABELS[AXIS_PRESET_Z_UP_NEG_X],
        "Variant mapping if left/right appears mirrored.",
    ),
    (
        AXIS_PRESET_Y_UP,
        AXIS_PRESET_LABELS[AXIS_PRESET_Y_UP],
        "Alternative mapping for some camera frames.",
    ),
]


@dataclass
class JointCacheEntry:
    position: Optional[Tuple[float, float, float]] = None
    last_good_position: Optional[Tuple[float, float, float]] = None
    confidence: float = 0.0
    timestamp: float = 0.0
    last_update_monotonic: float = 0.0


@dataclass
class RuntimeState:
    sock: Optional[socket.socket] = None
    listening: bool = False
    joint_cache: Dict[str, JointCacheEntry] = None
    active_cameras: int = 0
    valid_joints: int = 0
    last_packet_monotonic: float = 0.0
    last_error: str = ""

    def __post_init__(self) -> None:
        if self.joint_cache is None:
            self.joint_cache = {}


RUNTIME = RuntimeState()


def joint_empty_name(joint_name: str) -> str:
    return f"{JOINT_PREFIX}{joint_name}"


def bone_name(joint_a: str, joint_b: str) -> str:
    return f"{BONE_PREFIX}{joint_a}__{joint_b}"


def normalize_osc_prefix(prefix: str) -> str:
    value = str(prefix or "/mocap").strip()
    if not value:
        value = "/mocap"
    if not value.startswith("/"):
        value = f"/{value}"
    value = value.rstrip("/")
    return value or "/mocap"


def read_osc_padded_string(packet: bytes, offset: int) -> tuple[Optional[str], int]:
    if offset < 0 or offset >= len(packet):
        return None, offset
    end = packet.find(b"\x00", offset)
    if end < 0:
        return None, offset
    raw = packet[offset:end]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    next_offset = ((end + 4) // 4) * 4
    return text, next_offset


def _all_finite(values: Tuple[float, ...]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _score_joint_payload(values: Tuple[float, ...]) -> int:
    if len(values) != 5 or not _all_finite(values):
        return -10_000
    x, y, z, conf, timestamp = values
    score = 0
    if max(abs(x), abs(y), abs(z)) <= 25.0:
        score += 6
    elif max(abs(x), abs(y), abs(z)) <= 500.0:
        score += 2
    else:
        score -= 6
    if 0.0 <= conf <= 1.25:
        score += 8
    elif -0.25 <= conf <= 2.0:
        score += 2
    else:
        score -= 8
    if timestamp > 1_000_000.0:
        score += 8
    elif timestamp > 0.0:
        score += 3
    else:
        score -= 6
    return score


def _score_status_payload(values: Tuple[float, ...]) -> int:
    if len(values) != 2 or not _all_finite(values):
        return -10_000
    active, valid = values
    score = 0
    if 0.0 <= active <= 64.0:
        score += 6
    else:
        score -= 6
    if 0.0 <= valid <= 128.0:
        score += 6
    else:
        score -= 6
    if abs(active - round(active)) <= 0.05:
        score += 2
    if abs(valid - round(valid)) <= 0.05:
        score += 2
    # Wrong-endian decodes often collapse to near-zero denormals.
    if max(abs(active), abs(valid)) < 0.01:
        score -= 8
    elif valid >= 1.0:
        score += 2
    return score


def _score_generic_payload(values: Tuple[float, ...]) -> int:
    if not values or not _all_finite(values):
        return -10_000
    return sum(1 if abs(v) <= 1_000_000.0 else -2 for v in values)


def _choose_endianness(
    little_values: Tuple[float, ...],
    big_values: Tuple[float, ...],
    address: str,
) -> Tuple[float, ...]:
    if "/joint/" in address and len(little_values) == 5:
        little_score = _score_joint_payload(little_values)
        big_score = _score_joint_payload(big_values)
    elif address.endswith("/status") and len(little_values) == 2:
        little_score = _score_status_payload(little_values)
        big_score = _score_status_payload(big_values)
    else:
        little_score = _score_generic_payload(little_values)
        big_score = _score_generic_payload(big_values)
    return little_values if little_score >= big_score else big_values


def decode_osc_message(packet: bytes) -> Optional[dict]:
    if not packet:
        return None
    address, offset = read_osc_padded_string(packet, 0)
    if not address or not address.startswith("/"):
        return None
    tags, offset = read_osc_padded_string(packet, offset)
    if not tags or not tags.startswith(","):
        return None
    tags = tags[1:]
    if any(tag != "f" for tag in tags):
        return None
    arg_count = len(tags)
    payload = packet[offset:]
    expected_bytes = 4 * arg_count
    if len(payload) < expected_bytes:
        return None
    payload = payload[:expected_bytes]
    fmt = "f" * arg_count
    little = struct.unpack("<" + fmt, payload)
    big = struct.unpack(">" + fmt, payload)
    values = _choose_endianness(tuple(little), tuple(big), address)
    return {
        "address": address,
        "tags": tags,
        "values": tuple(float(v) for v in values),
    }


def decode_joint_packet(packet: bytes, prefix: str) -> Optional[dict]:
    msg = decode_osc_message(packet)
    if msg is None:
        return None
    normalized_prefix = normalize_osc_prefix(prefix)
    joint_root = f"{normalized_prefix}/joint/"
    address = str(msg["address"])
    if not address.startswith(joint_root):
        return None
    joint_name = address[len(joint_root) :]
    if joint_name not in JOINT_NAME_SET:
        return None
    values = msg["values"]
    if len(values) != 5:
        return None
    return {
        "joint_name": joint_name,
        "xyz": (float(values[0]), float(values[1]), float(values[2])),
        "confidence": float(values[3]),
        "timestamp": float(values[4]),
    }


def decode_status_packet(packet: bytes, prefix: str) -> Optional[dict]:
    msg = decode_osc_message(packet)
    if msg is None:
        return None
    normalized_prefix = normalize_osc_prefix(prefix)
    expected_address = f"{normalized_prefix}/status"
    if str(msg["address"]) != expected_address:
        return None
    values = msg["values"]
    if len(values) != 2:
        return None
    return {
        "active_cameras": float(values[0]),
        "valid_joints": float(values[1]),
    }


def apply_axis_preset(
    xyz: Tuple[float, float, float],
    preset: str,
    scale: float = 1.0,
) -> Tuple[float, float, float]:
    x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    if preset == AXIS_PRESET_Z_UP:
        mapped = (x, z, -y)
    elif preset == AXIS_PRESET_Z_UP_NEG_X:
        mapped = (-x, z, -y)
    elif preset == AXIS_PRESET_Y_UP:
        mapped = (x, -z, y)
    else:
        mapped = (x, y, z)
    mul = float(scale)
    return (mapped[0] * mul, mapped[1] * mul, mapped[2] * mul)


def update_joint_cache_entry(
    entry: JointCacheEntry,
    mapped_xyz: Tuple[float, float, float],
    confidence: float,
    packet_timestamp: float,
    now_monotonic: float,
    confidence_threshold: float,
) -> JointCacheEntry:
    pos = (float(mapped_xyz[0]), float(mapped_xyz[1]), float(mapped_xyz[2]))
    conf = float(confidence)
    entry.position = pos
    entry.confidence = conf
    entry.timestamp = float(packet_timestamp)
    entry.last_update_monotonic = float(now_monotonic)
    if conf >= float(confidence_threshold):
        entry.last_good_position = pos
    elif entry.last_good_position is None:
        entry.last_good_position = pos
    return entry


def resolve_joint_output_position(
    entry: JointCacheEntry,
    now_monotonic: float,
    stale_timeout_s: float,
) -> tuple[Optional[Tuple[float, float, float]], bool]:
    if entry.last_update_monotonic <= 0.0:
        return None, True
    stale = (float(now_monotonic) - float(entry.last_update_monotonic)) > float(stale_timeout_s)
    if entry.last_good_position is not None:
        return entry.last_good_position, stale
    return entry.position, stale


def reset_runtime_state(clear_socket: bool = True) -> None:
    if clear_socket and RUNTIME.sock is not None:
        try:
            RUNTIME.sock.close()
        except OSError:
            pass
    RUNTIME.sock = None
    RUNTIME.listening = False
    RUNTIME.joint_cache.clear()
    RUNTIME.active_cameras = 0
    RUNTIME.valid_joints = 0
    RUNTIME.last_packet_monotonic = 0.0
    RUNTIME.last_error = ""


try:
    import bpy
    from bpy.props import (
        BoolProperty,
        EnumProperty,
        FloatProperty,
        IntProperty,
        PointerProperty,
        StringProperty,
    )
except Exception:
    bpy = None


def _ensure_collection():
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION_NAME)
        bpy.context.scene.collection.children.link(collection)
    elif bpy.context.scene.collection.children.get(collection.name) is None:
        bpy.context.scene.collection.children.link(collection)
    return collection


def _ensure_joint_empty(collection, joint_name: str):
    name = joint_empty_name(joint_name)
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = "SPHERE"
        obj.empty_display_size = 0.03
    if collection.objects.get(obj.name) is None:
        collection.objects.link(obj)
    return obj


def _ensure_armature_object(collection):
    arm_obj = bpy.data.objects.get(ARMATURE_NAME)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        arm_data = bpy.data.armatures.new(f"{ARMATURE_NAME}_DATA")
        arm_obj = bpy.data.objects.new(ARMATURE_NAME, arm_data)
        collection.objects.link(arm_obj)
    elif collection.objects.get(arm_obj.name) is None:
        collection.objects.link(arm_obj)
    arm_obj.show_in_front = True
    return arm_obj


def _activate_object(obj) -> None:
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    for selected in list(bpy.context.selected_objects):
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _rebuild_armature_segments(arm_obj, empties: Dict[str, object]) -> None:
    _activate_object(arm_obj)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm_obj.data.edit_bones
    for bone in list(edit_bones):
        edit_bones.remove(bone)
    for joint_a, joint_b in BONE_EDGES:
        bone = edit_bones.new(bone_name(joint_a, joint_b))
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.1, 0.0)
        bone.use_connect = False
    bpy.ops.object.mode_set(mode="POSE")
    for joint_a, joint_b in BONE_EDGES:
        pb = arm_obj.pose.bones.get(bone_name(joint_a, joint_b))
        if pb is None:
            continue
        for cons in list(pb.constraints):
            pb.constraints.remove(cons)
        copy_loc = pb.constraints.new(type="COPY_LOCATION")
        copy_loc.target = empties[joint_a]
        stretch = pb.constraints.new(type="STRETCH_TO")
        stretch.target = empties[joint_b]
    bpy.ops.object.mode_set(mode="OBJECT")


def _create_or_rebuild_skeleton() -> str:
    collection = _ensure_collection()
    empties = {joint: _ensure_joint_empty(collection, joint) for joint in COCO_JOINTS}
    arm_obj = _ensure_armature_object(collection)
    _rebuild_armature_segments(arm_obj, empties)
    return f"Skeleton ready: {len(COCO_JOINTS)} joints, {len(BONE_EDGES)} bones."


def _listener_start(port: int) -> tuple[bool, str]:
    if RUNTIME.listening and RUNTIME.sock is not None:
        return True, "Already listening."
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", int(port)))
        sock.setblocking(False)
    except OSError as exc:
        reset_runtime_state(clear_socket=True)
        return False, f"Failed to bind UDP {port}: {exc}"
    RUNTIME.sock = sock
    RUNTIME.listening = True
    RUNTIME.last_error = ""
    return True, f"Listening on UDP {port}."


def _listener_stop() -> None:
    if RUNTIME.sock is not None:
        try:
            RUNTIME.sock.close()
        except OSError:
            pass
    RUNTIME.sock = None
    RUNTIME.listening = False


def _settings_from_context():
    scene = bpy.context.scene
    if scene is None:
        return None
    return getattr(scene, "mocap_live_settings", None)


def _apply_joint_cache_to_scene(settings, now_monotonic: float) -> tuple[int, int]:
    applied = 0
    stale_count = 0
    stale_timeout_s = float(settings.stale_timeout_s)
    for joint_name in COCO_JOINTS:
        obj = bpy.data.objects.get(joint_empty_name(joint_name))
        if obj is None:
            continue
        entry = RUNTIME.joint_cache.get(joint_name)
        if entry is None:
            continue
        pos, stale = resolve_joint_output_position(entry, now_monotonic, stale_timeout_s)
        if pos is None:
            continue
        if stale:
            stale_count += 1
        obj.location = pos
        applied += 1
    return applied, stale_count


def _sync_status_props(
    settings,
    now_monotonic: float,
    applied_joint_count: int,
    stale_joint_count: int,
) -> None:
    settings.is_listening = bool(RUNTIME.listening)
    if RUNTIME.last_packet_monotonic > 0.0:
        settings.last_packet_age_s = max(0.0, float(now_monotonic - RUNTIME.last_packet_monotonic))
    else:
        settings.last_packet_age_s = -1.0
    settings.last_error = str(RUNTIME.last_error)
    settings.status_active_cameras = int(round(float(RUNTIME.active_cameras)))
    settings.status_valid_joints = int(round(float(RUNTIME.valid_joints)))
    settings.status_applied_joints = int(applied_joint_count)
    settings.status_stale_joints = int(stale_joint_count)


def _ingest_packet(packet: bytes, settings, now_monotonic: float) -> None:
    prefix = normalize_osc_prefix(settings.osc_prefix)
    joint = decode_joint_packet(packet, prefix)
    if joint is not None:
        joint_name = joint["joint_name"]
        mapped = apply_axis_preset(
            joint["xyz"],
            settings.axis_preset,
            scale=float(settings.global_scale),
        )
        entry = RUNTIME.joint_cache.get(joint_name, JointCacheEntry())
        entry = update_joint_cache_entry(
            entry,
            mapped_xyz=mapped,
            confidence=float(joint["confidence"]),
            packet_timestamp=float(joint["timestamp"]),
            now_monotonic=now_monotonic,
            confidence_threshold=float(settings.confidence_threshold),
        )
        RUNTIME.joint_cache[joint_name] = entry
        return
    status = decode_status_packet(packet, prefix)
    if status is not None:
        RUNTIME.active_cameras = float(status["active_cameras"])
        RUNTIME.valid_joints = float(status["valid_joints"])


def _poll_timer():
    if bpy is None:
        return None
    settings = _settings_from_context()
    if settings is None:
        return None
    poll_interval_s = max(0.005, float(settings.poll_interval_s))
    if not RUNTIME.listening or RUNTIME.sock is None:
        _sync_status_props(settings, time.monotonic(), applied_joint_count=0, stale_joint_count=0)
        return None

    max_packets = max(1, int(settings.max_packets_per_tick))
    now_monotonic = time.monotonic()
    packets = 0
    try:
        while packets < max_packets:
            try:
                packet, _addr = RUNTIME.sock.recvfrom(8192)
            except BlockingIOError:
                break
            packets += 1
            now_monotonic = time.monotonic()
            RUNTIME.last_packet_monotonic = now_monotonic
            _ingest_packet(packet, settings, now_monotonic)
    except OSError as exc:
        RUNTIME.last_error = f"Socket error: {exc}"
        _listener_stop()
        _sync_status_props(settings, time.monotonic(), applied_joint_count=0, stale_joint_count=0)
        return None

    applied, stale = _apply_joint_cache_to_scene(settings, now_monotonic)
    _sync_status_props(
        settings,
        now_monotonic,
        applied_joint_count=applied,
        stale_joint_count=stale,
    )
    return poll_interval_s


if bpy is not None:

    class MocapLiveSettings(bpy.types.PropertyGroup):
        listen_port: IntProperty(
            name="Listen Port",
            description="UDP listen port for incoming OSC data.",
            default=9000,
            min=1,
            max=65535,
        )
        osc_prefix: StringProperty(
            name="OSC Prefix",
            description="OSC prefix used by the server.",
            default="/mocap",
        )
        axis_preset: EnumProperty(
            name="Axis Mapping",
            description="Axis remap preset from incoming data to Blender.",
            items=AXIS_PRESET_ITEMS,
            default=AXIS_PRESET_Z_UP,
        )
        global_scale: FloatProperty(
            name="Scale",
            description="Global multiplier for all incoming positions.",
            default=1.0,
            min=0.0001,
            soft_max=10.0,
        )
        confidence_threshold: FloatProperty(
            name="Confidence Min",
            description="If confidence is below this, the joint holds last good value.",
            default=0.55,
            min=0.0,
            max=1.0,
        )
        max_packets_per_tick: IntProperty(
            name="Packets/Tick",
            description="Maximum OSC packets processed per timer tick.",
            default=256,
            min=1,
            max=4096,
        )
        poll_interval_s: FloatProperty(
            name="Poll Interval (s)",
            description="Timer interval for processing incoming packets.",
            default=0.02,
            min=0.005,
            soft_max=0.5,
        )
        stale_timeout_s: FloatProperty(
            name="Stale Timeout (s)",
            description="Age threshold for stale packet detection.",
            default=0.5,
            min=0.05,
            soft_max=5.0,
        )
        is_listening: BoolProperty(name="Listening", default=False)
        last_packet_age_s: FloatProperty(name="Last Packet Age", default=-1.0)
        status_active_cameras: IntProperty(name="Active Cameras", default=0)
        status_valid_joints: IntProperty(name="Valid Joints", default=0)
        status_applied_joints: IntProperty(name="Applied Joints", default=0)
        status_stale_joints: IntProperty(name="Stale Joints", default=0)
        last_error: StringProperty(name="Last Error", default="")


    class MOCAP_OT_create_skeleton(bpy.types.Operator):
        bl_idname = "mocap_live.create_skeleton"
        bl_label = "Create Skeleton"
        bl_description = "Create or rebuild the MOCAP skeleton armature and joint empties."

        def execute(self, context):
            try:
                msg = _create_or_rebuild_skeleton()
            except Exception as exc:
                self.report({"ERROR"}, f"Create skeleton failed: {exc}")
                return {"CANCELLED"}
            self.report({"INFO"}, msg)
            return {"FINISHED"}


    class MOCAP_OT_start_sync(bpy.types.Operator):
        bl_idname = "mocap_live.start_sync"
        bl_label = "Start Sync"
        bl_description = "Start UDP listener and begin live sync."

        def execute(self, context):
            settings = context.scene.mocap_live_settings
            ok, msg = _listener_start(settings.listen_port)
            if not ok:
                settings.last_error = msg
                self.report({"ERROR"}, msg)
                return {"CANCELLED"}
            settings.last_error = ""
            if not bpy.app.timers.is_registered(_poll_timer):
                bpy.app.timers.register(_poll_timer, first_interval=0.01, persistent=True)
            settings.is_listening = True
            self.report({"INFO"}, msg)
            return {"FINISHED"}


    class MOCAP_OT_stop_sync(bpy.types.Operator):
        bl_idname = "mocap_live.stop_sync"
        bl_label = "Stop Sync"
        bl_description = "Stop UDP listener."

        def execute(self, context):
            _listener_stop()
            settings = context.scene.mocap_live_settings
            settings.is_listening = False
            self.report({"INFO"}, "Sync stopped.")
            return {"FINISHED"}


    class MOCAP_OT_reset_live_state(bpy.types.Operator):
        bl_idname = "mocap_live.reset_live_state"
        bl_label = "Reset Live State"
        bl_description = "Clear all runtime caches but keep current skeleton objects."

        def execute(self, context):
            reset_runtime_state(clear_socket=False)
            settings = context.scene.mocap_live_settings
            settings.last_error = ""
            settings.last_packet_age_s = -1.0
            settings.status_active_cameras = 0
            settings.status_valid_joints = 0
            settings.status_applied_joints = 0
            settings.status_stale_joints = 0
            self.report({"INFO"}, "Live cache reset.")
            return {"FINISHED"}


    class MOCAP_PT_live_panel(bpy.types.Panel):
        bl_label = "Mocap Live"
        bl_idname = "MOCAP_PT_live_panel"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Mocap Live"

        def draw(self, context):
            layout = self.layout
            settings = context.scene.mocap_live_settings

            layout.prop(settings, "listen_port")
            layout.prop(settings, "osc_prefix")
            layout.prop(settings, "axis_preset")
            layout.prop(settings, "global_scale")
            layout.prop(settings, "confidence_threshold")
            layout.prop(settings, "max_packets_per_tick")
            layout.prop(settings, "poll_interval_s")
            layout.prop(settings, "stale_timeout_s")

            row = layout.row(align=True)
            row.operator("mocap_live.create_skeleton", icon="ARMATURE_DATA")
            row = layout.row(align=True)
            row.operator("mocap_live.start_sync", icon="PLAY")
            row.operator("mocap_live.stop_sync", icon="PAUSE")
            layout.operator("mocap_live.reset_live_state", icon="FILE_REFRESH")

            box = layout.box()
            box.label(text=f"Listening: {'yes' if settings.is_listening else 'no'}")
            if settings.last_packet_age_s < 0.0:
                box.label(text="Last Packet Age: n/a")
            else:
                box.label(text=f"Last Packet Age: {settings.last_packet_age_s:.3f}s")
            box.label(text=f"Active Cameras: {settings.status_active_cameras}")
            box.label(text=f"Valid Joints: {settings.status_valid_joints}")
            box.label(text=f"Applied Joints: {settings.status_applied_joints}")
            box.label(text=f"Stale Joints: {settings.status_stale_joints}")
            if settings.last_error:
                box.label(text=f"Error: {settings.last_error}")


    CLASSES = [
        MocapLiveSettings,
        MOCAP_OT_create_skeleton,
        MOCAP_OT_start_sync,
        MOCAP_OT_stop_sync,
        MOCAP_OT_reset_live_state,
        MOCAP_PT_live_panel,
    ]


    def register():
        for cls in CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.Scene.mocap_live_settings = PointerProperty(type=MocapLiveSettings)


    def unregister():
        _listener_stop()
        if bpy.app.timers.is_registered(_poll_timer):
            bpy.app.timers.unregister(_poll_timer)
        if hasattr(bpy.types.Scene, "mocap_live_settings"):
            del bpy.types.Scene.mocap_live_settings
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)

else:

    def register():
        return None


    def unregister():
        return None


if __name__ == "__main__":
    register()
