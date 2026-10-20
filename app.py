from __future__ import annotations

import io
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


st.set_page_config(
    page_title="Unitree G1 Dynamic Movement Demo",
    page_icon="🤖",
    layout="wide",
)


# =============================================================================
# Simulation-only configuration
# =============================================================================
# This demo is not a hardware controller. It produces realistic-looking
# simulated telemetry and a rendered MP4 for review, planning, and demos.
# =============================================================================

CONFIG: Dict[str, Any] = {
    "project": "Unitree G1: Dynamic Movement (simulation-only demo)",
    "simulation": {
        "fps": 30,
        "duration_s": 16.0,
        "seed": 7,
    },
    "robot": {
        "name": "Unitree G1",
        "height_m": 1.3,
        "mass_kg": 35.0,
        "dof_note": "23-43 DoF depending on variant; this demo uses a simplified 12-joint mirror",
        "used_dof": 12,
    },
    "camera": {
        "width": 640,
        "height": 368,
        "scale": 150,
        "cx": 320,
        "cy": 265,
        "azimuth_deg": 35.0,
        "elevation_deg": 18.0,
    },
    "links": {
        "thigh": 0.34,
        "shank": 0.34,
        "foot": 0.13,
        "torso": 0.34,
        "head": 0.13,
        "upper_arm": 0.26,
        "lower_arm": 0.24,
        "hip_width": 0.12,
        "shoulder_width": 0.18,
    },
    "motor_delay_ms": 12.0,
    "sensor_noise": {
        "joint_pos": 0.004,
        "joint_vel": 0.05,
        "imu_acc": 0.08,
        "gyro": 0.03,
        "velocity_est": 0.03,
    },
    "safety": {
        "hardware_allowed": False,
        "emergency_stop_mode": "damping",
        "max_auto_retries": 0,
        "note": "All limits are enforced in one place. Hardware deployment requires a separate human-reviewed gate.",
    },
}


JOINT_LIMITS: Dict[str, Dict[str, float]] = {}

for side in ("left", "right"):
    JOINT_LIMITS.update(
        {
            f"{side}_hip_pitch": {
                "pos_min": -0.65,
                "pos_max": 1.25,
                "vel_max": 18.0,
                "torque_max": 45.0,
                "kp": 120.0,
                "kd": 4.0,
            },
            f"{side}_hip_roll": {
                "pos_min": -0.35,
                "pos_max": 0.35,
                "vel_max": 14.0,
                "torque_max": 30.0,
                "kp": 90.0,
                "kd": 3.0,
            },
            f"{side}_knee": {
                "pos_min": 0.0,
                "pos_max": 1.75,
                "vel_max": 20.0,
                "torque_max": 60.0,
                "kp": 150.0,
                "kd": 5.0,
            },
            f"{side}_ankle_pitch": {
                "pos_min": -0.45,
                "pos_max": 0.45,
                "vel_max": 16.0,
                "torque_max": 25.0,
                "kp": 70.0,
                "kd": 2.5,
            },
            f"{side}_shoulder_pitch": {
                "pos_min": -1.40,
                "pos_max": 1.40,
                "vel_max": 18.0,
                "torque_max": 20.0,
                "kp": 50.0,
                "kd": 2.0,
            },
            f"{side}_elbow": {
                "pos_min": 0.0,
                "pos_max": 1.70,
                "vel_max": 18.0,
                "torque_max": 18.0,
                "kp": 45.0,
                "kd": 2.0,
            },
        }
    )

JOINT_NAMES: List[str] = list(JOINT_LIMITS.keys())

SAFE_POSTURE: Dict[str, float] = {name: 0.0 for name in JOINT_NAMES}

for side in ("left", "right"):
    SAFE_POSTURE.update(
        {
            f"{side}_hip_pitch": 0.08,
            f"{side}_hip_roll": 0.0,
            f"{side}_knee": 0.35,
            f"{side}_ankle_pitch": -0.05,
            f"{side}_shoulder_pitch": -0.15,
            f"{side}_elbow": 0.35,
        }
    )


BASE_SKILLS: List[Tuple[str, float, float]] = [
    ("walk", 0.0, 4.0),
    ("turn", 4.0, 5.2),
    ("run", 5.2, 7.5),
    ("jump", 7.5, 9.0),
    ("push_recovery", 9.0, 10.8),
    ("fall", 10.8, 12.0),
    ("stand_up", 12.0, 14.5),
    ("kick", 14.5, 16.0),
]

SKILL_START: Dict[str, float] = {name: start for name, start, _ in BASE_SKILLS}


LINK_PAIRS: List[Tuple[str, str]] = [
    ("pelvis", "chest"),
    ("chest", "head"),
    ("chest", "left_shoulder"),
    ("chest", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("left_ankle", "left_heel"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
    ("right_ankle", "right_heel"),
]


# =============================================================================
# Small math helpers
# =============================================================================

def clamp_scalar(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar value to [lo, hi]."""
    return float(min(hi, max(lo, x)))


def smoothstep(x: float) -> float:
    """Smooth 0..1 ramp."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def skill_at(t: float) -> str:
    """Return the active skill name at time t."""
    for name, start, end in BASE_SKILLS:
        if start <= t < end:
            return name
    return "hold"


def command_profile(t: float, speed_scale: float) -> Dict[str, float]:
    """Generate high-level velocity commands for the scripted demo."""
    skill = skill_at(t)

    vx = 0.0
    vy = 0.0
    yaw_rate = 0.0

    if skill == "walk":
        vx = 0.80 * speed_scale
    elif skill == "turn":
        vx = 0.22 * speed_scale
        yaw_rate = 1.05
    elif skill == "run":
        vx = 1.35 * speed_scale
    elif skill == "jump":
        vx = 0.20 * speed_scale
    elif skill == "push_recovery":
        vx = 0.45 * speed_scale
    elif skill == "kick":
        vx = 0.12 * speed_scale
    elif skill in ("fall", "stand_up", "hold"):
        vx = 0.0

    return {
        "skill": skill,
        "vx": vx,
        "vy": vy,
        "yaw_rate": yaw_rate,
    }


# =============================================================================
# Kinematic reference generator
# =============================================================================

def desired_joints(
    t: float,
    skill: str,
    phase: float,
    speed: float,
    disturbance: float,
    estop: bool,
) -> Dict[str, float]:
    """
    Produce a desired joint-angle reference for the simplified G1 mirror.

    This is intentionally deterministic and simulation-only. It is shaped to
    look like a plausible humanoid motion stack for demos and UI validation.
    """
    j: Dict[str, float] = {}

    # Neutral standing baseline.
    for side in ("left", "right"):
        j[f"{side}_hip_pitch"] = 0.03
        j[f"{side}_hip_roll"] = 0.0
        j[f"{side}_knee"] = 0.14
        j[f"{side}_ankle_pitch"] = -0.04
        j[f"{side}_shoulder_pitch"] = -0.06
        j[f"{side}_elbow"] = 0.20

    if estop:
        for side in ("left", "right"):
            j[f"{side}_hip_pitch"] = SAFE_POSTURE[f"{side}_hip_pitch"]
            j[f"{side}_hip_roll"] = SAFE_POSTURE[f"{side}_hip_roll"]
            j[f"{side}_knee"] = SAFE_POSTURE[f"{side}_knee"]
            j[f"{side}_ankle_pitch"] = SAFE_POSTURE[f"{side}_ankle_pitch"]
            j[f"{side}_shoulder_pitch"] = SAFE_POSTURE[f"{side}_shoulder_pitch"]
            j[f"{side}_elbow"] = SAFE_POSTURE[f"{side}_elbow"]
        return j

    if skill in ("walk", "turn", "run", "push_recovery"):
        base_amp = 0.30 if skill in ("walk", "turn") else 0.44
        if skill == "push_recovery":
            base_amp *= 1.12

        speed_factor = 0.90 + 0.15 * clamp_scalar(speed, 0.0, 1.6)
        amp = base_amp * speed_factor
        knee_amp = 0.80 if skill in ("walk", "turn") else 1.05
        arm_amp = 0.18 if skill != "run" else 0.28

        for side, offset in (("left", 0.0), ("right", math.pi)):
            ph = phase + offset
            swing = max(0.0, math.sin(ph))
            stance = max(0.0, -math.sin(ph))

            j[f"{side}_hip_pitch"] = 0.03 + amp * math.sin(ph)
            j[f"{side}_hip_roll"] = (0.035 if side == "left" else -0.035) * math.sin(2.0 * ph)
            j[f"{side}_knee"] = 0.14 + knee_amp * swing + 0.08 * stance
            j[f"{side}_ankle_pitch"] = (
                -0.16 * j[f"{side}_hip_pitch"]
                + 0.20 * swing
                - 0.08 * stance
            )

            # Arms swing opposite to legs.
            arm_ph = phase + (math.pi if side == "left" else 0.0)
            j[f"{side}_shoulder_pitch"] = -0.06 - arm_amp * math.sin(arm_ph)
            j[f"{side}_elbow"] = 0.22 + 0.28 * max(0.0, math.sin(arm_ph + math.pi / 2.0))

        if skill == "push_recovery":
            j["left_hip_pitch"] += 0.12 * disturbance
            j["right_hip_pitch"] -= 0.12 * disturbance
            j["left_knee"] += 0.18 * abs(disturbance)
            j["right_knee"] += 0.18 * abs(disturbance)

    elif skill == "jump":
        local = t - SKILL_START["jump"]

        crouch = smoothstep(local / 0.30)
        launch = smoothstep((local - 0.30) / 0.18)
        air = clamp_scalar((local - 0.48) / 0.60, 0.0, 1.0)
        land = smoothstep((local - 1.08) / 0.30)

        for side in ("left", "right"):
            j[f"{side}_hip_pitch"] = (
                0.03
                + 0.28 * crouch
                - 0.22 * launch
                + 0.08 * air
                + 0.22 * land
            )
            j[f"{side}_knee"] = (
                0.14
                + 1.15 * crouch
                - 0.95 * launch
                + 0.25 * air
                + 0.95 * land
            )
            j[f"{side}_ankle_pitch"] = (
                -0.04
                - 0.22 * crouch
                + 0.18 * launch
                - 0.08 * air
                - 0.20 * land
            )
            j[f"{side}_shoulder_pitch"] = (
                -0.06
                - 0.45 * crouch
                + 0.75 * launch
                + 0.18 * air
                - 0.25 * land
            )
            j[f"{side}_elbow"] = (
                0.20
                + 0.35 * crouch
                + 0.55 * launch
                + 0.25 * air
                - 0.20 * land
            )

    elif skill == "fall":
        local = t - SKILL_START["fall"]
        p = smoothstep(local / 0.75)

        for side in ("left", "right"):
            j[f"{side}_hip_pitch"] = 0.03 + 0.55 * p
            j[f"{side}_hip_roll"] = 0.02 * p if side == "left" else -0.02 * p
            j[f"{side}_knee"] = 0.14 + 1.35 * p
            j[f"{side}_ankle_pitch"] = -0.04 + 0.28 * p
            j[f"{side}_shoulder_pitch"] = -0.06 - 0.92 * p
            j[f"{side}_elbow"] = 0.20 + 0.95 * p

    elif skill == "stand_up":
        local = t - SKILL_START["stand_up"]

        # Stage 1: recover from fallen posture into a crouch.
        # Stage 2: rise from crouch to standing.
        p1 = smoothstep(local / 0.75)
        p2 = smoothstep((local - 0.75) / 1.45)

        for side in ("left", "right"):
            j[f"{side}_hip_pitch"] = (
                0.55 * (1.0 - p1)
                + 0.32 * p1 * (1.0 - p2)
                + 0.04 * p2
            )
            j[f"{side}_hip_roll"] = 0.0
            j[f"{side}_knee"] = (
                1.45 * (1.0 - p1)
                + 1.10 * p1 * (1.0 - p2)
                + 0.15 * p2
            )
            j[f"{side}_ankle_pitch"] = (
                0.25 * (1.0 - p1)
                + 0.05 * p1 * (1.0 - p2)
                - 0.04 * p2
            )
            j[f"{side}_shoulder_pitch"] = (
                -0.95 * (1.0 - p1)
                - 0.25 * p1 * (1.0 - p2)
                - 0.06 * p2
            )
            j[f"{side}_elbow"] = (
                1.15 * (1.0 - p1)
                + 0.65 * p1 * (1.0 - p2)
                + 0.20 * p2
            )

    elif skill == "kick":
        local = t - SKILL_START["kick"]

        rise = smoothstep(local / 0.28)
        peak = smoothstep((local - 0.28) / 0.18)
        fall_back = smoothstep((local - 0.75) / 0.75)
        swing = clamp_scalar(rise + peak - 2.0 * fall_back, 0.0, 1.0)

        j["right_hip_pitch"] = 0.03 + 1.05 * swing
        j["right_knee"] = (
            0.14
            + 0.80 * rise * (1.0 - peak)
            + 0.05 * peak
            + 0.35 * fall_back
        )
        j["right_ankle_pitch"] = -0.04 - 0.25 * swing

        j["left_hip_pitch"] = 0.03 - 0.12 * swing
        j["left_knee"] = 0.14 + 0.22 * swing

        j["right_shoulder_pitch"] = -0.06 - 0.35 * swing
        j["left_shoulder_pitch"] = -0.06 + 0.42 * swing
        j["right_elbow"] = 0.20 + 0.45 * swing
        j["left_elbow"] = 0.20 + 0.35 * swing

    elif skill == "hold":
        # Gentle idle breathing.
        for side in ("left", "right"):
            j[f"{side}_hip_pitch"] += 0.008 * math.sin(t * 1.2)
            j[f"{side}_knee"] += 0.010 * math.sin(t * 1.7 + 0.4)
            j[f"{side}_shoulder_pitch"] += 0.006 * math.sin(t * 1.1 + 1.0)

    # Enforce limits in one place.
    for name in list(j.keys()):
        lim = JOINT_LIMITS[name]
        j[name] = clamp_scalar(j[name], lim["pos_min"], lim["pos_max"])

    return j


# =============================================================================
# Simple 3D humanoid projection for rendered demo video
# =============================================================================

def robot_body_points(joints: Dict[str, float], links: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Forward kinematics for a simplified biped in body coordinates."""
    pts: Dict[str, np.ndarray] = {}

    pts["pelvis"] = np.array([0.0, 0.0, 0.0], dtype=float)
    pts["chest"] = np.array([0.0, 0.0, links["torso"]], dtype=float)
    pts["head"] = np.array([0.0, 0.0, links["torso"] + links["head"]], dtype=float)

    for side, sign in (("left", 1.0), ("right", -1.0)):
        # Arm.
        shoulder = np.array([0.0, sign * links["shoulder_width"], links["torso"]], dtype=float)
        sp = joints[f"{side}_shoulder_pitch"]
        el = joints[f"{side}_elbow"]

        upper_arm = np.array(
            [
                links["upper_arm"] * math.sin(sp),
                sign * 0.03,
                -links["upper_arm"] * math.cos(sp),
            ],
            dtype=float,
        )
        elbow = shoulder + upper_arm

        lower_arm = np.array(
            [
                links["lower_arm"] * math.sin(sp - el),
                sign * 0.03,
                -links["lower_arm"] * math.cos(sp - el),
            ],
            dtype=float,
        )
        wrist = elbow + lower_arm

        pts[f"{side}_shoulder"] = shoulder
        pts[f"{side}_elbow"] = elbow
        pts[f"{side}_wrist"] = wrist

        # Leg.
        hip = np.array([0.0, sign * links["hip_width"], 0.0], dtype=float)
        hp = joints[f"{side}_hip_pitch"]
        hr = joints[f"{side}_hip_roll"]
        kn = joints[f"{side}_knee"]
        an = joints[f"{side}_ankle_pitch"]

        thigh = np.array(
            [
                links["thigh"] * math.sin(hp),
                sign * links["thigh"] * math.sin(hr),
                -links["thigh"] * math.cos(hp) * math.cos(hr),
            ],
            dtype=float,
        )
        knee_pos = hip + thigh

        shank_angle = hp - kn
        shank = np.array(
            [
                links["shank"] * math.sin(shank_angle),
                sign * links["shank"] * math.sin(hr) * 0.5,
                -links["shank"] * math.cos(shank_angle) * math.cos(hr),
            ],
            dtype=float,
        )
        ankle_pos = knee_pos + shank

        toe = ankle_pos + np.array(
            [
                links["foot"] * math.cos(an),
                0.0,
                links["foot"] * math.sin(an),
            ],
            dtype=float,
        )
        heel = ankle_pos + np.array([-0.045, 0.0, 0.0], dtype=float)

        pts[f"{side}_hip"] = hip
        pts[f"{side}_knee"] = knee_pos
        pts[f"{side}_ankle"] = ankle_pos
        pts[f"{side}_toe"] = toe
        pts[f"{side}_heel"] = heel

    return pts


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    return rz @ ry @ rx


def project_point(
    point: np.ndarray,
    base_xy: np.ndarray,
    cam: Dict[str, Any],
) -> Tuple[int, int]:
    """Project a world point into image coordinates with a following camera."""
    rel = point - np.array([base_xy[0], base_xy[1], 0.0], dtype=float)

    az = math.radians(float(cam["azimuth_deg"]))
    el = math.radians(float(cam["elevation_deg"]))

    x1 = rel[0] * math.cos(az) + rel[1] * math.sin(az)
    y1 = -rel[0] * math.sin(az) + rel[1] * math.cos(az)
    z1 = rel[2]

    sx = float(cam["cx"]) + float(cam["scale"]) * x1
    sy = float(cam["cy"]) - float(cam["scale"]) * (z1 * math.cos(el) - y1 * math.sin(el))

    return int(round(sx)), int(round(sy))


def render_frame(i: int, df: pd.DataFrame, cam: Dict[str, Any]) -> np.ndarray:
    """Render one RGB frame from simulated telemetry."""
    width = int(cam["width"])
    height = int(cam["height"])

    img = Image.new("RGB", (width, height), (10, 14, 22))
    draw = ImageDraw.Draw(img)

    # Background.
    draw.rectangle([0, 0, width, height - 90], fill=(16, 24, 38))
    draw.rectangle([0, height - 90, width, height], fill=(34, 41, 50))
    draw.line([0, height - 90, width, height - 90], fill=(70, 85, 105), width=2)

    row = df.iloc[i]

    base = np.array(
        [
            float(row["base_x"]),
            float(row["base_y"]),
            float(row["base_z"]),
        ],
        dtype=float,
    )
    base_xy = np.array([base[0], base[1]], dtype=float)

    joints = {name: float(row[f"q_{name}"]) for name in JOINT_NAMES}
    body_pts = robot_body_points(joints, CONFIG["links"])

    rot = rotation_matrix(
        float(row["roll"]),
        float(row["pitch"]),
        float(row["yaw"]),
    )
    world_pts = {key: base + rot @ vec for key, vec in body_pts.items()}

    # Shadow.
    shadow = project_point(
        np.array([float(row["base_x"]), float(row["base_y"]), 0.02], dtype=float),
        base_xy,
        cam,
    )
    draw.ellipse(
        [shadow[0] - 48, shadow[1] - 12, shadow[0] + 48, shadow[1] + 12],
        fill=(6, 9, 13),
    )

    # Links.
    for a, b in LINK_PAIRS:
        if a not in world_pts or b not in world_pts:
            continue

        pa = project_point(world_pts[a], base_xy, cam)
        pb = project_point(world_pts[b], base_xy, cam)

        is_core = a in {"pelvis", "chest"} or b in {"pelvis", "chest"}
        line_width = 10 if is_core else 6

        if "knee" in a or "knee" in b or "elbow" in a or "elbow" in b:
            color = (90, 180, 255)
        else:
            color = (235, 240, 245)

        draw.line([pa, pb], fill=color, width=line_width)

    # Joint nodes.
    node_keys = (
        "pelvis",
        "chest",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    for key in node_keys:
        if key not in world_pts:
            continue
        p = project_point(world_pts[key], base_xy, cam)
        r = 10 if key == "head" else 5
        draw.ellipse(
            [p[0] - r, p[1] - r, p[0] + r, p[1] + r],
            fill=(255, 190, 80),
            outline=(15, 15, 15),
        )

    # Foot contact indicators.
    for side in ("left", "right"):
        contact = float(row[f"{side}_contact"]) > 0.5
        p = project_point(world_pts[f"{side}_toe"], base_xy, cam)
        color = (70, 220, 120) if contact else (255, 110, 70)
        draw.ellipse(
            [p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7],
            fill=color,
            outline=(0, 0, 0),
        )

    # HUD.
    font = ImageFont.load_default()

    draw.rectangle([0, 0, width, 82], fill=(0, 0, 0))
    draw.text(
        (8, 6),
        "Unitree G1 | simulation only | no hardware",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text(
        (8, 24),
        f"t={float(row['time']):05.2f}s  skill={row['skill']}",
        fill=(150, 220, 255),
        font=font,
    )
    draw.text(
        (8, 42),
        (
            f"cmd vx={float(row['cmd_vx']):.2f}  "
            f"actual vx={float(row['actual_vx']):.2f}  "
            f"est vx={float(row['vx_est']):.2f} m/s"
        ),
        fill=(210, 210, 210),
        font=font,
    )
    draw.text(
        (8, 60),
        (
            f"max torque={float(row['max_tau_norm']) * 100.0:.0f}%  "
            f"z={float(row['base_z']):.2f} m  "
            f"pitch={float(row['pitch']):.2f} rad"
        ),
        fill=(210, 210, 210),
        font=font,
    )

    if float(row["estop"]) > 0.5:
        draw.text(
            (width - 150, 10),
            "E-STOP DAMPING",
            fill=(255, 80, 80),
            font=font,
        )

    # Bottom progress bar.
    total_time = float(df["time"].iloc[-1]) if len(df) > 1 else 1.0
    frac = float(row["time"]) / max(total_time, 1e-6)

    draw.rectangle([0, height - 30, width, height], fill=(0, 0, 0))
    draw.rectangle(
        [
            8,
            height - 20,
            8 + int((width - 16) * frac),
            height - 10,
        ],
        fill=(80, 200, 255),
    )
    draw.text(
        (width - 190, height - 25),
        "LIVE DEMO | MP4 GENERATED",
        fill=(255, 210, 120),
        font=font,
    )

    return np.asarray(img, dtype=np.uint8)


def encode_mp4(df: pd.DataFrame, fps: int, cam: Dict[str, Any]) -> bytes:
    """Encode rendered frames to MP4 bytes."""
    last_error: Exception | None = None

    for codec in ("libx264", "mpeg4"):
        buf = io.BytesIO()
        writer = None

        try:
            writer = imageio.get_writer(
                buf,
                format="mp4",
                mode="I",
                fps=fps,
                codec=codec,
                pixelformat="yuv420p",
            )

            for i in range(len(df)):
                frame = render_frame(i, df, cam)
                writer.append_data(frame)

            writer.close()
            writer = None

            buf.seek(0)
            return buf.getvalue()

        except Exception as exc:
            last_error = exc
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

    raise RuntimeError(f"MP4 encoding failed: {last_error}")


# =============================================================================
# Full simulated telemetry generator
# =============================================================================

def run_simulation(
    seed: int,
    fps: int,
    duration: float,
    speed_scale: float,
    push_force: float,
    noise_level: float,
    estop: bool,
) -> Dict[str, Any]:
    """Generate simulated G1 telemetry and an MP4 demo."""
    rng = np.random.default_rng(int(seed))

    n = max(3, int(round(float(duration) * float(fps))))
    dt = 1.0 / float(fps)
    t = np.arange(n, dtype=float) * dt

    num_keys = [
        "time",
        "cmd_vx",
        "cmd_vy",
        "cmd_yaw",
        "actual_vx",
        "actual_vy",
        "actual_yaw_rate",
        "base_x",
        "base_y",
        "base_z",
        "pitch",
        "roll",
        "yaw",
        "imu_ax",
        "imu_ay",
        "imu_az",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "vx_est",
        "error_vx",
        "max_tau_norm",
        "left_contact",
        "right_contact",
        "estop",
    ]

    A: Dict[str, np.ndarray] = {k: np.zeros(n, dtype=float) for k in num_keys}
    skill_list: List[str] = []
    phase_arr = np.zeros(n, dtype=float)

    q_des: Dict[str, np.ndarray] = {name: np.zeros(n, dtype=float) for name in JOINT_NAMES}

    phase = 0.0

    x = 0.0
    y = 0.0
    psi = 0.0
    theta = 0.0
    phi = 0.0
    z = 0.75

    vx = 0.0
    vy = 0.0
    yaw_rate = 0.0

    for i in range(n):
        ti = float(t[i])
        prof = command_profile(ti, speed_scale)
        skill = str(prof["skill"])
        skill_list.append(skill)

        cmd_vx = float(prof["vx"])
        cmd_vy = float(prof["vy"])
        cmd_yaw = float(prof["yaw_rate"])

        disturbance = 0.0
        if skill == "push_recovery":
            local = ti - SKILL_START["push_recovery"]
            local = max(local, 0.0)
            disturbance = float(
                push_force * math.exp(-local * 3.0) * math.sin(local * 16.0)
            )

        if estop:
            cmd_vx = 0.0
            cmd_vy = 0.0
            cmd_yaw = 0.0
            disturbance = 0.0

        # Continuous gait phase.
        if skill in ("walk", "turn", "run", "push_recovery", "kick"):
            gait_freq = 0.85 + abs(cmd_vx) / 0.55
        else:
            gait_freq = 0.0

        phase += 2.0 * math.pi * gait_freq * dt
        phase_arr[i] = phase

        # First-order velocity response plus push disturbance.
        effective_cmd_vx = cmd_vx
        if skill == "push_recovery":
            effective_cmd_vx += 0.25 * disturbance

        vx += (effective_cmd_vx - vx) * 0.22
        vy += (cmd_vy - vy) * 0.22
        yaw_rate += (cmd_yaw - yaw_rate) * 0.25

        x += vx * dt
        y += vy * dt
        psi += yaw_rate * dt

        # Base pose profiles.
        z_nom = 0.75
        theta_nom = 0.0
        phi_nom = 0.015 * math.sin(0.8 * ti)

        if skill == "jump":
            local = ti - SKILL_START["jump"]

            if local < 0.30:
                z_nom = 0.75 - 0.11 * smoothstep(local / 0.30)
            elif local < 1.08:
                z_nom = 0.64 + 0.31 * math.sin(math.pi * (local - 0.30) / 0.78)
            else:
                z_nom = 0.75 + 0.035 * math.exp(-(local - 1.08) * 7.0) * math.sin(
                    (local - 1.08) * 22.0
                )

        elif skill == "push_recovery":
            theta_nom = 0.20 * disturbance
            z_nom = 0.75 - 0.035 * abs(disturbance)

        elif skill == "fall":
            local = ti - SKILL_START["fall"]
            p = smoothstep(local / 0.75)
            z_nom = 0.75 * (1.0 - p) + 0.17 * p
            theta_nom = 1.35 * p
            phi_nom = 0.12 * p

        elif skill == "stand_up":
            local = ti - SKILL_START["stand_up"]
            p1 = smoothstep(local / 0.75)
            p2 = smoothstep((local - 0.75) / 1.45)

            z_nom = (
                0.17
                + (0.58 - 0.17) * p1
                + (0.75 - 0.58) * p2
            )
            theta_nom = (
                1.35 * (1.0 - p1)
                + 0.25 * p1 * (1.0 - p2)
                + 0.0 * p2
            )
            phi_nom = 0.12 * (1.0 - p1)

        elif skill == "kick":
            local = ti - SKILL_START["kick"]
            z_nom = 0.75 + 0.02 * math.sin(local * 8.0)

        if estop:
            z_nom = 0.72
            theta_nom = 0.0
            phi_nom = 0.0

        z = z_nom
        theta = theta_nom
        phi = phi_nom

        A["time"][i] = ti
        A["cmd_vx"][i] = cmd_vx
        A["cmd_vy"][i] = cmd_vy
        A["cmd_yaw"][i] = cmd_yaw
        A["actual_vx"][i] = vx
        A["actual_vy"][i] = vy
        A["actual_yaw_rate"][i] = yaw_rate
        A["base_x"][i] = x
        A["base_y"][i] = y
        A["base_z"][i] = z
        A["pitch"][i] = theta
        A["roll"][i] = phi
        A["yaw"][i] = psi
        A["estop"][i] = 1.0 if estop else 0.0

        jd = desired_joints(
            t=ti,
            skill=skill,
            phase=phase,
            speed=abs(vx),
            disturbance=disturbance,
            estop=estop,
        )

        for name in JOINT_NAMES:
            q_des[name][i] = jd[name]

    # Apply motor delay, sensor noise, and one-place limit enforcement.
    delay_steps = int(round(float(CONFIG["motor_delay_ms"]) * float(fps) / 1000.0))
    delay_steps = max(0, min(n - 1, delay_steps))

    q: Dict[str, np.ndarray] = {}
    dq: Dict[str, np.ndarray] = {}
    tau: Dict[str, np.ndarray] = {}

    for name in JOINT_NAMES:
        lim = JOINT_LIMITS[name]
        desired = q_des[name]

        if delay_steps == 0:
            delayed = desired.copy()
        else:
            delayed = np.empty_like(desired)
            delayed[:delay_steps] = desired[0]
            delayed[delay_steps:] = desired[:-delay_steps]

        joint_noise = rng.normal(
            0.0,
            float(CONFIG["sensor_noise"]["joint_pos"]) * float(noise_level),
            n,
        )
        actual = delayed + joint_noise
        actual = np.clip(actual, lim["pos_min"], lim["pos_max"])
        q[name] = actual

        desired_vel = np.gradient(desired, dt)
        actual_vel = np.gradient(actual, dt)
        actual_vel = np.clip(actual_vel, -lim["vel_max"], lim["vel_max"])
        dq[name] = actual_vel

        if estop:
            tau_raw = (
                18.0 * (SAFE_POSTURE[name] - actual)
                - 7.0 * actual_vel
            )
        else:
            tau_noise = rng.normal(0.0, 0.35 * float(noise_level), n)
            tau_raw = (
                lim["kp"] * (desired - actual)
                + lim["kd"] * (desired_vel - actual_vel)
                + tau_noise
            )

        tau[name] = np.clip(tau_raw, -lim["torque_max"], lim["torque_max"])

    # IMU and simple velocity estimator.
    acc_x = np.gradient(A["actual_vx"], dt)
    acc_y = np.gradient(A["actual_vy"], dt)
    acc_z = np.gradient(A["base_z"], dt)

    g = 9.81

    A["imu_ax"] = (
        acc_x
        + g * np.sin(A["pitch"])
        + rng.normal(0.0, float(CONFIG["sensor_noise"]["imu_acc"]) * float(noise_level), n)
    )
    A["imu_ay"] = (
        acc_y
        - g * np.sin(A["roll"])
        + rng.normal(0.0, float(CONFIG["sensor_noise"]["imu_acc"]) * float(noise_level), n)
    )
    A["imu_az"] = (
        acc_z
        + g * np.cos(A["pitch"]) * np.cos(A["roll"])
        + rng.normal(0.0, float(CONFIG["sensor_noise"]["imu_acc"]) * float(noise_level), n)
    )

    A["gyro_x"] = (
        np.gradient(A["roll"], dt)
        + rng.normal(0.0, float(CONFIG["sensor_noise"]["gyro"]) * float(noise_level), n)
    )
    A["gyro_y"] = (
        np.gradient(A["pitch"], dt)
        + rng.normal(0.0, float(CONFIG["sensor_noise"]["gyro"]) * float(noise_level), n)
    )
    A["gyro_z"] = (
        np.gradient(A["yaw"], dt)
        + rng.normal(0.0, float(CONFIG["sensor_noise"]["gyro"]) * float(noise_level), n)
    )

    vx_est = 0.0
    for i in range(n):
        meas = A["actual_vx"][i] + rng.normal(
            0.0,
            float(CONFIG["sensor_noise"]["velocity_est"]) * float(noise_level),
        )
        vx_est = 0.55 * vx_est + 0.45 * meas
        A["vx_est"][i] = vx_est

    A["error_vx"] = np.abs(A["vx_est"] - A["actual_vx"])

    # Contact flags.
    for i in range(n):
        ti = float(t[i])
        skill = skill_list[i]

        left_contact = 1.0
        right_contact = 1.0

        if skill in ("walk", "turn", "run", "push_recovery"):
            left_contact = 0.0 if math.sin(phase_arr[i]) > 0.12 else 1.0
            right_contact = 0.0 if math.sin(phase_arr[i] + math.pi) > 0.12 else 1.0

        elif skill == "jump":
            local = ti - SKILL_START["jump"]
            airborne = 0.38 < local < 1.05
            left_contact = 0.0 if airborne else 1.0
            right_contact = 0.0 if airborne else 1.0

        elif skill == "fall":
            left_contact = 0.0
            right_contact = 0.0

        elif skill == "stand_up":
            local = ti - SKILL_START["stand_up"]
            contacted = local > 1.9
            left_contact = 1.0 if contacted else 0.0
            right_contact = 1.0 if contacted else 0.0

        elif skill == "kick":
            local = ti - SKILL_START["kick"]
            right_air = 0.22 < local < 1.05
            left_contact = 1.0
            right_contact = 0.0 if right_air else 1.0

        elif skill == "hold":
            left_contact = 1.0
            right_contact = 1.0

        A["left_contact"][i] = left_contact
        A["right_contact"][i] = right_contact

    # Torque utilization.
    max_tau_norm = np.zeros(n, dtype=float)
    for name in JOINT_NAMES:
        limit = float(JOINT_LIMITS[name]["torque_max"])
        max_tau_norm = np.maximum(max_tau_norm, np.abs(tau[name]) / limit)

    A["max_tau_norm"] = max_tau_norm

    # Assemble DataFrame.
    df = pd.DataFrame(A)
    df.insert(1, "skill", skill_list)

    for name in JOINT_NAMES:
        df[f"q_{name}"] = q[name]
        df[f"qdes_{name}"] = q_des[name]
        df[f"dq_{name}"] = dq[name]
        df[f"tau_{name}"] = tau[name]

    # Render MP4.
    encoding_error: str | None = None
    try:
        mp4 = encode_mp4(df, fps=fps, cam=CONFIG["camera"])
    except Exception as exc:
        mp4 = b""
        encoding_error = str(exc)

    summary = {
        "duration_s": float(duration),
        "frames": int(n),
        "fps": int(fps),
        "seed": int(seed),
        "speed_scale": float(speed_scale),
        "push_force": float(push_force),
        "noise_level": float(noise_level),
        "estop_enabled": bool(estop),
        "motor_delay_steps": int(delay_steps),
        "mean_abs_velocity_error_m_s": float(np.mean(A["error_vx"])),
        "max_tau_utilization_percent": float(np.max(A["max_tau_norm"]) * 100.0),
        "left_contact_ratio": float(np.mean(A["left_contact"])),
        "right_contact_ratio": float(np.mean(A["right_contact"])),
        "joint_limits_enforced_in_one_place": True,
        "hardware_used": False,
    }

    events = [
        {"skill": name, "start_s": float(start), "end_s": float(end)}
        for name, start, end in BASE_SKILLS
    ]

    log = {
        "meta": {
            "project": CONFIG["project"],
            "mode": "simulation_only",
            "robot": CONFIG["robot"],
            "safety": CONFIG["safety"],
        },
        "events": events,
        "summary": summary,
        "joint_limits": JOINT_LIMITS,
        "config": CONFIG,
    }

    if encoding_error is not None:
        log["encoding_error"] = encoding_error

    return {
        "df": df,
        "mp4": mp4,
        "log": log,
        "summary": summary,
    }


# =============================================================================
# Streamlit UI
# =============================================================================

def main() -> None:
    st.title("🤖 Unitree G1 Dynamic Movement — Simulation-Only Demo")
    st.caption(
        "This app auto-generates a simulated walking / running / jumping / push-recovery / "
        "fall-recovery / kick sequence and renders an MP4 live in the browser. "
        "No hardware is used. No API key is required."
    )

    if "result" not in st.session_state:
        st.session_state["result"] = None

    with st.sidebar:
        st.header("Demo Controls")

        seed = int(
            st.number_input(
                "Random seed",
                value=int(CONFIG["simulation"]["seed"]),
                step=1,
            )
        )

        fps = int(
            st.selectbox(
                "Render FPS",
                options=[15, 24, 30],
                index=2,
            )
        )

        duration = float(
            st.slider(
                "Demo duration (s)",
                min_value=12.0,
                max_value=20.0,
                value=float(CONFIG["simulation"]["duration_s"]),
                step=0.5,
            )
        )

        speed_scale = float(
            st.slider(
                "Speed scale",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.05,
            )
        )

        push_force = float(
            st.slider(
                "Push intensity",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
            )
        )

        noise_level = float(
            st.slider(
                "Sensor noise level",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
            )
        )

        estop = bool(
            st.checkbox(
                "Emergency stop / damping mode",
                value=False,
                help="Forces a safe damping posture and zero velocity commands.",
            )
        )

        generate = st.button(
            "Generate / regenerate demo",
            type="primary",
            use_container_width=True,
        )

        need_generate = st.session_state["result"] is None or bool(generate)

        if need_generate:
            with st.spinner("Loading simulated telemetry and rendering MP4..."):
                st.session_state["result"] = run_simulation(
                    seed=seed,
                    fps=fps,
                    duration=duration,
                    speed_scale=speed_scale,
                    push_force=push_force,
                    noise_level=noise_level,
                    estop=estop,
                )

    result = st.session_state["result"]

    if result is None:
        st.info("Press Generate to build the demo.")
        return

    df: pd.DataFrame = result["df"]
    mp4: bytes = result["mp4"]
    log: Dict[str, Any] = result["log"]
    summary: Dict[str, Any] = result["summary"]

    tab_video, tab_telemetry, tab_safety, tab_logs = st.tabs(
        ["Live demo", "Telemetry", "Safety & limits", "Logs"]
    )

    # ---------------------------------------------------------------------
    # Live demo tab
    # ---------------------------------------------------------------------
    with tab_video:
        st.subheader("Generated MP4 demo")

        if mp4:
            try:
                st.video(
                    mp4,
                    format="video/mp4",
                    autoplay=True,
                    muted=True,
                    loop=True,
                )
            except Exception:
                st.video(mp4, format="video/mp4")

            st.download_button(
                label="Download MP4",
                data=mp4,
                file_name="unitree_g1_dynamic_demo.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        else:
            st.error(
                "MP4 encoding failed on this machine. Telemetry is still available. "
                "See the Logs tab for the encoding error."
            )

        st.markdown("### Scripted skill timeline")
        timeline_df = pd.DataFrame(
            BASE_SKILLS,
            columns=["skill", "start_s", "end_s"],
        )
        st.dataframe(timeline_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{summary['duration_s']:.1f} s")
        c2.metric("Frames", f"{summary['frames']}")
        c3.metric("Mean velocity error", f"{summary['mean_abs_velocity_error_m_s']:.3f} m/s")
        c4.metric("Peak torque use", f"{summary['max_tau_utilization_percent']:.1f}%")

    # ---------------------------------------------------------------------
    # Telemetry tab
    # ---------------------------------------------------------------------
    with tab_telemetry:
        st.subheader("Velocity tracking")
        vel_chart = df[["time", "cmd_vx", "actual_vx", "vx_est"]].set_index("time")
        st.line_chart(vel_chart)

        st.subheader("Base pose")
        pose_chart = df[["time", "base_z", "pitch", "roll"]].set_index("time")
        st.line_chart(pose_chart)

        st.subheader("Torque utilization")
        torque_chart = df[["time", "max_tau_norm"]].set_index("time")
        st.line_chart(torque_chart)

        st.subheader("Foot contacts")
        contact_chart = df[["time", "left_contact", "right_contact"]].set_index("time")
        st.line_chart(contact_chart)

        st.subheader("Joint inspector")
        selected_joint = st.selectbox("Inspect joint", JOINT_NAMES)
        lim = JOINT_LIMITS[selected_joint]

        joint_pose = pd.DataFrame(
            {
                "time": df["time"],
                "q_actual": df[f"q_{selected_joint}"],
                "q_desired": df[f"qdes_{selected_joint}"],
                "lower_limit": lim["pos_min"],
                "upper_limit": lim["pos_max"],
            }
        ).set_index("time")
        st.line_chart(joint_pose)

        joint_torque = pd.DataFrame(
            {
                "time": df["time"],
                "torque": df[f"tau_{selected_joint}"],
                "positive_limit": lim["torque_max"],
                "negative_limit": -lim["torque_max"],
            }
        ).set_index("time")
        st.line_chart(joint_torque)

    # ---------------------------------------------------------------------
    # Safety tab
    # ---------------------------------------------------------------------
    with tab_safety:
        st.subheader("Safety posture")
        st.write(
            "This is a simulation-only demo. Joint position, velocity, and torque limits "
            "are enforced in one place. The emergency-stop path drops commands to a damping "
            "posture. Hardware testing must be a separate human-reviewed step."
        )

        max_position_violation = 0.0

        for name in JOINT_NAMES:
            limit = JOINT_LIMITS[name]
            q_col = df[f"q_{name}"]

            q_min = float(q_col.min())
            q_max = float(q_col.max())

            violation = max(
                float(limit["pos_min"]) - q_min,
                q_max - float(limit["pos_max"]),
                0.0,
            )
            max_position_violation = max(max_position_violation, violation)

        peak_torque = float(df["max_tau_norm"].max())
        vel_mae = float(df["error_vx"].mean())

        m1, m2, m3 = st.columns(3)
        m1.metric("Max joint position violation", f"{max_position_violation:.6f} rad")
        m2.metric("Peak torque utilization", f"{peak_torque * 100.0:.1f}%")
        m3.metric("Mean absolute velocity error", f"{vel_mae:.3f} m/s")

        st.subheader("Unified limit table")
        limit_df = pd.DataFrame.from_dict(JOINT_LIMITS, orient="index")
        limit_df.index.name = "joint"
        limit_df = limit_df.reset_index()
        st.dataframe(limit_df)

        st.subheader("Emergency-stop behavior")
        st.json(CONFIG["safety"])

    # ---------------------------------------------------------------------
    # Logs tab
    # ---------------------------------------------------------------------
    with tab_logs:
        st.subheader("Downloads")

        st.download_button(
            label="Download telemetry CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="unitree_g1_sim_telemetry.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            label="Download JSON log",
            data=json.dumps(log, indent=2).encode("utf-8"),
            file_name="unitree_g1_sim_log.json",
            mime="application/json",
            use_container_width=True,
        )

        st.subheader("JSON log")
        st.json(log)


main()