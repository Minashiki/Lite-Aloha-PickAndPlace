"""M1 场景检查：模型摘要、工作空间跟踪、抓取和录像。

子命令都从 keyframe ``home`` 出发：``mj_resetDataKeyframe``，把 mocap 对齐到
pinch，再 ``mj_forward``。路径从本文件位置推导。任一验收失败则退出码为 1。
``view`` 会打开窗口，本里程碑的自动测试不调用它。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pick_place.xml"
RUN_DIR = ROOT / "runs" / "m1"

# gripper_base 坐标系下的 pinch。张开时两指垫片质心，x=0 为开合中线。
PINCH_POS = np.array([0.0, 0.003636, 0.108806])
# 阻尼最小二乘 IK：pinch 在 (0.30, 0, 0.20)、姿态为 GRASP_QUAT。6 个臂关节 + 两指张开。
HOME_QPOS = np.array(
    [
        4.0629901495e-04,
        -4.7747780177e-02,
        1.0111069065e-01,
        -3.1106394060e-07,
        1.5182284990e00,
        -3.9001127121e-04,
        0.037,
        0.037,
    ]
)
# pinch 系：接近方向为世界 -z，开合方向为世界 y（x 轴指向世界 -y）。
GRASP_QUAT = np.array([0.0, -0.7071067811865475, 0.7071067811865476, 0.0])

# 跟踪网格用的最终裁剪盒。整盒 plan 初值有不可达角点，缩小后全部合格。
WS_LOW = np.array([0.15, -0.15, 0.02])
WS_HIGH = np.array([0.40, 0.15, 0.30])

# 任务规定的 5 个抓取位置，不随裁剪盒缩小。
GRASP_XYS = (
    (0.35, 0.0),
    (0.25, 0.15),
    (0.25, -0.15),
    (0.45, 0.15),
    (0.45, -0.15),
)

OBJ_Z = 0.02
APPROACH_Z = OBJ_Z + 0.08
LIFT_Z = OBJ_Z + 0.15
GRIP_OPEN = 0.037
GRIP_CLOSE = 0.002
MAX_MOCAP_STEP = 0.05
CTRL_DT = 0.04

POS_LIMIT_MM = 5.0
ROT_LIMIT_DEG = 3.0
FLUT_POS_MM = 2.0
FLUT_ROT_DEG = 1.0
DROP_LIMIT_M = 0.01
LIFT_MIN_M = 0.14
PEN_LIMIT_M = 0.005
WELD_LIMIT_MM = 1.0


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def emit(line: str = "") -> None:
    print(line, flush=True)


def load_model():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def _id(model, obj_type, name: str) -> int:
    import mujoco

    index = mujoco.mj_name2id(model, obj_type, name)
    if index < 0:
        raise RuntimeError(f"模型里没有 {name}")
    return int(index)


class Ids:
    def __init__(self, model):
        import mujoco

        self.site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "left/pinch")
        self.pinch = _id(model, mujoco.mjtObj.mjOBJ_BODY, "left/pinch")
        self.gripper_base = _id(model, mujoco.mjtObj.mjOBJ_BODY, "left/gripper_base")
        self.mocap_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
        self.mocap = int(model.body_mocapid[self.mocap_body])
        self.obj_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_site = _id(model, mujoco.mjtObj.mjOBJ_SITE, "obj_site")
        self.obj_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
        self.obj_q = int(model.jnt_qposadr[self.obj_joint])
        self.key = _id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.weld = next(
            i for i in range(model.neq) if int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_WELD)
        )


def site_quat(model, data, site_id: int) -> np.ndarray:
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array(data.site_xmat[site_id], dtype=float))
    return quat


def rotvec(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """World-frame rotation that takes ``q_from`` onto ``q_to``."""
    import mujoco

    qneg = np.zeros(4)
    qdif = np.zeros(4)
    mujoco.mju_negQuat(qneg, q_from)
    mujoco.mju_mulQuat(qdif, q_to, qneg)
    if qdif[0] < 0.0:
        qdif *= -1.0
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, qdif, 1.0)
    return err


def angle_deg(q_from: np.ndarray, q_to: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(rotvec(q_from, q_to))))


def reset_home(model, data, ids: Ids) -> None:
    """Keyframe reset, then match mocap to the pinch body before the first step."""
    import mujoco

    mujoco.mj_resetDataKeyframe(model, data, ids.key)
    mujoco.mj_forward(model, data)
    data.mocap_pos[ids.mocap] = data.xpos[ids.pinch]
    data.mocap_quat[ids.mocap] = data.xquat[ids.pinch]
    mujoco.mj_forward(model, data)


def set_object(model, data, ids: Ids, pos, quat=None) -> None:
    import mujoco

    adr = ids.obj_q
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = quat if quat is not None else (1.0, 0.0, 0.0, 0.0)
    dof = int(model.jnt_dofadr[ids.obj_joint])
    data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def weld_errors(model, data, ids: Ids) -> tuple[float, float, np.ndarray]:
    """Return position residual (mm), rotation residual (deg), and the 6 efc values."""
    import mujoco

    rows = []
    for i in range(data.nefc):
        if int(data.efc_type[i]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        if int(data.efc_id[i]) != ids.weld:
            continue
        rows.append(float(data.efc_pos[i]))
    vals = np.array(rows, dtype=float)
    if vals.size != 6:
        raise RuntimeError(f"weld 约束残差不是 6 维，而是 {vals.size}")
    pos_mm = float(np.linalg.norm(vals[:3]) * 1000.0)
    rot_deg = float(np.degrees(np.linalg.norm(vals[3:])))
    return pos_mm, rot_deg, vals


def max_penetration(data) -> float:
    if data.ncon <= 0:
        return 0.0
    dist = np.asarray(data.contact.dist[: data.ncon], dtype=float)
    return float(np.max(np.maximum(0.0, -dist)))


def finite_state(data) -> bool:
    return bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())


def body_name(model, geom_id: int) -> str:
    import mujoco

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
    return name or "world"


def worst_contact(model, data) -> tuple[float, str]:
    if data.ncon <= 0:
        return 0.0, ""
    dist = np.asarray(data.contact.dist[: data.ncon], dtype=float)
    pen = np.maximum(0.0, -dist)
    index = int(np.argmax(pen))
    contact = data.contact[index]
    pair = f"{body_name(model, contact.geom1)}/{body_name(model, contact.geom2)}"
    return float(pen[index]), pair


def n_substeps(model) -> int:
    return int(round(CTRL_DT / float(model.opt.timestep)))


def step_hold(model, data, ids: Ids, pos, ctrl: float, n: int, hook=None) -> float:
    pen = 0.0
    for _ in range(n):
        data.ctrl[0] = ctrl
        data.mocap_pos[ids.mocap] = pos
        data.mocap_quat[ids.mocap] = GRASP_QUAT
        import mujoco

        mujoco.mj_step(model, data)
        pen = max(pen, max_penetration(data))
        if hook is not None:
            hook()
    return pen


def move_toward(model, data, ids: Ids, target, max_step: float, ctrl: float, hook=None) -> tuple[float, float]:
    """Move mocap at most ``max_step`` meters per control interval. Returns (pen, max interval step)."""
    import mujoco

    pen = 0.0
    largest = 0.0
    sub = n_substeps(model)
    for _ in range(800):
        cur = data.mocap_pos[ids.mocap].copy()
        delta = target - cur
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-4:
            data.mocap_pos[ids.mocap] = target
            data.mocap_quat[ids.mocap] = GRASP_QUAT
            data.ctrl[0] = ctrl
            mujoco.mj_step(model, data)
            pen = max(pen, max_penetration(data))
            if hook is not None:
                hook()
            break
        if dist > max_step:
            delta = delta * (max_step / dist)
            dist = max_step
        largest = max(largest, dist)
        for s in range(sub):
            data.mocap_pos[ids.mocap] = cur + delta * ((s + 1) / sub)
            data.mocap_quat[ids.mocap] = GRASP_QUAT
            data.ctrl[0] = ctrl
            mujoco.mj_step(model, data)
            pen = max(pen, max_penetration(data))
            if hook is not None:
                hook()
    return pen, largest


def run_grasp(model, data, ids: Ids, xy, hook=None) -> dict:
    import mujoco

    reset_home(model, data, ids)
    set_object(model, data, ids, (xy[0], xy[1], OBJ_Z))
    data.ctrl[0] = GRIP_OPEN
    home_pos = data.mocap_pos[ids.mocap].copy()
    pen = step_hold(model, data, ids, home_pos, GRIP_OPEN, 100, hook)
    z0 = float(data.site_xpos[ids.obj_site][2])
    above = np.array([xy[0], xy[1], APPROACH_Z])
    grasp = np.array([xy[0], xy[1], OBJ_Z])
    lift = np.array([xy[0], xy[1], LIFT_Z])
    pen_m, step_m = move_toward(model, data, ids, above, MAX_MOCAP_STEP, GRIP_OPEN, hook)
    pen = max(pen, pen_m)
    largest = step_m
    pen = max(pen, step_hold(model, data, ids, above, GRIP_OPEN, 150, hook))
    pen_m, step_m = move_toward(model, data, ids, grasp, 0.01, GRIP_OPEN, hook)
    pen = max(pen, pen_m)
    largest = max(largest, step_m)
    pen = max(pen, step_hold(model, data, ids, grasp, GRIP_OPEN, 150, hook))
    n_close = int(round(0.8 / float(model.opt.timestep)))
    for i in range(n_close):
        ctrl = GRIP_OPEN + (i + 1) / n_close * (GRIP_CLOSE - GRIP_OPEN)
        pen = max(pen, step_hold(model, data, ids, grasp, ctrl, 1, hook))
    pen = max(pen, step_hold(model, data, ids, grasp, GRIP_CLOSE, int(round(0.4 / model.opt.timestep)), hook))
    pen_m, step_m = move_toward(model, data, ids, lift, 0.02, GRIP_CLOSE, hook)
    pen = max(pen, pen_m)
    largest = max(largest, step_m)
    pen = max(pen, step_hold(model, data, ids, lift, GRIP_CLOSE, 100, hook))
    rel0 = data.site_xpos[ids.obj_site].copy() - data.site_xpos[ids.site]
    zs = []
    slips = []
    n_hold = int(round(2.0 / float(model.opt.timestep)))
    for _ in range(n_hold):
        pen = max(pen, step_hold(model, data, ids, lift, GRIP_CLOSE, 1, hook))
        zs.append(float(data.site_xpos[ids.obj_site][2]))
        rel = data.site_xpos[ids.obj_site] - data.site_xpos[ids.site]
        slips.append(float(np.linalg.norm(rel - rel0)))
    zs_arr = np.asarray(zs, dtype=float)
    z_lift = float(zs_arr[0])
    drop = float(z_lift - zs_arr.min())
    depth, pair = worst_contact(model, data)
    return {
        "xy": xy,
        "z0": z0,
        "z_lift": z_lift,
        "z_end": float(zs_arr[-1]),
        "rise": float(zs_arr[-1] - z0),
        "drop": drop,
        "slip": float(max(slips)),
        "pen": pen,
        "pen_pair": pair,
        "finger": float(data.qpos[6]),
        "finger_r": float(data.qpos[7]),
        "gap_mm": float(abs(data.qpos[7] - data.qpos[6]) * 1000.0),
        "finite": finite_state(data) and bool(np.isfinite(zs_arr).all()),
        "max_step": largest,
        "pinch": data.site_xpos[ids.site].copy(),
    }


def grasp_passes(row: dict) -> bool:
    return bool(
        row["finite"]
        and row["rise"] >= LIFT_MIN_M
        and row["drop"] < DROP_LIMIT_M
        and row["pen"] <= PEN_LIMIT_M
        and row["max_step"] <= MAX_MOCAP_STEP + 1e-9
    )


def world_to_body(data, body_id: int, pts: np.ndarray) -> np.ndarray:
    rot = data.xmat[body_id].reshape(3, 3)
    origin = data.xpos[body_id]
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return (pts - origin) @ rot


def mesh_world(model, data, geom_id: int) -> np.ndarray:
    import mujoco

    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return np.asarray(data.geom_xpos[geom_id], dtype=float).reshape(1, 3)
    mesh_id = int(model.geom_dataid[geom_id])
    adr = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    verts = np.asarray(model.mesh_vert[adr : adr + count], dtype=float)
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    return data.geom_xpos[geom_id] + verts @ rot.T


def finger_metrics(model, data, ids: Ids) -> dict:
    import mujoco

    pads = {}
    for name in (
        "left/left_g0",
        "left/left_g1",
        "left/left_g2",
        "left/right_g0",
        "left/right_g1",
        "left/right_g2",
    ):
        gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        local = world_to_body(data, ids.gripper_base, data.geom_xpos[gid]).reshape(3)
        pads[name] = (local, float(model.geom_size[gid, 0]))
    left = np.stack([pads[n][0] for n in pads if "/left_g" in n])
    right = np.stack([pads[n][0] for n in pads if "/right_g" in n])
    radius = pads["left/left_g0"][1]
    pad_gap = float((left[:, 0].min() - radius) - (right[:, 0].max() + radius))
    left_x = []
    right_x = []
    tip_z = -1e9
    low_z = 1e9
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if bname not in ("left/left_finger_link", "left/right_finger_link"):
            continue
        if int(model.geom_contype[gid]) == 0:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        local = world_to_body(data, ids.gripper_base, mesh_world(model, data, gid))
        tip_z = max(tip_z, float(local[:, 2].max()))
        low_z = min(low_z, float(data.geom_xpos[gid][2]))
        world = mesh_world(model, data, gid)
        low_z = min(low_z, float(world[:, 2].min()))
        if bname.endswith("left_finger_link"):
            left_x.append(float(local[:, 0].min()))
        else:
            right_x.append(float(local[:, 0].max()))
    pinch_local_z = float(model.body_pos[ids.pinch][2])
    tip_below = tip_z - pinch_local_z
    return {
        "pad_gap": pad_gap,
        "mesh_gap": float(min(left_x) - max(right_x)),
        "pad_centroid": np.concatenate([left, right]).mean(axis=0),
        "tip_below": float(tip_below),
        "finger_low_z": float(low_z),
    }


def fmt(values, digits=6) -> str:
    return " ".join(f"{float(v):.{digits}g}" for v in np.asarray(values, dtype=float).reshape(-1))


def cmd_info(args) -> int:
    import mujoco

    np.random.seed(args.seed)
    model, data = load_model()
    ids = Ids(model)
    text = MODEL_PATH.read_text(encoding="utf-8")
    emit("== model ==")
    emit(f"path {MODEL_PATH}")
    emit(
        f"nq {model.nq}  nv {model.nv}  nu {model.nu}  neq {model.neq}  "
        f"timestep {model.opt.timestep}  integrator {int(model.opt.integrator)}  "
        f"cone {int(model.opt.cone)}  impratio {model.opt.impratio}  "
        f"noslip_iterations {int(model.opt.noslip_iterations)}"
    )
    emit(f"integrator name implicitfast={int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)} "
         f"cone elliptic={int(mujoco.mjtCone.mjCONE_ELLIPTIC)}")
    emit("== actuators ==")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
        emit(f"  {i} {name} joint {joint} ctrlrange {fmt(model.actuator_ctrlrange[i])}")
    emit("== equality ==")
    for i in range(model.neq):
        kind = int(model.eq_type[i])
        if kind == int(mujoco.mjtEq.mjEQ_JOINT):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj1id[i]))
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[i]))
            label = "joint"
        elif kind == int(mujoco.mjtEq.mjEQ_WELD):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.eq_obj1id[i]))
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.eq_obj2id[i]))
            label = "weld"
        else:
            n1 = n2 = "?"
            label = str(kind)
        emit(
            f"  {i} {label} {n1} {n2} solref {fmt(model.eq_solref[i])} "
            f"data {fmt(model.eq_data[i], 4)}"
        )
    emit("== gravcomp ==")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"
        emit(f"  {name} gravcomp {model.body_gravcomp[i]:.1f} mass {model.body_mass[i]:.6f}")
    emit("== constants ==")
    emit(f"PINCH_POS {fmt(PINCH_POS, 8)}")
    emit(f"body_pos  {fmt(model.body_pos[ids.pinch], 8)}")
    emit(f"HOME_QPOS {fmt(HOME_QPOS, 10)}")
    emit(f"GRASP_QUAT {fmt(GRASP_QUAT, 16)}")
    key_q = np.asarray(model.key_qpos[ids.key], dtype=float)
    emit(f"keyframe qpos {fmt(key_q, 10)}")
    emit(f"keyframe ctrl {fmt(model.key_ctrl[ids.key])}")
    emit(f"keyframe mpos {fmt(model.key_mpos[ids.key])}")
    emit(f"keyframe mquat {fmt(model.key_mquat[ids.key], 16)}")
    reset_home(model, data, ids)
    pos_mm, rot_deg, vals = weld_errors(model, data, ids)
    manual_mm = float(np.linalg.norm(data.xpos[ids.pinch] - data.mocap_pos[ids.mocap]) * 1000.0)
    manual_deg = angle_deg(data.xquat[ids.pinch], data.mocap_quat[ids.mocap])
    emit("== weld residual before first step ==")
    emit(f"efc_mm {pos_mm:.6f}  efc_deg {rot_deg:.6f}  efc {fmt(vals, 8)}")
    emit(f"manual_mm {manual_mm:.6f}  manual_deg {manual_deg:.6f}")
    emit(f"pinch_world {fmt(data.site_xpos[ids.site], 6)}")
    rot = data.xmat[ids.pinch].reshape(3, 3)
    emit(f"pinch_x {fmt(rot[:, 0], 4)}")
    emit(f"pinch_y {fmt(rot[:, 1], 4)}")
    emit(f"pinch_z {fmt(rot[:, 2], 4)}")
    metrics = finger_metrics(model, data, ids)
    emit("== gripper opening at home (fingers 0.037) ==")
    emit(f"pad_inner_gap_mm {metrics['pad_gap'] * 1000:.2f}")
    emit(f"mesh_inner_gap_mm {metrics['mesh_gap'] * 1000:.2f}")
    emit(f"pad_centroid_in_gripper_base {fmt(metrics['pad_centroid'], 6)}")
    emit(f"tip_below_pinch_mm {metrics['tip_below'] * 1000:.2f}")
    emit(f"finger_lowest_world_z {metrics['finger_low_z']:.4f}")
    emit(
        "lowest pinch z with 1 mm ground clearance "
        f"{metrics['tip_below'] + 0.001:.4f}"
    )
    down = float(np.dot(rot[:, 2], np.array([0.0, 0.0, -1.0])))
    opening = float(np.dot(rot[:, 0], np.array([0.0, -1.0, 0.0])))
    joint_ranges = model.jnt_range[:6]
    margins = np.minimum(HOME_QPOS[:6] - joint_ranges[:, 0], joint_ranges[:, 1] - HOME_QPOS[:6])
    emit(f"joint_margin_rad {fmt(margins, 4)} min {float(margins.min()):.4f}")
    relpose_ok = 'relpose="0 0 0 1 0 0 0"' in text
    n_joint_eq = sum(int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT) for i in range(model.neq))
    n_weld = sum(int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_WELD) for i in range(model.neq))
    key_match = float(np.max(np.abs(key_q[:8] - HOME_QPOS))) < 1e-8
    quat_match = float(np.max(np.abs(model.key_mquat[ids.key] - GRASP_QUAT))) < 1e-8
    pinch_match = float(np.max(np.abs(model.body_pos[ids.pinch] - PINCH_POS))) < 1e-8
    checks = {
        "nu==1": model.nu == 1,
        "neq==2": model.neq == 2,
        "one joint equality": n_joint_eq == 1,
        "one weld": n_weld == 1,
        "explicit relpose": relpose_ok,
        "weld pos < 1 mm": pos_mm < WELD_LIMIT_MM and manual_mm < WELD_LIMIT_MM,
        "home qpos matches keyframe": key_match,
        "grasp quat matches keyframe": quat_match,
        "pinch pos matches body": pinch_match,
        "gripper down": down > np.cos(np.radians(1.0)),
        "opening along world y": opening > np.cos(np.radians(1.0)),
        "opening fits 40 mm": metrics["pad_gap"] > 0.04 and metrics["mesh_gap"] > 0.04,
        "home joint margin > 0.1": float(margins.min()) > 0.1,
        "finite": finite_state(data),
    }
    ok = True
    emit("== info checks ==")
    for name, passed in checks.items():
        emit(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    emit(f"RESULT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_view(args) -> int:
    import mujoco
    import mujoco.viewer

    np.random.seed(args.seed)
    model, data = load_model()
    ids = Ids(model)
    reset_home(model, data, ids)
    emit("passive viewer. close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    return 0


def grid_points(low, high, nx, ny, nz):
    xs = np.linspace(low[0], high[0], nx)
    ys = np.linspace(low[1], high[1], ny)
    zs = np.linspace(low[2], high[2], nz)
    pts = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    order = []
    pending = set(range(len(pts)))
    cursor = np.array([0.30, 0.0, 0.20])
    while pending:
        nxt = min(pending, key=lambda k: float(np.linalg.norm(pts[k] - cursor)))
        order.append(nxt)
        pending.remove(nxt)
        cursor = pts[nxt]
    return pts, order


def cmd_track(args) -> int:
    import mujoco

    np.random.seed(args.seed)
    model, data = load_model()
    ids = Ids(model)
    low = np.asarray(args.low, dtype=float)
    high = np.asarray(args.high, dtype=float)
    pts, order = grid_points(low, high, args.nx, args.ny, args.nz)
    reset_home(model, data, ids)
    set_object(model, data, ids, (-1.0, -1.0, OBJ_Z))
    hold_pos = data.mocap_pos[ids.mocap].copy()
    step_hold(model, data, ids, hold_pos, GRIP_OPEN, 200)
    n_hold = int(round(args.hold / float(model.opt.timestep)))
    n_window = int(round(args.window / float(model.opt.timestep)))
    rows = []
    pen = 0.0
    pen_pair = ""
    exploded = False
    for index in order:
        target = pts[index]
        data.mocap_pos[ids.mocap] = target
        data.mocap_quat[ids.mocap] = GRASP_QUAT
        window_p = []
        window_r = []
        for step in range(n_hold):
            mujoco.mj_step(model, data)
            depth, pair = worst_contact(model, data)
            if depth > pen:
                pen = depth
                pen_pair = pair
            if not finite_state(data) or float(np.linalg.norm(data.site_xpos[ids.site])) > 5.0:
                exploded = True
                break
            if step >= n_hold - n_window:
                window_p.append(float(np.linalg.norm(data.site_xpos[ids.site] - target) * 1000.0))
                window_r.append(angle_deg(site_quat(model, data, ids.site), GRASP_QUAT))
        if exploded or not window_p:
            rows.append((target, 1e9, 1e9, 1e9, 1e9, False))
            break
        pos_max = max(window_p)
        rot_max = max(window_r)
        pos_span = max(window_p) - min(window_p)
        rot_span = max(window_r) - min(window_r)
        # A rising error inside the window means the point has not settled.
        diverged = window_p[-1] > window_p[0] + 1.0 or window_r[-1] > window_r[0] + 0.5
        passed = (
            pos_max < POS_LIMIT_MM
            and rot_max < ROT_LIMIT_DEG
            and pos_span <= FLUT_POS_MM
            and rot_span <= FLUT_ROT_DEG
            and not diverged
        )
        rows.append((target, pos_max, rot_max, pos_span, rot_span, passed))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RUN_DIR / "track.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["x", "y", "z", "pos_mm_max", "rot_deg_max", "pos_fluct_mm", "rot_fluct_deg", "pass"]
        )
        for target, pos_max, rot_max, pos_span, rot_span, passed in rows:
            writer.writerow(
                [
                    f"{target[0]:.6f}",
                    f"{target[1]:.6f}",
                    f"{target[2]:.6f}",
                    f"{pos_max:.6f}",
                    f"{rot_max:.6f}",
                    f"{pos_span:.6f}",
                    f"{rot_span:.6f}",
                    int(passed),
                ]
            )
    pos = np.array([row[1] for row in rows], dtype=float)
    rot = np.array([row[2] for row in rows], dtype=float)
    bad = [row for row in rows if not row[5]]
    emit(
        f"track points {len(rows)}/{len(pts)}  low {fmt(low, 3)}  high {fmt(high, 3)}  "
        f"hold {args.hold}s  window {args.window}s"
    )
    emit(
        f"pos_mm max {pos.max():.3f}  p95 {np.percentile(pos, 95):.3f}  "
        f"rot_deg max {rot.max():.3f}  p95 {np.percentile(rot, 95):.3f}"
    )
    emit(
        f"fluct_pos_mm max {max(row[3] for row in rows):.3f}  "
        f"fluct_rot_deg max {max(row[4] for row in rows):.3f}"
    )
    emit(f"penetration_mm {pen * 1000:.3f}  pair {pen_pair or '-'}  exploded {exploded}")
    emit(f"csv {csv_path}")
    emit("x y z pos_mm rot_deg flut_pos flut_rot")
    for target, pos_max, rot_max, pos_span, rot_span, passed in rows:
        mark = "OK" if passed else "FAIL"
        emit(
            f"  {target[0]:7.3f} {target[1]:7.3f} {target[2]:7.3f}  "
            f"{pos_max:7.3f} {rot_max:7.3f} {pos_span:7.3f} {rot_span:7.3f}  {mark}"
        )
    if bad:
        emit("failed points:")
        for target, pos_max, rot_max, pos_span, rot_span, _passed in bad:
            emit(
                f"  {fmt(target, 3)} pos {pos_max:.3f} rot {rot_max:.3f} "
                f"flut {pos_span:.3f} {rot_span:.3f}"
            )
    ok = (not bad) and (not exploded) and len(rows) == len(pts) and pen <= PEN_LIMIT_M and finite_state(data)
    emit(f"RESULT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _emit_grasp_row(row: dict) -> None:
    flag = "OK" if grasp_passes(row) else "FAIL"
    emit(
        f"  {row['xy'][0]:.2f} {row['xy'][1]:.2f}  z0 {row['z0']:.4f}  "
        f"z_lift {row['z_lift']:.4f}  z_end {row['z_end']:.4f}  rise {row['rise']:.4f}  "
        f"drop_cm {row['drop'] * 100:.3f}  slip_mm {row['slip'] * 1000:.2f}  "
        f"pen_mm {row['pen'] * 1000:.2f}  finger {row['finger']:.4f}  "
        f"finger_r {row['finger_r']:.4f}  gap_mm {row['gap_mm']:.2f}  "
        f"max_step {row['max_step']:.4f}  {flag}"
    )


def cmd_grasp(args) -> int:
    np.random.seed(args.seed)
    model, data = load_model()
    ids = Ids(model)
    emit("xy  z0 z_lift z_end rise drop_cm slip_mm pen_mm finger finger_r gap_mm max_step")
    ok = True
    for xy in GRASP_XYS:
        row = run_grasp(model, data, ids, xy)
        _emit_grasp_row(row)
        ok = ok and grasp_passes(row)
    emit(f"RESULT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_video(args) -> int:
    import mujoco

    np.random.seed(args.seed)
    model, data = load_model()
    ids = Ids(model)
    positions = list(GRASP_XYS) if args.all else [GRASP_XYS[0]]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    names = []
    for index, xy in enumerate(positions):
        if len(positions) == 1:
            path = RUN_DIR / "grasp_center.mp4"
        else:
            path = RUN_DIR / f"grasp_{index}_{xy[0]:.2f}_{xy[1]:.2f}.mp4"
        frames: list[np.ndarray] = []
        renderer = mujoco.Renderer(model, 240, 320)
        counter = {"n": 0}

        def hook(renderer=renderer, frames=frames, counter=counter):
            if counter["n"] % 20 == 0:
                renderer.update_scene(data, camera="eval_cam")
                frame = np.array(renderer.render(), copy=True)
                frames.append(frame)
            counter["n"] += 1

        try:
            row = run_grasp(model, data, ids, xy, hook=hook)
        finally:
            renderer.close()
        _emit_grasp_row(row)
        if not frames:
            emit(f"no frames for {xy}")
            ok = False
            continue
        if frames[0].shape != (240, 320, 3):
            emit(f"bad frame shape {frames[0].shape}")
            ok = False
        import imageio.v2 as imageio

        imageio.mimsave(
            path,
            frames,
            fps=25,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
        )
        size = path.stat().st_size
        emit(f"video {path} frames {len(frames)} bytes {size}")
        names.append(path)
        ok = ok and grasp_passes(row) and size > 0
    emit(f"RESULT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1 Aloha pick-and-place model checks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("info", help="print model summary and the home weld residual")
    info.add_argument("--seed", type=int, default=0)
    info.set_defaults(func=cmd_info)

    view = sub.add_parser("view", help="open a passive MuJoCo viewer")
    view.add_argument("--seed", type=int, default=0)
    view.set_defaults(func=cmd_view)

    track = sub.add_parser("track", help="mocap grid tracking errors")
    track.add_argument("--seed", type=int, default=0)
    track.add_argument("--nx", type=int, default=5)
    track.add_argument("--ny", type=int, default=5)
    track.add_argument("--nz", type=int, default=4)
    track.add_argument("--low", type=float, nargs=3, default=WS_LOW.tolist())
    track.add_argument("--high", type=float, nargs=3, default=WS_HIGH.tolist())
    track.add_argument("--hold", type=float, default=1.0)
    track.add_argument("--window", type=float, default=0.2)
    track.set_defaults(func=cmd_track)

    grasp = sub.add_parser("grasp", help="pick-and-hold at the five default xy positions")
    grasp.add_argument("--seed", type=int, default=0)
    grasp.set_defaults(func=cmd_grasp)

    video = sub.add_parser("video", help="record grasp from eval_cam")
    video.add_argument("--seed", type=int, default=0)
    video.add_argument("--all", action="store_true", help="record all five grasp positions")
    video.set_defaults(func=cmd_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
