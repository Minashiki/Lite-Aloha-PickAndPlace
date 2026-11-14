"""确定性抓取-放置状态机。

只读取观测字典，不接触仿真对象。动作是
``a[:3] = clip(k * (目标 - 当前) / action_scale, -1, 1)``，夹爪张开 +1、闭合 -1。
"""

from __future__ import annotations

import numpy as np

PHASES = ("APPROACH", "DESCEND", "GRASP", "LIFT", "MOVE", "HOLD")

_GRIP = slice(0, 3)
_OBJ = slice(3, 6)
_FINGERS = slice(9, 11)


class ScriptedPickPlacePolicy:
    """按 APPROACH → DESCEND → GRASP → LIFT → MOVE → HOLD 搬运物块。

    阈值默认值来自 ``tasks/M3_task.md``：接近水平 0.5 cm、接近高度 1 cm、
    下降竖直 0.5 cm、抬升到位容差 1 cm、放置距离 1 cm、掉落 2 cm、空抓
    两指 qpos < 0.015。``action_scale`` 必须与动作定义的 0.05 m 一致。
    """

    def __init__(
        self,
        gain: float = 1.0,
        action_scale: float = 0.05,
        approach_height: float = 0.08,
        approach_xy_tol: float = 0.005,
        approach_z_tol: float = 0.01,
        approach_settle: int = 1,
        descend_z_tol: float = 0.005,
        descend_z_gain: float = 1.0,
        grasp_stable_delta: float = 1e-4,
        grasp_min_steps: int = 3,
        grasp_max_steps: int = 12,
        empty_qpos: float = 0.015,
        open_qpos: float = 0.030,
        lift_margin: float = 0.05,
        lift_z_tol: float = 0.01,
        move_goal_tol: float = 0.01,
        move_xy_tol: float = 0.005,
        move_z_tol: float = 0.01,
        place_z_gain: float = 1.0,
        drop_tol: float = 0.02,
        drop_patience: int = 2,
        press_tol: float = 0.01,
        approach_timeout: int = 20,
        descend_timeout: int = 12,
        lift_timeout: int = 15,
        move_timeout: int = 25,
        hold_timeout: int = 50,
    ) -> None:
        self.gain = float(gain)
        self.action_scale = float(action_scale)
        self.approach_height = float(approach_height)
        self.approach_xy_tol = float(approach_xy_tol)
        self.approach_z_tol = float(approach_z_tol)
        self.approach_settle = int(approach_settle)
        self.descend_z_tol = float(descend_z_tol)
        self.descend_z_gain = float(descend_z_gain)
        self.grasp_stable_delta = float(grasp_stable_delta)
        self.grasp_min_steps = int(grasp_min_steps)
        self.grasp_max_steps = int(grasp_max_steps)
        self.empty_qpos = float(empty_qpos)
        self.open_qpos = float(open_qpos)
        self.lift_margin = float(lift_margin)
        self.lift_z_tol = float(lift_z_tol)
        self.move_goal_tol = float(move_goal_tol)
        self.move_xy_tol = float(move_xy_tol)
        self.move_z_tol = float(move_z_tol)
        self.place_z_gain = float(place_z_gain)
        self.drop_tol = float(drop_tol)
        self.drop_patience = int(drop_patience)
        self.press_tol = float(press_tol)
        self.approach_timeout = int(approach_timeout)
        self.descend_timeout = int(descend_timeout)
        self.lift_timeout = int(lift_timeout)
        self.move_timeout = int(move_timeout)
        self.hold_timeout = int(hold_timeout)
        self._timeout_of = {
            "APPROACH": self.approach_timeout,
            "DESCEND": self.descend_timeout,
            "GRASP": self.grasp_max_steps,
            "LIFT": self.lift_timeout,
            "MOVE": self.move_timeout,
            "HOLD": self.hold_timeout,
        }
        self.reset()

    def reset(self) -> None:
        """清空阶段、计步和本回合的统计。"""
        self._phase = "APPROACH"
        self._phase_steps = 0
        self._step = 0
        self._prev_fingers: np.ndarray | None = None
        self._offset: np.ndarray | None = None
        self._z_lift: float | None = None
        self._lift_xy: np.ndarray | None = None
        self._move_stage = "above"
        self._descend_z0: float | None = None
        self._grasp_hold: np.ndarray | None = None
        self._approach_hits = 0
        self._drop_streak = 0

        self.first_entry: dict[str, int | None] = {name: None for name in PHASES}
        self.first_entry["APPROACH"] = 0
        self.transitions: list[tuple[int, str]] = [(0, "APPROACH")]
        self.events: list[str] = []
        self.n_drop = 0
        self.n_empty = 0
        self.n_timeout = 0
        self.n_press = 0
        self.n_grasp = 0
        self.timeout_by_phase = {name: 0 for name in PHASES}
        self.descended_unaligned = False
        self.max_descend_xy = 0.0
        self.last_grasp_fingers: np.ndarray | None = None
        self.trace: list[tuple] = []

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def label(self) -> str:
        """视频标注用。MOVE 会带上“先到上方 / 再下降”的子阶段。"""
        if self._phase == "MOVE" and self._move_stage == "down":
            return "MOVE_DOWN"
        if self._phase == "MOVE":
            return "MOVE_ABOVE"
        return self._phase

    def __call__(self, obs) -> np.ndarray:
        observation = np.asarray(obs["observation"], dtype=np.float64)
        grip = observation[_GRIP].copy()
        obj = observation[_OBJ].copy()
        fingers = observation[_FINGERS].copy()
        goal = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(3).copy()

        if self._phase == "DESCEND":
            xy = float(np.linalg.norm(grip[:2] - obj[:2]))
            if xy > self.max_descend_xy:
                self.max_descend_xy = xy

        handled = self._safety(grip, obj)
        if not handled:
            handled = self._timeout(grip, obj, fingers, goal)
        if not handled:
            self._transitions(grip, obj, fingers, goal)

        action = self._command(grip, obj, goal)
        self.trace.append(
            (
                int(self._step),
                self._phase,
                float(grip[0]),
                float(grip[1]),
                float(grip[2]),
                float(obj[0]),
                float(obj[1]),
                float(obj[2]),
                float(fingers[0]),
                float(fingers[1]),
                float(action[0]),
                float(action[1]),
                float(action[2]),
                float(action[3]),
            )
        )
        self._prev_fingers = fingers
        self._phase_steps += 1
        self._step += 1
        return action

    def _event(self, name: str) -> None:
        self.events.append(f"{name}@{int(self._step)}")

    def _enter(self, phase: str) -> None:
        if phase == "APPROACH":
            self._offset = None
            self._z_lift = None
            self._lift_xy = None
            self._move_stage = "above"
            self._approach_hits = 0
            self._drop_streak = 0
        elif phase == "MOVE":
            self._move_stage = "above"
        elif phase == "GRASP":
            self.n_grasp += 1
        self._phase = phase
        self._phase_steps = 0
        if self.first_entry[phase] is None:
            self.first_entry[phase] = int(self._step)
        self.transitions.append((int(self._step), phase))

    def _holding(self, obj: np.ndarray, grip: np.ndarray) -> bool:
        if self._offset is None:
            return False
        return float(np.linalg.norm(obj - grip - self._offset)) <= self.drop_tol

    def _safety(self, grip: np.ndarray, obj: np.ndarray) -> bool:
        if self._phase in ("LIFT", "MOVE", "HOLD") and self._offset is not None:
            error = float(np.linalg.norm(obj - grip - self._offset))
            if error > self.drop_tol:
                self._drop_streak += 1
            else:
                self._drop_streak = 0
            if self._drop_streak >= self.drop_patience:
                self.n_drop += 1
                self._event("drop")
                self._enter("APPROACH")
                return True
        if (
            self._phase == "DESCEND"
            and self._descend_z0 is not None
            and float(obj[2]) < self._descend_z0 - self.press_tol
        ):
            self.n_press += 1
            self._event("press")
            self._enter("APPROACH")
            return True
        return False

    def _timeout(
        self,
        grip: np.ndarray,
        obj: np.ndarray,
        fingers: np.ndarray,
        goal: np.ndarray,
    ) -> bool:
        limit = self._timeout_of[self._phase]
        if self._phase_steps < limit:
            return False
        self.n_timeout += 1
        self.timeout_by_phase[self._phase] += 1
        self._event(f"timeout_{self._phase}")
        phase = self._phase
        if phase == "APPROACH":
            # 没对准就继续接近，不允许转入下降。
            self._phase_steps = 0
            return False
        if phase == "DESCEND":
            self._enter("APPROACH")
            return True
        if phase == "GRASP":
            self._finish_grasp(fingers, grip, obj, goal)
            return True
        if phase == "LIFT":
            if self._holding(obj, grip):
                self._enter("MOVE")
            else:
                self._enter("APPROACH")
            return True
        if phase == "MOVE":
            if not self._holding(obj, grip):
                self._enter("APPROACH")
            elif self._move_stage == "above":
                self._move_stage = "down"
                self._phase_steps = 0
            else:
                self._enter("HOLD")
            return True
        self._phase_steps = 0
        return False

    def _transitions(
        self,
        grip: np.ndarray,
        obj: np.ndarray,
        fingers: np.ndarray,
        goal: np.ndarray,
    ) -> None:
        phase = self._phase
        if phase == "APPROACH":
            target = self._approach_target(obj)
            xy = float(np.linalg.norm(grip[:2] - target[:2]))
            height = abs(float(grip[2] - target[2]))
            opened = float(np.min(fingers)) > self.open_qpos
            if xy < self.approach_xy_tol and height < self.approach_z_tol and opened:
                self._approach_hits += 1
            else:
                self._approach_hits = 0
            if self._approach_hits >= self.approach_settle:
                if xy >= self.approach_xy_tol:
                    self.descended_unaligned = True
                self._descend_z0 = float(obj[2])
                self._enter("DESCEND")
            return
        if phase == "DESCEND":
            xy = float(np.linalg.norm(grip[:2] - obj[:2]))
            if abs(float(grip[2] - obj[2])) < self.descend_z_tol:
                if xy >= self.approach_xy_tol:
                    self._enter("APPROACH")
                else:
                    self._grasp_hold = grip.copy()
                    self._enter("GRASP")
            return
        if phase == "GRASP":
            stable = (
                self._prev_fingers is not None
                and float(np.max(np.abs(fingers - self._prev_fingers))) < self.grasp_stable_delta
            )
            left_open_stop = float(np.min(fingers)) < self.open_qpos
            if self._phase_steps >= self.grasp_min_steps and stable and left_open_stop:
                self._finish_grasp(fingers, grip, obj, goal)
            return
        if phase == "LIFT":
            if self._z_lift is not None and float(obj[2]) >= self._z_lift - self.lift_z_tol:
                self._enter("MOVE")
            return
        if phase == "MOVE":
            if float(np.linalg.norm(obj - goal)) < self.move_goal_tol:
                self._enter("HOLD")
                return
            if self._move_stage == "above":
                waypoint = self._above_target(goal)
                xy = float(np.linalg.norm(grip[:2] - waypoint[:2]))
                height = abs(float(grip[2] - waypoint[2]))
                if xy < self.move_xy_tol and height < self.move_z_tol:
                    self._move_stage = "down"
                    self._phase_steps = 0
            return

    def _finish_grasp(
        self,
        fingers: np.ndarray,
        grip: np.ndarray,
        obj: np.ndarray,
        goal: np.ndarray,
    ) -> None:
        self.last_grasp_fingers = fingers.copy()
        if float(np.min(fingers)) > self.open_qpos:
            self._event("grasp_open")
            self._enter("APPROACH")
            return
        if bool(np.all(fingers < self.empty_qpos)):
            self.n_empty += 1
            self._event("empty")
            self._enter("APPROACH")
            return
        self._offset = obj - grip
        self._z_lift = float(max(float(goal[2]), float(obj[2])) + self.lift_margin)
        self._lift_xy = grip[:2].copy()
        self._drop_streak = 0
        self._enter("LIFT")

    def _approach_target(self, obj: np.ndarray) -> np.ndarray:
        target = obj.copy()
        target[2] = float(obj[2]) + self.approach_height
        return target

    def _above_target(self, goal: np.ndarray) -> np.ndarray:
        # 夹爪目标使物块保持在进入 LIFT 时算定的 z_lift，xy 对齐目标。
        target = goal - self._offset
        target[2] = float(self._z_lift) - float(self._offset[2])
        return target

    def _place_target(self, goal: np.ndarray) -> np.ndarray:
        return goal - self._offset

    def _lift_target(self) -> np.ndarray:
        target = np.empty(3, dtype=np.float64)
        target[:2] = self._lift_xy
        target[2] = float(self._z_lift) - float(self._offset[2])
        return target

    def _command(self, grip: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> np.ndarray:
        phase = self._phase
        if phase == "APPROACH":
            return self._move(grip, self._approach_target(obj), 1.0)
        if phase == "DESCEND":
            target = obj.copy()
            # 水平再次偏出 0.5 cm 时只修 xy，避免从上方压到物块。
            if float(np.linalg.norm(grip[:2] - obj[:2])) > self.approach_xy_tol:
                target[2] = float(grip[2])
            return self._move(grip, target, 1.0, z_gain=self.descend_z_gain)
        if phase == "GRASP":
            hold = self._grasp_hold if self._grasp_hold is not None else grip
            return self._move(grip, hold, -1.0)
        if phase == "LIFT":
            return self._move(grip, self._lift_target(), -1.0)
        if phase == "MOVE":
            if self._move_stage == "down":
                place = self._place_target(goal)
                if float(np.linalg.norm(grip[:2] - place[:2])) > self.move_xy_tol:
                    self._move_stage = "above"
                    return self._move(grip, self._above_target(goal), -1.0)
                return self._move(grip, place, -1.0, z_gain=self.place_z_gain)
            return self._move(grip, self._above_target(goal), -1.0)
        return self._move(grip, self._place_target(goal), -1.0, z_gain=self.place_z_gain)

    def _move(
        self,
        grip: np.ndarray,
        target: np.ndarray,
        gripper: float,
        z_gain: float | None = None,
    ) -> np.ndarray:
        gain_z = self.gain if z_gain is None else float(z_gain)
        delta = np.asarray(target, dtype=np.float64) - grip
        raw = np.empty(3, dtype=np.float64)
        raw[:2] = self.gain * delta[:2] / self.action_scale
        raw[2] = gain_z * delta[2] / self.action_scale
        action = np.empty(4, dtype=np.float32)
        action[:3] = np.clip(raw, -1.0, 1.0).astype(np.float32)
        action[3] = np.float32(gripper)
        return action
