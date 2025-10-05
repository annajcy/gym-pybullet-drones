# HoverAviaryQ3.py
import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviaryQ3(BaseRLAviary):
    """
    单机 RL 任务：到达并稳定于目标位置 (0, 0.5, 0.8)。

    改动要点：
    - 密奖励：距离二次项 + 速度/倾角/控制偏差轻惩罚（更稳、更有梯度）
    - 终止：距离阈值内连续命中 N 步才算“成功”，避免抖动
    - 截断：越界/过倾/超时
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        # 任务参数
        self.TARGET_POS = np.array([0.0, 0.5, 0.8], dtype=np.float32)
        self.EPISODE_LEN_SEC = 8
        self.SUCCESS_DIST = 0.10          # 成功距离阈值（m）
        self.SUCCESS_STREAK_N = 15        # 连续命中步数
        self._success_streak = 0

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         neighbourhood_radius=np.inf,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    # ---------------- Reward / Done / Trunc / Info ---------------- #

    def _computeReward(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        ret = max(0, 2 - dist**4)
        return ret

    def _computeTerminated(self):
        """
        连续命中 N 步判成功，防止“擦边进圈-出圈”抖动。
        """
        if getattr(self, "step_counter", 0) == 0:
            self._success_streak = 0

        pos = self._getDroneStateVector(0)[0:3]
        dist = np.linalg.norm(self.TARGET_POS - pos)
        if dist < self.SUCCESS_DIST:
            self._success_streak += 1
        else:
            self._success_streak = 0
        return self._success_streak >= self.SUCCESS_STREAK_N

    def _computeTruncated(self):
        """
        越界/过倾/超时截断。
        """
        s = self._getDroneStateVector(0)
        x, y, z = s[0], s[1], s[2]
        roll, pitch = s[7], s[8]

        too_far = (abs(x) > 1.5) or (abs(y) > 1.5) or (z > 2.0) or (z < 0.0)
        too_tilt = (abs(roll) > 0.40) or (abs(pitch) > 0.40)
        timeout = (self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC

        return bool(too_far or too_tilt or timeout)

    def _computeInfo(self):
        return {"target": self.TARGET_POS.copy(), "success_streak": int(self._success_streak)}
