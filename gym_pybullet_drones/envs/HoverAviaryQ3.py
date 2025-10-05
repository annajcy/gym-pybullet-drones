# HoverAviaryQ3.py
import dis
from os import error
import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviaryQ3(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.TARGET_POS = np.array([0, 0.5, 0.8])
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.last_action = np.zeros(4)
        self.action = np.zeros(4)
        self.is_first_step = True
        
    def _preprocessAction(self,
                          action
                          ):
        ret = super()._preprocessAction(action)
        self.action = action
        return ret

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        rot = s[7:9]
        vel = s[10:13]
   
        err = self.TARGET_POS - pos
        
        weight_err = np.array([
            err[0] * 1.15,  # X error
            err[1] * 1.05,        # Y error
            err[2] * 1.25   # Z error with 1.25x weight!
        ])
        
        weighted_dist = np.linalg.norm(weight_err)
        r = max(0, 2 - weighted_dist**4)
        velocity_penalty = np.linalg.norm(vel)
        r -= 0.5 * velocity_penalty
        r -= 0.1 * np.linalg.norm(err[2])
        orientation_penalty = np.linalg.norm(rot) # Penalize only roll and pitch
        r -= 0.2 * orientation_penalty
        if self.is_first_step:
            self.is_first_step = False
            self.last_action = self.action
        action_smoothness_penalty = np.linalg.norm(self.action - self.last_action)
        self.last_action = self.action
        r -= 0.1 * action_smoothness_penalty
        if np.linalg.norm(err) < 0.02:
            r += 0.5  # Bonus for being very close to the target
        return r

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # state = self._getDroneStateVector(0)
        # pos = state[0:3]
        # if np.linalg.norm(self.TARGET_POS-pos) < 0.01: #.0001:
        #     return True
        # else:
        return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
