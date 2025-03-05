import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class SpatialCurve:
    """
    定义空间曲线的类，用户可以重载 get_position 方法来定义自己的曲线。
    示例中我们实现一个简单的螺旋线：x = A*cos(omega*t), y = A*sin(omega*t), z = z0 + v*t
    """
    def __init__(self, A=1.0, omega=0.5, z0=1.0, v=0.1):
        self.A = A
        self.omega = omega
        self.z0 = z0
        self.v = v

    def get_position(self, t):
        """
        根据时间 t 返回曲线上的位置
        """
        x = self.A * np.cos(self.omega * t) - 1
        y = self.A * np.sin(self.omega * t)
        z = self.z0 + self.v * t
        return np.array([x, y, z])

class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
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
                 act: ActionType=ActionType.RPM,
                 spatial_curve: SpatialCurve = None  # spatial_curve
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
        
        # 如果没有给定曲线，则使用固定目标位置
        self.spatial_curve = spatial_curve
        if self.spatial_curve is None:
            self.TARGET_POS = np.array([1, 1, 1])
        else:
            # 初始目标点由曲线在t=0处的值决定
            self.TARGET_POS = self.spatial_curve.get_position(0)

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

    ################################################################################

    def _update_target(self):
        """
        根据当前仿真时间更新目标位置，如果提供了 spatial_curve。
        """
        if self.spatial_curve is not None:
            # 使用 step_counter 与 PyBullet 的频率来估计时间 t
            t = self.step_counter / self.PYB_FREQ
            self.TARGET_POS = self.spatial_curve.get_position(t)

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        self._update_target()  # update the target position
        state = self._getDroneStateVector(0)
        ret = max(0, 2 - np.linalg.norm(self.TARGET_POS-state[0:3])**4)
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
            return True
        else:
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
