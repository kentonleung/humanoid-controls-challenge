import numpy as np
import mujoco

class TrajectoryGenerator:
    """
    Figure-eight trajectory with coupled SE(3) target frames.
    Centre: (0.45, 0.0, 0.5) in Panda base frame.
    Radius: 0.15 m (conservative — well within reachable workspace).
    """
    def __init__(self, r=0.15, omega=2*np.pi/12.5, z0=0.5):
        self.r     = r
        self.omega = omega
        self.z0    = z0
        self.cx    = 0.45
        self.cy    = 0.0

    def get_target(self, t: float) -> np.ndarray:
        """Returns 3D target position at time t."""
        x = self.cx + self.r       * np.cos(self.omega * t)
        y = self.cy + (self.r / 2) * np.sin(2 * self.omega * t)
        return np.array([x, y, self.z0], dtype=np.float32)

    def get_target_orientation(self, t: float) -> np.ndarray:
        """
        Returns target quaternion [w, x, y, z] at time t.

        Frame convention:
          - z-axis: always world -Z (tool pointing down)
          - x-axis: tangent to the figure-eight velocity direction
          - y-axis: cross product z x_tangent (right-hand rule)

        The tangent is the time derivative of get_target(t):
          dx/dt = -r * omega * sin(omega * t)
          dy/dt =  r * omega * cos(2 * omega * t)   [chain rule on sin(2wt)]
        """
        dx = -self.r       * self.omega * np.sin(self.omega * t)
        dy =  self.r       * self.omega * np.cos(2 * self.omega * t)
        tangent = np.array([dx, dy, 0.0])
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            tangent = np.array([1.0, 0.0, 0.0])  # fallback at degenerate points
        else:
            tangent /= norm

        z_axis = np.array([0.0, 0.0, -1.0])      # tool-down
        y_axis = np.cross(z_axis, tangent)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0])
        else:
            y_axis /= y_norm
        x_axis = np.cross(y_axis, z_axis)

        # Build rotation matrix (columns = x, y, z axes of target frame)
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion [w, x, y, z]
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, R.flatten().astype(np.float64))
        return quat.astype(np.float32)
