#!/usr/bin/env python3

import rospy
from typing import Optional, List
import numpy as np
import pinocchio

from geometry_msgs.msg import Twist
from tbai_msgs.msg import RbdState

from scipy.spatial.transform import Rotation as R

from threading import Lock


class PinocchioInterface:
    def __init__(self, urdf) -> None:
        self._model, self._data = self._build_model(urdf)

    def _build_model(self, urdf: str) -> None:
        """Build model from urdf and connect base link to world using freeflyer joint."""
        world_joint = pinocchio.JointModelFreeFlyer()
        model = pinocchio.buildModelFromXML(urdf, world_joint)
        data = model.createData()
        return model, data

    def get_model(self) -> pinocchio.Model:
        return self._model

    def get_data(self) -> pinocchio.Data:
        return self._data

    def compute_forward_kinematics(self, q: np.ndarray, v: Optional[np.ndarray] = None) -> None:
        pinocchio.forwardKinematics(self._model, self._data, q, v)


class EndEffectorKinematics:
    def __init__(self, model: pinocchio.Model, data: pinocchio.Data, ee_names: List[str]):
        self._model = model
        self._data = data
        self.ee_names = ee_names
        self.ee_idxs = [model.getFrameId(name) for name in ee_names]

    def get_model(self) -> pinocchio.Model:
        return self._model

    def get_data(self) -> pinocchio.Data:
        return self._data

    def get_positions(self) -> List[np.ndarray]:
        return [self._data.oMf[idx].translation for idx in self.ee_idxs]

    def get_velocities(self) -> List[np.ndarray]:
        return [
            pinocchio.getFrameVelocity(
                self._model,
                self._data,
                idx,
                pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ).linear
            for idx in self.ee_idxs
        ]


class SlipDetector:
    def __init__(
        self,
        slip_threshold: float,
        pinocchio_interface: PinocchioInterface,
        end_effector_kinematics: EndEffectorKinematics,
    ) -> None:
        assert pinocchio_interface.get_model() is end_effector_kinematics.get_model(), "Model mismatch"
        assert pinocchio_interface.get_data() is end_effector_kinematics.get_data(), "Data mismatch"
        self.pinocchio_interface = pinocchio_interface
        self.end_effector_kinematics = end_effector_kinematics
        self.slip_threshold = slip_threshold

    def detect_slips(self, q: np.ndarray, v: np.ndarray, contacts: np.ndarray):
        self.pinocchio_interface.compute_forward_kinematics(q, v)
        slips = list()
        ee_velocities = self.end_effector_kinematics.get_velocities()
        for contact in contacts:
            slip = np.linalg.norm(ee_velocities[contact]) > self.slip_threshold
            slips.append(slip)
        return slips


class StateSubscriber:
    def __init__(self, state_topic: str) -> None:
        self._lock = Lock()
        self.state = None
        self.contact_flags = None
        self.state_sub = rospy.Subscriber(state_topic, RbdState, self.callback)
        self.last_yaw = 0.0

    def callback(self, msg: RbdState):
        with self._lock:
            self.state = msg.rbd_state
            self.contact_flags = msg.contact_flags

    def get_xyz_yaw(self):  # for AutonomousMode
        with self._lock:
            xyz = self.state[3:6]
            yaw = self._rbd2yaw(self.state)
            return xyz, yaw

    def get_pin_and_contacts(self):  # For slip detector
        with self._lock:
            pinstate = self._rbd2pinstate(self.state)
            return *pinstate, self.contact_flags

    def _rbd2yaw(self, rbd_state: np.ndarray) -> float:
        rotocs2 = self._ocs2xyz2mat(rbd_state[0:3])
        rpy = pinocchio.rpy.matrixToRpy(rotocs2)
        rpy[2] = self._anglemod(rpy[2], self.last_yaw)
        self.last_yaw = rpy[2]
        return self.last_yaw

    def _rbd2pinstate(self, rbd_state: np.ndarray) -> np.ndarray:
        base_position = rbd_state[3:6]  # xyz world position
        base_orientation = self._ocs2xyz2quat(rbd_state[0:3])
        joint_angles = rbd_state[12:24]
        q = np.concatenate([base_position, base_orientation, joint_angles])

        # FreeFlyer likes local velocities :D
        base_linear_velocity = rbd_state[9:12]
        base_angular_velocity = rbd_state[6:9]
        joint_velocities = rbd_state[24:36]
        v = np.concatenate([base_linear_velocity, base_angular_velocity, joint_velocities])

        return q, v

    def _rx(self, theta):
        return R.from_euler("x", theta, degrees=False).as_matrix()

    def _ry(self, theta):
        return R.from_euler("y", theta, degrees=False).as_matrix()

    def _rz(self, theta):
        return R.from_euler("z", theta, degrees=False).as_matrix()

    def _anglemod(self, angle, ref):
        upper_bound = ref + np.pi
        lower_bound = ref - np.pi
        if angle > upper_bound:
            angle = lower_bound + np.fmod(angle - lower_bound, 2 * np.pi)
        elif angle < lower_bound:
            angle = upper_bound - np.fmod(lower_bound - angle, 2 * np.pi)
        return angle

    def _ocs2xyz2quat(self, ocs2xyz: np.ndarray):
        return R.from_euler("xyz", ocs2xyz, degrees=False).as_quat()

    def _ocs2xyz2mat(self, ocs2xyz: np.ndarray):
        x, y, z = ocs2xyz[0], ocs2xyz[1], ocs2xyz[2]
        rotocs2 = self._rx(x) @ self._ry(y) @ self._rz(z)
        return rotocs2

    def _ocs2xyz2rpy(self, ocs2xyz: np.ndarray):
        rotocs2 = self._ocs2xyz2mat(ocs2xyz)
        rpy = pinocchio.rpy.matrixToRpy(rotocs2)
        rpy[2] = self._anglemod(rpy[2], self.last_yaw)
        self.last_yaw = rpy[2]
        return rpy


class TrackModel:
    class Waypoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

    def __init__(self, waypoints: List[Waypoint]):
        self.waypoints = waypoints
        self.total_length = self._compute_total_length()
        self.sublengths = self._compute_sublengths()

    def _compute_total_length(self):
        length = 0.0
        for i in range(len(self.waypoints)):
            ip = (i + 1) % len(self.waypoints)
            dx = self.waypoints[ip].x - self.waypoints[i].x
            dy = self.waypoints[ip].y - self.waypoints[i].y
            length += np.sqrt(dx * dx + dy * dy)
        return length

    def _compute_sublengths(self):
        sublengths = []
        length = 0.0
        for i in range(len(self.waypoints)):
            sublengths.append(length)
            ip = (i + 1) % len(self.waypoints)
            dx = self.waypoints[ip].x - self.waypoints[i].x
            dy = self.waypoints[ip].y - self.waypoints[i].y
            length += np.sqrt(dx * dx + dy * dy)
        sublengths.append(length)
        return sublengths

    def calculate_waypoint(self, phase: float):
        assert phase >= 0.0 and phase <= 1.0, "Phase out of bounds"
        distance_traveled = phase * self.total_length
        for i in range(len(self.waypoints)):
            if distance_traveled < self.sublengths[i + 1]:
                break
        wp_last = self.waypoints[i]
        wp_next = self.waypoints[(i + 1) % len(self.waypoints)]
        dx = wp_next.x - wp_last.x
        dy = wp_next.y - wp_last.y
        x = wp_last.x + (dx) * (distance_traveled - self.sublengths[i]) / (self.sublengths[i + 1] - self.sublengths[i])
        y = wp_last.y + (dy) * (distance_traveled - self.sublengths[i]) / (self.sublengths[i + 1] - self.sublengths[i])
        return self.Waypoint(x, y)

    @staticmethod
    def benchmark_track():
        # These values were copied from the benchmark's sdf file
        t = [
            "0.037268 0.557564",
            "-8.99859 0.429756",
            "-12.0103 1.39205",
            "-12.0559 2.95944",
            "-10.6984 2.95193",
            "-0.032725 2.93892",
            "1.64538 3.79553",
            "1.0071 4.95789",
            "-10.4925 5.02624",
            "-13.0383 6.07977",
            "-10.1211 7.49309",
            "-0.030184 7.41345",
        ]
        waypoints = [TrackModel.Waypoint(float(x), float(y)) for s in t for x, y in [s.split()]]
        track = TrackModel(waypoints)
        return track


class Statistician:
    def __init__(self, track_model: TrackModel, initial_phase: float = 0.0) -> None:
        self.track_model = track_model
        self.phase = initial_phase
        self.wp = self.track_model.calculate_waypoint(self.phase)
        self.t_start = rospy.Time.now()

    def update(self, phase: float):
        distance_traveled = phase * self.track_model.total_length
        for i in range(len(self.track_model.waypoints)):
            if distance_traveled < self.track_model.sublengths[(i + 1) % len(self.track_model.waypoints)]:
                break
        p = self.track_model.waypoints[i]
        if p == self.wp:
            return False
        t_now = rospy.Time.now()
        t_diff_s = (t_now - self.t_start).to_sec()
        print("Time taken to reach waypoint: ", t_diff_s)
        self.wp = p
        self.t_start = t_now

        return True


class TrackFollower:
    def __init__(
        self,
        track_model: TrackModel,
        phase_step: float = 0.03,
        tol: float = 1.0,
    ) -> None:
        self.track_model = track_model
        self.phase_step = phase_step
        self.tol = tol
        self.N = 100

    def run(self, xyz, yaw, phase: float):
        waypoint = self.track_model.calculate_waypoint(phase)
        current_x, current_y = xyz[0], xyz[1]
        desired_x, desired_y = waypoint.x, waypoint.y
        dx, dy = desired_x - current_x, desired_y - current_y
        current_yaw = yaw
        desired_yaw = np.arctan2(dy, dx)
        yaw_diff = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

        if np.abs(yaw_diff) >= np.pi / 3:
            lin_vel = 0.0
        else:
            lin_vel = 0.7

        twist = Twist()
        twist.linear.x = lin_vel
        twist.angular.z = np.clip(0.5 * (yaw_diff), -0.7, 0.7)
        return twist

    def update_phase(self, xyz: np.ndarray, current_phase: float) -> float:
        x, y = xyz[0], xyz[1]
        new_phase = min(current_phase + self.phase_step, 1.0)
        # Project phase back to make sure it's within reach of 1 meter
        # perform binary search
        lb, ub = current_phase, new_phase
        for i in range(self.N):
            mid = (lb + ub) / 2
            w = self.track_model.calculate_waypoint(mid)
            dx = w.x - x
            dy = w.y - y
            d = np.sqrt(dx * dx + dy * dy)
            if d >= self.tol:
                ub = mid
            else:
                lb = mid
        return lb
