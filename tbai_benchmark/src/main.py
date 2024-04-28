#!/usr/bin/env python3


import numpy as np
import rospy
from track import Track, Waypoint

from tbai_msgs.msg import RbdState
from geometry_msgs.msg import Twist

class Robot:
    def __init__(self, x, y, yaw, tol):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.track_phase = 0.0
        self.tol = tol

        state_topic = "/anymal_d/state"
        self.state_subscriber = rospy.Subscriber(state_topic, RbdState, self.state_callback)

    def state_callback(self, msg):
        self.x = msg.rbd_state[3+0]
        self.y = msg.rbd_state[3+1]
        self.yaw = msg.rbd_state[0+2]

    def generate_command(self, track):
        if self.x is None or self.track_phase == 1.0:
            return Twist()
        w = track.get_location(self.track_phase)
        dx = w.x - self.x
        dy = w.y - self.y
        desired_yaw = np.arctan2(dy, dx)

        yaw_diff = (desired_yaw - self.yaw + np.pi) % (2*np.pi) - np.pi


        if yaw_diff > np.pi / 3 or yaw_diff < -np.pi / 3:
            lin_vel = 0.0
        else:
            lin_vel = 0.4
        command = Twist()
        command.linear.x = lin_vel
        command.angular.z = np.clip(0.5 * (yaw_diff), -0.7, 0.7)
        return command

    def update_goal(self, track):
        next_phase = self.track_phase + 0.1
        next_phase = min(next_phase, 1.0)
        # Project phase back to make sure it's within reach of 1 meter
        N = 100 ## max iterations - perform binary search

        lb, ub = self.track_phase, next_phase
        for i in range(N):
            mid = (lb + ub) / 2
            w = track.get_location(mid)
            dx = w.x - self.x
            dy = w.y - self.y
            d = np.sqrt(dx * dx + dy * dy)
            if d >= self.tol:
                ub = mid
            else:
                lb = mid
        self.track_phase = lb


def main():

    # Just copy-pasted waypoint locations from world sdf
    t = ["0.037268 0.557564", "-8.99859 0.429756", "-12.0103 1.39205", "-12.0559 2.95944", 
         "-10.6984 2.95193", "-0.032725 2.93892", "1.64538 3.79553", 
         "1.0071 4.95789", "-10.4925 5.02624", "-13.0383 6.07977",
         "-10.1211 7.49309", "-0.030184 7.41345"]

    waypoints = [Waypoint(float(x), float(y)) for s in t for x, y in [s.split()]]
    track = Track(waypoints)
    robot = Robot(0, 0, 0, 0.4)

    rospy.init_node("tbai_benchmark")
    publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        command = robot.generate_command(track)
        print(command)
        robot.update_goal(track)
        publisher.publish(command)
        rate.sleep()

    
if __name__ == "__main__":
    main()