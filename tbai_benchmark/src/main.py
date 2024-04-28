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

        command = Twist()
        command.linear.x = 1.0
        command.angular.z = np.clip(0.5 * (yaw_diff), -0.5, 0.5)
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

    Waypoints = [Waypoint(0, 0), Waypoint(3, 0), Waypoint(3, 3), Waypoint(5,5), Waypoint(0, 3)]
    track = Track(Waypoints)
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