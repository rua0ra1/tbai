#!/usr/bin/env python3


import time
import numpy as np
import matplotlib.pyplot as plt


class Waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Track:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.total_length = self.get_length()
        self.sublengths = self.get_sublengths()

    def get_length(self):
        length = 0.0
        for i in range(len(self.waypoints)):
            ip = (i + 1) % len(self.waypoints)
            dx = self.waypoints[ip].x - self.waypoints[i].x
            dy = self.waypoints[ip].y - self.waypoints[i].y
            length += np.sqrt(dx * dx + dy * dy)
        return length

    def get_sublengths(self):
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

    def get_location(self, phase):
        assert phase >= 0.0 and phase <= 1.0, "Phase out of bounds"
        length = phase * self.total_length
        for i in range(len(self.waypoints)):
            if length < self.sublengths[i + 1]:
                break
        wp = self.waypoints[i]
        wp_next = self.waypoints[(i + 1) % len(self.waypoints)]
        x = wp.x + (wp_next.x - wp.x) * (length - self.sublengths[i]) / (
            self.sublengths[i + 1] - self.sublengths[i]
        )
        y = wp.y + (wp_next.y - wp.y) * (length - self.sublengths[i]) / (
            self.sublengths[i + 1] - self.sublengths[i]
        )
        return Waypoint(x, y)

    def plot(self, n_points=100):
        xs, ys = list(), list()
        for i in range(n_points):
            phase = i / n_points
            w = self.get_location(phase)
            xs.append(w.x)
            ys.append(w.y)
        plt.scatter(xs, ys)
        plt.show()


class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.track_phase = 0.0

    def move(self, track):
        w = track.get_location(self.track_phase)
        dx = w.x - self.x
        dy = w.y - self.y
        self.x += dx * 0.1
        self.y += dy * 0.1

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
            if d >= 0.1:
                ub = mid
            else:
                lb = mid
        self.track_phase = lb
        print(self.track_phase)

    def simulate(self, track):
        xs, ys = list(), list()

        t1 = time.time()
        while self.track_phase < 1.0:
            self.move(track)
            self.update_goal(track)
            xs.append(self.x)
            ys.append(self.y)
            # xs.append(w.x)
            # ys.append(w.y)
        t2 = time.time()
        print("Time taken: ", t2 - t1)

        txs, tys = list(), list()
        for i in range(len(track.waypoints)):
            txs.append(track.waypoints[i].x)
            tys.append(track.waypoints[i].y)

        plt.plot(txs, tys, "ro-")

        plt.scatter(xs, ys)
        plt.show()

if __name__ == "__main__":
    waypoints = [
        Waypoint(0, 0),
        Waypoint(1, 0),
        Waypoint(1, 1),
        Waypoint(1, 3),
        Waypoint(-1, 3),
        Waypoint(3, 3),
        Waypoint(3.123, 1),
        Waypoint(0, 0),
    ]
    track = Track(waypoints)

    robot = Robot(0, 0)
    robot.simulate(track)
