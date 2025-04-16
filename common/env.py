import numpy as np
import random
import math
import matplotlib.pyplot as plt
import cv2


class V2Ichannels:
    # Simulator of the V2I channels
    def __init__(self):
        self.h_bs = 25                               # RSU高度
        self.h_ms = 1.5                              # 车辆高度
        self.Decorrelation_distance = 50
        self.BS_position = [800 / 2, 0]             # 路口处，假设单条公路场景，将监控范围边缘作为起点0，则RSU坐标为监测范围，此处取平行公路间距的一半
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)



class Vehicle:
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []



class Env:
    def __init__(self):
        self.v2i_channel = V2Ichannels()
        self.vehicles = []
        self.vehicles_num = 10