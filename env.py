#! /usr/bin/env python
"""Environment for Microsoft AirSim Unity Quadrotor using AirSim python API

- Author: Subin Yang
- Contact: subinlab.yang@gmail.com
- Date: 2019.06.20.
"""
import csv
import math
import pprint
import time

import torch
from PIL import Image

import numpy as np

import airsim
#import setup_path

MOVEMENT_INTERVAL = 1


class DroneEnv(object):
    """Drone environment class using AirSim python API"""

    def __init__(self):
        self.client = airsim.MultirotorClient()

        self.last_dist = 1000000
        collision = False
        self.quad_offset = (0, 0, 0)
        self.ep = 0

    def step(self, action):
        """Step"""
        #print("new step ------------------------------")

        self.quad_offset = self.interpret_action(action)
        #print("quad_offset: ", self.quad_offset)

        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            quad_vel.x_val + self.quad_offset[0],
            quad_vel.y_val + self.quad_offset[1],
            quad_vel.z_val + self.quad_offset[2],
            MOVEMENT_INTERVAL
        ).join()
        time.sleep(0.5)

        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        collision = self.client.simGetCollisionInfo().has_collided

        result = self.compute_reward(quad_state, quad_vel, collision)
        state = self.get_obs()
        done = self.isDone(result)
        return state, result, done

    def reset(self):
        self.client.reset()
        self.last_dist = 1000000
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        collision = False

        print("takeoff")
        self.client.takeoffAsync().join()
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(quad_state.x_val, quad_state.y_val, -7, 1).join()
        print("ready")

        obs = self.get_obs()

        return obs

    def get_obs(self):
        """Get observation"""
        responses = self.client.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
        )
        obs = self.transform_input(responses)
        return obs

    def get_distance(self, quad_state):
        """Get distance between current state and goal state"""
        pts = np.array([3, -76, -7])
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
        dist = np.linalg.norm(quad_pt - pts)
        return dist

    def compute_reward(self, quad_state, quad_vel, collision):
        """Compute reward"""

        reward = -1

        if collision:
            reward = -100
            print("collided")
        elif quad_state.z_val < -23:
            reward = -100
            print("too high")
        else:
            dist = self.get_distance(quad_state)


            diff = dist - self.last_dist
            #print("dist: ", dist, " last_dist: ", self.last_dist, "diff", diff)

            if diff > 0:
                reward = -2
            elif diff == 0.0:
                reward = -100
                print("stucked")
            elif diff < -1:
                reward = 2
            elif dist < 10:
                reward = 500

            self.last_dist = dist



        print("reward: ", reward)

        return reward


    def isDone(self, reward):
        """Check if episode is done"""
        done = 0
        if reward <= -10:
            done = 1
            self.reset()
            time.sleep(1)
        elif reward > 499:
            done = 1
            self.reset()
            time.sleep(1)
        return done

    def transform_input(self, responses):
        """Transform input binary array to image"""
        response = responses[0]
        img1d = np.fromstring(
            response.image_data_uint8, dtype=np.uint8
        )
        img_rgba = img1d.reshape(
            response.height, response.width, 3
        )
        image = Image.fromarray(img_rgba)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final

    def transformToTensor(self, img):
        tensor = torch.Tensor(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def interpret_action(self, action):
        """Interprete action"""
        scaling_factor = 3
        if action.item() == 0:
            self.quad_offset = (0, 0, 0)
        elif action.item() == 1:
            self.quad_offset = (scaling_factor, 0, 0)
        elif action.item() == 2:
            self.quad_offset = (0, scaling_factor, 0)
        elif action.item() == 3:
            self.quad_offset = (0, 0, scaling_factor)
        elif action.item() == 4:
            self.quad_offset = (-scaling_factor, 0, 0)
        elif action.item() == 5:
            self.quad_offset = (0, -scaling_factor, 0)
        elif action.item() == 6:
            self.quad_offset = (0, 0, -scaling_factor)

        return self.quad_offset
