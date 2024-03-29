#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import pygame,sys,time,random
from pygame.locals import *
# 定义颜色变量
redColour = pygame.Color(255,0,0)
blackColour = pygame.Color(0,0,0)
whiteColour = pygame.Color(255,255,255)
greenColour = pygame.Color(0,255,0)

class Snake(gym.Env):
    """
    """
    def __init__(self):
        self.fpsClock = pygame.time.Clock()
        self.is_show = False
        
        self.step_num = 0

        self.snakePosition = [100, 100]
        self.snakeSegments = [[100, 100], [80, 100], [60, 100]]
        x = random.randrange(3, 10)
        y = random.randrange(3, 10)
        self.raspberryPosition = [int(x * 20), int(y * 20)]
        self.direction = 0


    def set_show(self, is_show=True):
        self.is_show = is_show

        if not is_show: return
        
        self.playSurface = pygame.display.set_mode((320, 240))
        pygame.init()
        pygame.display.set_caption('Raspberry Snake')


    def build_data(self):
        """
        """
        data = np.full((16, 12, 1), 0)
        data[int(self.snakePosition[0]/20-1)][int(self.snakePosition[1]/20-1)][0] += 1     # 头
        data[int(self.raspberryPosition[0] / 20 - 1)][int(self.raspberryPosition[1] / 20 - 1)][0] += 4  # 草莓
        for i in self.snakeSegments[1:]:
            data[int(i[0] / 20 - 1)][int(i[1] / 20 - 1)][0] += 2  # 身子
        return data

        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # return image_data

    def step(self, action):
        """
        """
        # 检测例如按键等pygame事件
        if self.is_show: pygame.event.get()

        reward = 1
        done = False

        self.direction = action

        # 根据方向移动蛇头的坐标
        if self.direction == 0:
            self.snakePosition[0] += 20
        if self.direction == 1:
            self.snakePosition[0] -= 20
        if self.direction == 2:
            self.snakePosition[1] -= 20
        if self.direction == 3:
            self.snakePosition[1] += 20
        # 增加蛇的长度

        raspberrySpawned = 1
        self.snakeSegments.insert(0, list(self.snakePosition))
        # 判断是否吃掉了树莓
        if self.snakePosition[0] == self.raspberryPosition[0] and self.snakePosition[1] == self.raspberryPosition[1]:
            raspberrySpawned = 0
        else:
            self.snakeSegments.pop()

        self.step_num += 1

        # reward -= self.step_num * 0.2

        # 如果吃掉树莓，则重新生成树莓
        if raspberrySpawned == 0:
            x = random.randrange(3, 10)
            y = random.randrange(3, 10)
            self.raspberryPosition = [int(x * 20), int(y * 20)]

            reward = 100

        # 判断是否死亡
        if self.snakePosition[0] > 300 or self.snakePosition[0] < 0:
            self.reset()
            reward = -100
            done = True

        if self.snakePosition[1] > 220 or self.snakePosition[1] < 0:
            self.reset()
            reward = -100
            done = True

        for snakeBody in self.snakeSegments[1:]:
            if self.snakePosition[0] == snakeBody[0] and self.snakePosition[1] == snakeBody[1]:
                self.reset()
                reward = -100
                done = True

        # 控制游戏速度
        # self.fpsClock.tick(5)

        return self.build_data(), reward, done, len(self.snakeSegments) - 3

    def reset(self):
        self.snakePosition = [100, 100]
        self.snakeSegments = [[100, 100], [80, 100], [60, 100]]
        x = random.randrange(3, 10)
        y = random.randrange(3, 10)
        self.raspberryPosition = [int(x * 20), int(y * 20)]
        self.direction = 0
        self.step_num = 0
        return self.build_data()


    def render(self, mode='human', close=False):
        # 绘制pygame显示层
        self.playSurface.fill(blackColour)

        pygame.draw.rect(self.playSurface, greenColour, Rect(self.snakePosition[0], self.snakePosition[1], 20, 20))
        for position in self.snakeSegments[1:]:
            pygame.draw.rect(self.playSurface, whiteColour, Rect(position[0], position[1], 20, 20))

        pygame.draw.rect(self.playSurface, redColour, Rect(self.raspberryPosition[0], self.raspberryPosition[1], 20, 20))

        # 刷新pygame显示层
        pygame.display.flip()
        # self.fpsClock.tick(5)

        # from gym.envs.classic_control import rendering
        # viewer = rendering.Viewer(640, 480)
        #
        # return viewer.render(return_rgb_array=mode == 'rgb_array')
