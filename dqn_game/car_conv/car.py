#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym

import numpy as np

import pygame
import random
from pygame.locals import *
# 定义颜色变量
redColour = pygame.Color(255,0,0)
blackColour = pygame.Color(0,0,0)
whiteColour = pygame.Color(255,255,255)
greenColour = pygame.Color(0,255,0)

SHOW_POS = range(9)
RANGE = 40
W = 10
H = 16
P = 70   # 出现炮弹概率

class Car(gym.Env):
    """
    """
    def __init__(self):
        self.fpsClock = pygame.time.Clock()
        self.is_show = False

        self.score = 0
        self.position = 0
        self.shell_pos = {}

    def reset(self):
        """
        """
        self.score = 0
        self.position = int(W/2)
        self.shell_pos = {}

        pos = random.choice(SHOW_POS)
        self.shell_pos[pos] = [pos, 0]

        return self.build_data()


    def set_show(self, is_show=True):
        """
        """
        self.is_show = is_show
        if not is_show: return
        
        self.playSurface = pygame.display.set_mode((W*RANGE, H*RANGE))
        pygame.init()
        pygame.display.set_caption('Car')


    def build_data(self):
        """
        """
        data = np.full((W, H, 1), 0)
        data[self.position][H-1][0] = 1     # 车位置

        for i in self.shell_pos:
            data[self.shell_pos[i][0]][self.shell_pos[i][1]][0] = 2  # 炮弹

        return data

        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # return image_data

    def step(self, action):
        """
        """
        # 检测例如按键等pygame事件
        if self.is_show: pygame.event.get()

        done = False

        if action == 1 and self.position > 0:
            self.position -= 1
        if action == 2 and self.position < W-2:
            self.position += 1
        if action == 0:
            pass

        reward = 0.1
        drop_list = []
        # 下落一下
        for i in self.shell_pos:
            self.shell_pos[i][1] += 1
            if self.shell_pos[i][1] >= H:
                drop_list.append(i)

        # 剔除出界的炮弹
        for i in drop_list:
            del self.shell_pos[i]
            reward = 1
            self.score += 1

        # 是否随机出新的
        if random.randint(0,100) >= P:
            pos = random.choice(SHOW_POS)
            if pos not in self.shell_pos:
                self.shell_pos[pos] = [pos, 0]

        # 判断撞击
        for i in self.shell_pos:
            if self.shell_pos[i][0] == self.position and self.shell_pos[i][1] == H-1:  # 撞了
                reward = -1
                done = True
                break


        # 控制游戏速度
        # self.fpsClock.tick(5)

        return self.build_data(), reward, done


    def render(self, mode='human', close=False):
        # 绘制pygame显示层
        self.playSurface.fill(blackColour)


        # self.playSurface.blit(pygame.image.load("assets/car.jpg").convert_alpha(), (100, 30))

        myfont = pygame.font.Font(None, 60)
        textImage = myfont.render(str(self.score), True, (255, 255, 255))
        self.playSurface.blit(textImage, (0, 0))

        pygame.draw.rect(self.playSurface, greenColour, Rect(self.position*RANGE, (H-1)*RANGE, RANGE, RANGE))

        for i in self.shell_pos:
            pygame.draw.rect(self.playSurface, redColour, Rect(self.shell_pos[i][0]*RANGE, self.shell_pos[i][1]*RANGE, RANGE, RANGE))

        # 刷新pygame显示层
        pygame.display.flip()
        # self.fpsClock.tick(5)

        # from gym.envs.classic_control import rendering
        # viewer = rendering.Viewer(640, 480)
        #
        # return viewer.render(return_rgb_array=mode == 'rgb_array')
