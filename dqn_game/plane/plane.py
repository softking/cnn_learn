# -*- coding: utf-8 -*-
import pygame
import gym
from sys import exit
from pygame.locals import *
from role import *
import random

# 初始化游戏
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('飞机大战')

# 载入背景图
background = pygame.image.load('resources/image/background.png').convert()
game_over = pygame.image.load('resources/image/gameover.png')
plane_img = pygame.image.load('resources/image/shoot.png')

# 定义敌机对象使用的surface相关参数
enemy_rect = pygame.Rect(534, 612, 57, 43)
enemy_img = plane_img.subsurface(enemy_rect)


class Plane(gym.Env):
    """
    """
    def __init__(self):
        """
        """
        self.player = Player(plane_img)
        self.enemies = pygame.sprite.Group()
        self.player_down_index = 16
        self.score = 0

    def reset(self):
        self.player = Player(plane_img)
        self.enemies = pygame.sprite.Group()
        self.player_down_index = 16
        self.score = 0

        return self.build_data()

    def build_data(self):
        """
        """
        data = []
        data.append(self.player.rect.topleft)
        for i in self.enemies:
            data.append(i.rect.topleft)

        for i in range(len(data), 10):
            data.append([0,0])

        return data

    def step(self, action):
        """
        """
        done = False
        reward = 0

        # 生成敌机
        while len(self.enemies) < 5:
            enemy_pos = [random.randint(0, SCREEN_WIDTH - enemy_rect.width), 0]
            enemy = Enemy(enemy_img, enemy_pos)
            self.enemies.add(enemy)


        # 移动子弹，若超出窗口范围则删除
        for bullet in self.player.bullets:
            bullet.move()
            if bullet.rect.bottom < 0:
                self.player.bullets.remove(bullet)

        # 移动敌机，若超出窗口范围则删除
        for enemy in self.enemies:
            enemy.move()
            # 判断玩家是否被击中
            if pygame.sprite.collide_circle(enemy, self.player):
                self.enemies.remove(enemy)
                done = True
                reward = -50
                break
            if enemy.rect.top > SCREEN_HEIGHT:
                self.enemies.remove(enemy)
                reward += 2

        # 将被击中的敌机对象添加到击毁敌机Group中，用来渲染击毁动画
        enemies_down = pygame.sprite.groupcollide(self.enemies, self.player.bullets, 1, 1)
        self.score += len(enemies_down)

        reward += (len(enemies_down)*10)

        pygame.event.get()

        {
            0: self.player.moveUp,
            1: self.player.moveDown,
            2: self.player.moveLeft,
            3: self.player.moveRight,
            4: self.player.shoot
        }[action]()

        return self.build_data(), reward, done


    def render(self, mode='human'):
        """
        """

        # 绘制背景
        screen.fill(0)
        screen.blit(background, (0, 0))
        # 绘制玩家飞机
        screen.blit(self.player.image, self.player.rect)
        # 绘制子弹和敌机
        self.player.bullets.draw(screen)
        self.enemies.draw(screen)
        # 绘制得分
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render(str(self.score), True, (128, 128, 128))
        text_rect = score_text.get_rect()
        text_rect.topleft = [10, 10]
        screen.blit(score_text, text_rect)
        # 更新屏幕
        pygame.display.update()


