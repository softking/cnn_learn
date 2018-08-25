# -*- coding: utf-8 -*-

import pygame

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 800

TYPE_SMALL = 1
TYPE_MIDDLE = 2
TYPE_BIG = 3

plane_img = pygame.image.load('resources/image/shoot.png')
bullet_img = plane_img.subsurface(pygame.Rect(1004, 987, 9, 21))

# 子弹类
class Bullet(pygame.sprite.Sprite):
    def __init__(self, bullet_img, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.rect = self.image.get_rect()
        self.rect.midbottom = init_pos
        self.speed = 10

    def move(self):
        self.rect.top -= self.speed

# 玩家类
class Player(pygame.sprite.Sprite):
    def __init__(self, plane_img):
        pygame.sprite.Sprite.__init__(self)

        self.image = (plane_img.subsurface(pygame.Rect(0, 99, 102, 126)).convert_alpha())
        self.rect = pygame.Rect(0, 99, 102, 126)        # 初始化图片所在的矩形
        self.rect.topleft = [200, 600]                  # 初始化矩形的左上角坐标
        self.speed = 8                                  # 初始化玩家速度，这里是一个确定的值
        self.bullets = pygame.sprite.Group()            # 玩家飞机所发射的子弹的集合
        self.img_index = 0                              # 玩家精灵图片索引
        self.is_hit = False                             # 玩家是否被击中

    def shoot(self):

        # 定义子弹对象使用的surface相关参数
        bullet = Bullet(bullet_img, self.rect.midtop)
        self.bullets.add(bullet)

    def moveUp(self):
        if self.rect.top <= 0:
            self.rect.top = 0
        else:
            self.rect.top -= self.speed

    def moveDown(self):
        if self.rect.top >= SCREEN_HEIGHT - self.rect.height:
            self.rect.top = SCREEN_HEIGHT - self.rect.height
        else:
            self.rect.top += self.speed

    def moveLeft(self):
        if self.rect.left <= 0:
            self.rect.left = 0
        else:
            self.rect.left -= self.speed

    def moveRight(self):
        if self.rect.left >= SCREEN_WIDTH - self.rect.width:
            self.rect.left = SCREEN_WIDTH - self.rect.width
        else:
            self.rect.left += self.speed

# 敌人类
class Enemy(pygame.sprite.Sprite):
    def __init__(self, img, init_pos):
       pygame.sprite.Sprite.__init__(self)
       self.image = img
       self.rect = self.image.get_rect()
       self.rect.topleft = init_pos
       self.speed = 2
       self.down_index = 0

    def move(self):
        self.rect.top += self.speed