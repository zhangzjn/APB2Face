import cv2
import numpy as np
import os
import time
import logging

import torch
import torch.nn as nn

from APBNet import *
from utils import *
from loss import *



class GANModel():

    def __init__(self, opt, logger):

        self.gpus = opt.gpus[0]
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.every = opt.every
        self.epoch = 0
        self.best_loss = float("inf")
        self.idt_name = opt.idt_name
        self.logdir = opt.logdir
        self.logger = logger
        # loss
        self.criterionGAN = GANLoss(gan_mode='mse').cuda()
        self.criterionL1 = nn.L1Loss()
        # G
        self.netG = APBNet()
        self.netG.apply(weight_init)
        if opt.resume:
            checkpoint = torch.load('{}/{}.pth'.format(self.logdir, opt.resume_epoch if opt.resume_epoch else '{}_best'.format(self.idt_name)))
            self.netG.load_state_dict(checkpoint['net_G'])
            self.epoch = checkpoint['epoch']
        self.netG.cuda()
        # D
        if self.isTrain:  # define discriminators
            self.netD = Discriminator()
            self.netD.apply(weight_init)
            if opt.resume:
                self.netD.load_state_dict(checkpoint['net_D'])
            self.netD.cuda()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.99, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.99, 0.999))

    def train(self):
        self.isTrain = True

    def eval(self):
        self.isTrain = False

    def reset(self):
        self.loss_log_L1 = 0
        self.loss_log_G_A = 0

        self.loss_log_D_A_F = 0
        self.loss_log_D_A_T = 0

    def test_draw(self, dataloader):
        def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
            for i in range(len(shape) // 2):
                img = cv2.circle(img, (int(shape[2 * i]), int(shape[2 * i + 1])), radius, color, thickness)
            return img

        def drawArrow(img, shape1, shape2, ):
            for i in range(len(shape1) // 2):
                point1 = (int(shape1[2 * i]), int(shape1[2 * i + 1]))
                point2 = (int(shape1[2 * i] + shape2[2 * i]), int(shape1[2 * i + 1] + shape2[2 * i + 1]))
                img = cv2.circle(img, point2, radius=6, color=(0, 0, 255), thickness=2)
                img = cv2.line(img, point1, point2, (255, 255, 255), thickness=2)
            return img

        root = self.logdir
        s_pathA = '{}/resultA'.format(root)
        s_pathB = '{}/resultB'.format(root)
        if not os.path.exists(s_pathA):
            os.mkdir(s_pathA)
        if not os.path.exists(s_pathB):
            os.mkdir(s_pathB)
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                self.set_input(data)
                self.forward()
                img_size = 256
                img_template = np.zeros((img_size, img_size, 3))
                img_fake_A1 = drawCircle(img_template.copy(), self.fake_A.squeeze(0).data, radius=1,
                                         color=(255, 255, 255), thickness=2)
                img_A1 = drawCircle(img_template.copy(), self.land_A1.squeeze(0).data, radius=1,
                                    color=(255, 255, 255), thickness=2)
                img_fake_B1 = drawCircle(img_template.copy(), self.fake_B.squeeze(0).data, radius=1,
                                         color=(255, 255, 255), thickness=2)
                img_B1 = drawCircle(img_template.copy(), self.land_B1.squeeze(0).data, radius=1,
                                    color=(255, 255, 255), thickness=2)

                img_compareA = np.concatenate([img_template[:, :, 0][:, :, np.newaxis], img_fake_A1[:, :, 0][:, :, np.newaxis],
                                               img_A1[:, :, 0][:, :, np.newaxis]], axis=2)
                img_compareB = np.concatenate([img_template[:, :, 0][:, :, np.newaxis], img_fake_B1[:, :, 0][:, :, np.newaxis],
                                               img_A1[:, :, 2][:, :, np.newaxis]], axis=2)
                cv2.imwrite('{}/{}.jpg'.format(s_pathA, batch_idx), img_compareA)
                cv2.imwrite('{}/{}.jpg'.format(s_pathB, batch_idx), img_compareB)
                print('\r{}'.format(batch_idx + 1), end='')

    def run_train(self, dataloader, epoch=None):
        self.epoch += 1
        if epoch:
            self.epoch = epoch
        self.reset()
        adjust_learning_rate(self.optimizer_G, self.lr, self.epoch, every=self.every)
        adjust_learning_rate(self.optimizer_D, self.lr, self.epoch, every=self.every)
        for batch_idx, train_data in enumerate(dataloader):
            self.batch_idx = batch_idx + 1
            self.set_input(train_data)
            self.optimize_parameters()
            log_string = 'train\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_L1 {:.5f}'.format(self.loss_log_L1 / (batch_idx + 1))
            log_string += '|loss_G_A {:.5f}'.format(self.loss_log_G_A / (batch_idx + 1))
            log_string += '|loss_D_A_F {:.5f}'.format(self.loss_log_D_A_F / (batch_idx + 1))
            log_string += '|loss_D_A_T {:.5f}'.format(self.loss_log_D_A_T / (batch_idx + 1))
            print('\r' + log_string, end='')
        print('\r', end='')
        self.logger.info(log_string)

    def run_test(self, dataloader, epoch=None):
        if epoch:
            self.epoch = epoch
        self.reset()
        for batch_idx, test_data in enumerate(dataloader):
            self.batch_idx = batch_idx + 1
            self.set_input(test_data)
            self.evaluate_loss()
            log_string = 'test\t -> '
            log_string += 'epoch {:>3} '.format(self.epoch)
            log_string += 'batch {:>4} '.format(batch_idx + 1)
            log_string += '|loss_L1 {:.5f}'.format(self.loss_log_L1 / (batch_idx + 1))
            log_string += '|loss_G_A {:.5f}'.format(self.loss_log_G_A / (batch_idx + 1))
            log_string += '|loss_D_A_F {:.5f}'.format(self.loss_log_D_A_F / (batch_idx + 1))
            log_string += '|loss_D_A_T {:.5f}'.format(self.loss_log_D_A_T / (batch_idx + 1))
            print('\r'+log_string, end='')
        print('\r', end='')
        self.logger.info(log_string)
        if self.loss_log_L1 / self.batch_idx < self.best_loss and not self.isTrain:
            self.best_loss = self.loss_log_L1 / self.batch_idx
            self.logger.info('save_best {:.5f}'.format(self.best_loss))
            self.save(mode='best')
        if self.epoch % 50 == 0:
            self.logger.info('save_epoch {:d}'.format(self.epoch))
            self.save(mode=self.epoch)

    def set_input(self, training_data):
        self.audio_feature_A1, self.pose_A1, self.eye_A1 = training_data[0][0].to(self.gpus),\
                                                           training_data[0][1].to(self.gpus),\
                                                           training_data[0][2].to(self.gpus)
        self.land_A1, self.land_A2 = training_data[1][0].to(self.gpus), \
                                     training_data[1][1].to(self.gpus)

    def optimize_parameters(self):
        self.forward()
        # G
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        if self.batch_idx % 1 == 0:
            self.set_requires_grad([self.netD], True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

    def evaluate_loss(self):
        self.forward()
        # G
        self.loss_L1 = self.criterionL1(self.fake_A, self.land_A1)
        self.loss_G_A = self.criterionGAN(self.netD(self.fake_A, self.land_A2), True)
        self.loss_log_L1 += self.loss_L1.item()
        self.loss_log_G_A += self.loss_G_A.item()
        # D
        loss_D_A_F = self.criterionGAN(self.netD(self.fake_A.detach(), self.land_A2.detach()), False)
        loss_D_A_T = self.criterionGAN(self.netD(self.land_A1.detach(), self.land_A2.detach()), True)
        self.loss_log_D_A_F += loss_D_A_F.item()
        self.loss_log_D_A_T += loss_D_A_T.item()

    def forward(self):
        self.fake_A = self.netG(self.audio_feature_A1, self.pose_A1, self.eye_A1)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_G(self):
        lambda_L1 = 100
        lambda_gan = 0.1

        self.loss_L1 = self.criterionL1(self.fake_A, self.land_A1)
        self.loss_G_A = self.criterionGAN(self.netD(self.fake_A, self.land_A2), True)

        self.loss_G = self.loss_L1 * lambda_L1 + self.loss_G_A * lambda_gan
        self.loss_G.backward()
        # log
        self.loss_log_L1 += self.loss_L1.item()
        self.loss_log_G_A += self.loss_G_A.item()

    def backward_D(self):
        lambda_D = 0.1
        loss_D_A_F = self.criterionGAN(self.netD(self.fake_A.detach(), self.land_A2.detach()), False)
        loss_D_A_T = self.criterionGAN(self.netD(self.land_A1.detach(), self.land_A2.detach()), True)
        # Combined loss and calculate gradients
        loss_D = (loss_D_A_F + loss_D_A_T) * 0.5 * lambda_D
        loss_D.backward()
        # log
        self.loss_log_D_A_F += loss_D_A_F.item()
        self.loss_log_D_A_T += loss_D_A_T.item()

    def save(self, mode=None):
        state = {
            'net_G': self.netG.state_dict(),
            'net_D': self.netD.state_dict(),
            'epoch': self.epoch,
        }
        torch.save(state, '{}/{}.pth'.format(self.logdir, '{}_{}'.format(self.idt_name, mode if mode else self.epoch)))
