# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import torch as th

from utils.image_pool import ImagePool
from utils.loss import GANLoss
from utils.meters import AverageMeter


class Trainer(object):
    def __init__(self, opt, G_A, G_B, D_A, D_B, optimizer_G, optimizer_D, summary_writer):
        self.opt = opt
        self.device = th.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else th.device('cpu')
        self.G_A = G_A
        self.G_B = G_B
        self.D_A = D_A
        self.D_B = D_B
        # define optimizer G and D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

        self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
        self.criterionCycle = th.nn.L1Loss()
        self.criterionIdt = th.nn.L1Loss()
        self.summary_writer = summary_writer
        self.fake_B_pool = ImagePool(self.opt.pool_size)
        self.fake_A_pool = ImagePool(self.opt.pool_size)

    def train(self, epoch, data_loader):
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_G = AverageMeter()
        loss_D = AverageMeter()

        start = time.time()
        for i, data in enumerate(data_loader):
            self._parse_data(data)
            data_time.update(time.time() - start)
            self._forward()
            # optimizer G_A and G_B
            self.set_requires_grad([self.D_A, self.D_B], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # optimizer D_A and D_B
            self.set_requires_grad([self.D_A, self.D_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
            batch_time.update(time.time() - start)
            start = time.time()
            loss_G.update(self.loss_G.item())
            loss_D.update(self.loss_D_A.item() + self.loss_D_B.item())
            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch {} [{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss_G {:.3f} ({:.3f})\t'
                      'Loss_D {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              loss_G.val, loss_G.mean,
                              loss_D.val, loss_D.mean))
        print('Epoch {}\tEpoch Time: {:.3f}\tLoss_G: {:.3f}\tLoss_D: {:.3f}\t'
              .format(epoch, batch_time.sum, loss_G.mean, loss_D.mean))
        print()

    def backward_D_basic(self, netD, real, fake):
        # real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # combine loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambdaA = self.opt.lambda_A
        lambdaB = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed
            self.idt_A = self.G_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambdaB * lambda_idt
            # G_B should be identity if real_A is fed
            self.idt_B = self.G_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambdaA * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A)) and D_B(G_B(B))
        self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A), True)
        # forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambdaA
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambdaB
        # combine loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
                      + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def _parse_data(self, inputs):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = inputs['A' if AtoB else 'B'].to(self.device)
        self.real_B = inputs['B' if AtoB else 'A'].to(self.device)

    def _forward(self):
        self.fake_B = self.G_A(self.real_A)
        self.rec_A = self.G_B(self.fake_B)

        self.fake_A = self.G_B(self.real_B)
        self.rec_B = self.G_A(self.fake_A)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
