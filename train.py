#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from net  import LDF

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='../data/DUTS-small', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=60)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, edge, illumination) in enumerate(loader):
            image, mask, edge, illumination = image.cuda(), mask.cuda(), edge.cuda(), illumination.cuda()
            out1, oute1, outi1 = net(image)
            
            losse1 = F.binary_cross_entropy_with_logits(oute1, edge)
            loss1  = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)

            loss_illu = F.l1_loss(outi1, illumination)


            loss = (losse1+loss1)/2+loss_illu

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'losse1':losse1.item(), 'loss':loss.item(), 'loss_illu': loss_illu.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | losse1=%.6f | loss1=%.6f| loss_illu=%.6f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), losse1.item(), loss1.item(), loss_illu.item()))

        if epoch > cfg.epoch*2/3:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, LDF)
