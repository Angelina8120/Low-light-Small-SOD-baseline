#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)



class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        #------------------------------------
        #scale-1 3cheng3  x1,x2,x3,x4
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        #-------------------------------------------
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #-----------------------------------------
        spx = torch.split(out, self.width, 1) #divide tensor into blocks
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)# join out and sp by dimension(column)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        #-----------------------------------------
        # spx = torch.split(out, self.width, 1)
        # for i in range(self.nums):
        #     if i==0 or self.stype=='stage':
        #         sp = spx[i]
        #     else:
        #         sp = sp + spx[i]
        #     sp = self.convs[i](sp)
        #     sp = self.relu(self.bns[i](sp))
        #     out = sp
        # if self.scale != 1 and self.stype=='normal':
        #     out = torch.cat((out, spx[self.nums]),1)
        # elif self.scale != 1 and self.stype=='stage':
        #     out = torch.cat((out, self.pool(spx[self.nums])),1)

        #--------------------------------------------------------

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes=64
        super(Res2Net,self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        
        out2 = self.layer1(out1)

        #-----IGL------

        out3 = self.layer2(out2)

        #-----IGL------

        out4 = self.layer3(out3)

        #-----IGL------

        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/res2net50_26w_8s-2c7c9f12.pth'), strict=False)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg      = cfg
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./res/resnet50-19c8e357.pth'), strict=False)

#out1 = self.decoder([out0_0, out1_1, out2_2, out3_3], [0, out11i, out22i, 0])
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)


    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')

        #--------------IGL-----------------------------------------------------
        input2[1]=torch.mul(input1[1],torch.sigmoid(input2[1]))
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')

        #-------------------IGL-----------------------------------------------------
        input2[2]=torch.mul(input1[2],torch.sigmoid(input2[2]))

        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')

        #-------------------IGL------------------------------------------
        # input2[3]=

        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)

class Decodere(nn.Module):
    def __init__(self):
        super(Decodere, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0o = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0o, size=input1[1].size()[2:], mode='bilinear')
        out1o = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1o, size=input1[2].size()[2:], mode='bilinear')
        out2o = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2o, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out0o, out1o, out2o, out3
    
    def initialize(self):
        weight_init(self)

class Decoderi(nn.Module):
    def __init__(self):
        super(Decoderi, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0o = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0o, size=input1[1].size()[2:], mode='bilinear')
        out1o = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1o, size=input1[2].size()[2:], mode='bilinear')
        out2o = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2o, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out0o, out1o, out2o, out3
    
    def initialize(self):
        weight_init(self)
		

class LDF(nn.Module):
    def __init__(self, cfg):
        super(LDF, self).__init__()
        self.cfg      = cfg

        self.bkbone   = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 8, num_classes=1000)

        self.conv5   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5i   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4i   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3i   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2i   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv5e   = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4e   = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3e   = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2e   = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))


        self.decoder = Decoder() #saliency 
        self.decodere = Decodere() #edge
        self.decoderi = Decoderi() #illumination 


        self.lineare  = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.lineari  = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.initialize()

    def forward(self, x, shape=None):

        out1, out2, out3, out4, out5 = self.bkbone(x)

        out2b, out3b, out4b, out5b   = self.conv2(out2), self.conv3(out3), self.conv4(out4), self.conv5(out5)

        out2e, out3e, out4e, out5e   = self.conv2e(out2), self.conv3e(out3), self.conv4e(out4), self.conv5e(out5)

        out00, out11, out22, oute1 = self.decodere([out5e, out4e, out3e, out2e])

        #------------------------illumination branch---------------------------------------------------------------

        # out1_i, out2_i, out3_i, out4_i, out5_i = self.bkbone_i(x)

        out2i, out3i, out4i, out5i   = self.conv2i(out2), self.conv3i(out3), self.conv4i(out4), self.conv5i(out5)

        out00i, out11i, out22i, outi1 = self.decoderi([out5i, out4i, out3i, out2i])

        #-----------------------------------------------------------------------------------------------------------

        out0_0, out1_1, out2_2, out3_3 = self.decodere([out5b, out4b, out3b, out2b],[out00, out11, out22, oute1])

        out1 = self.decoder([out0_0, out1_1, out2_2, out3_3], [0, out11i, out22i, 0])

        out1  = torch.cat([oute1, out1], dim=1)

        if shape is None:
            shape = x.size()[2:]
        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        oute1 = F.interpolate(self.lineare(oute1), size=shape, mode='bilinear')
        outi1 = F.interpolate(self.lineari(outi1), size=shape, mode='bilinear')

        return out1, oute1, outi1

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
