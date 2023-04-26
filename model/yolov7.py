import math
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out

class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv7 = ConvBlock(out_channels * 4, out_channels * 4, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.conv1(x) # branch 1
        out2 = self.conv2(x) # branch 2
        out3 = self.conv3(out2)
        out4 = self.conv4(out3) # branch 3
        out5 = self.conv5(out4)
        out6 = self.conv6(out5) # branch 4
        out = torch.cat((out1, out2, out4, out6), dim=1)
        out = self.conv7(out)

        return out

class MP(nn.Module):
    def __init__(self, in_channels):
        super(MP, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.maxpool(x)
        out2 = self.conv1(out1) # branch 1
        out3 = self.conv2(x)
        out4 = self.conv3(out3) # branch 2
        out = torch.cat((out2, out4), dim=1)

        return out

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.elan1 = ELAN(128, 64)
        self.mp1 = MP(128)
        self.elan2 = ELAN(256, 128)
        self.mp2 = MP(256)
        self.elan3 = ELAN(512, 256)
        self.mp3 = MP(512)
        self.elan4 = ELAN(1024, 256)

    def forward(self, x):
        out = self.conv1(x) # 32 * H * W
        out = self.conv2(out) # 64 * (1 / 2) H * (1 / 2) W
        out = self.conv3(out) # 64 * (1 / 2) H * (1 / 2) W
        out = self.conv4(out) # 128 * (1 / 4) H * (1 / 4) W
        out = self.elan1(out) # 256 * (1 / 4) H * (1 / 4) W
        out = self.mp1(out) # 256 * (1 / 8) H * (1 / 8) W
        small = self.elan2(out) # 512 * (1 / 8) H * (1 / 8) W
        out = self.mp2(small) # 512 * (1 / 16) H * (1 / 16) W
        medium = self.elan3(out) # 1024 * (1 / 16) H * (1 / 16) W
        out = self.mp3(medium) # 1024 * (1 / 32) H * (1 / 32) W
        large = self.elan4(out) # 1024 * (1 / 32) H * (1 / 32) W

        return small, medium, large

class SPPCSPC(nn.Module):
    def __init__(self, in_channels):
        super(SPPCSPC, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(in_channels // 2, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv5 = ConvBlock(in_channels * 2, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = ConvBlock(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv7 = ConvBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.conv4(self.conv3(self.conv1(x)))
        out2 = self.maxpool1(out1)
        out3 = self.maxpool2(out1)
        out4 = self.maxpool3(out1)
        b1 = self.conv6(self.conv5(torch.cat((out1, out2, out3, out4), dim=1)))
        b2 = self.conv2(x)
        out = torch.cat((b1, b2), dim=1)
        out = self.conv7(out)

        return out

class ELANH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELANH, self).__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(out_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1)
        self.conv7 = ConvBlock(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.conv1(x) # branch 1
        out2 = self.conv2(x) # branch 2
        out3 = self.conv3(out2) # branch 3
        out4 = self.conv4(out3) # branch 4
        out5 = self.conv5(out4) # branch 5
        out6 = self.conv6(out5) # branch 6
        out = torch.cat((out1, out2, out3, out4, out5, out6), dim=1)
        out = self.conv7(out)

        return out

class MPH(nn.Module):
    def __init__(self, in_channels):
        super(MPH, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.maxpool(x)
        out2 = self.conv1(out1) # branch 1
        out3 = self.conv2(x)
        out4 = self.conv3(out3) # branch 2
        out = torch.cat((out2, out4), dim=1)

        return out

class RepConv(nn.Module):
    def __init__(self, in_channels):
        super(RepConv, self).__init__()

        self.act = nn.SiLU()
        self.b1 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(in_channels))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        return self.act(self.b1(x) + self.b2(x))

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

        self.spp = SPPCSPC(1024)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(1024, 256, kernel_size=1, stride=1, padding=0)
        self.elanh1 = ELANH(512, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = ConvBlock(512, 128, kernel_size=1, stride=1, padding=0)
        self.elanh2 = ELANH(256, 128)
        self.mph1 = MPH(128)
        self.elanh3 = ELANH(512, 256)
        self.mph2 = MPH(256)
        self.elanh4 = ELANH(1024, 512)
        self.repconv1 = RepConv(1024)
        self.repconv2 = RepConv(512)
        self.repconv3 = RepConv(256)

    def forward(self, x1, x2, x3):
        out1 = self.spp(x3) # 512 * (1 / 32) H * (1 / 32) W
        out2 = self.up1(self.conv1(out1)) # 256 * (1 / 16) H * (1 / 16) W
        out3 = self.conv2(x2) # 256 * (1 / 16) H * (1 / 16) W
        out4 = torch.cat((out2, out3), dim=1) # 512 * (1 / 16) H * (1 / 16) W
        out5 = self.elanh1(out4) # 256 * (1 / 16) H * (1 / 16) W
        out6 = self.up2(self.conv3(out5)) # 128 * (1 / 8) H * (1 / 8) W
        out7 = self.conv4(x1) # 128 * (1 / 8) H * (1 / 8) W
        out8 = torch.cat((out6, out7), dim=1) # 256 * (1 / 8) H * (1 / 8) W
        out9 = self.elanh2(out8) # 128 * (1 / 8) H * (1 / 8) W
        out10 = self.mph1(out9) # 256 * (1 / 16) H * (1 / 16) W
        out11 = torch.cat((out10, out5), dim=1) # 512 * (1 / 16) H * (1 / 16) W
        out12 = self.elanh3(out11) # 256 * (1 / 16) H * (1 / 16) W
        out13 = self.mph2(out12) # 512 * (1 / 32) H * (1 / 32) W
        out14 = torch.cat((out13, out1), dim=1) # 1024 * (1 / 32) H * (1 / 32) W
        out15 = self.elanh4(out14) # 512 * (1 / 32) H * (1 / 32) W
        small = self.repconv1(out15) # 1024 * (1 / 32) H * (1 / 32) W
        medium = self.repconv2(out12) # 512 * (1 / 16) H * (1 / 16) W
        large = self.repconv3(out9) # 256 * (1 / 8) H * (1 / 8) W

        return [large, medium, small]

class ImplicitAdd(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitAdd, self).__init__()

        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x

class ImplicitMul(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitMul, self).__init__()

        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class ImplicitDetect(nn.Module):
    def __init__(self, anchors, num_classes, stride):
        super(ImplicitDetect, self).__init__()

        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.anchors = anchors.view(self.num_layers, -1, 2) # [self.num_layers, self.num_anchors, 2]
        self.anchor_grid = self.anchors.clone().view(self.num_layers, 1, -1, 1, 1, 2)
        self.grid = [torch.zeros(1)] * self.num_layers
        self.num_classes = num_classes
        self.stride = stride
        self.channels = [256, 512, 1024]

        self.conv_out = nn.ModuleList(nn.Conv2d(x, (self.num_classes + 5) * self.num_anchors, kernel_size=1, stride=1, padding=0, bias=True) for x in self.channels)
        self.implicit_add = nn.ModuleList(ImplicitAdd(x) for x in self.channels)
        self.implicit_mul = nn.ModuleList(ImplicitMul((self.num_classes + 5) * self.num_anchors) for x in self.channels)

    def forward(self, x):
        z = []
        for i in range(0, self.num_layers):
            x[i] = self.conv_out[i](self.implicit_add[i](x[i]))
            x[i] = self.implicit_mul[i](x[i])
            B, C, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(B, self.num_anchors, self.num_classes + 5, grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                self.grid[i] = self.make_grid(grid_x, grid_y).to(x[i].device)
                y = x[i].sigmoid()
                y[:, :, :, :, 0:2] = (y[:, :, :, :, 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[:, :, :, :, 2:4] = ((y[:, :, :, :, 2:4] * 2) ** 2) * (self.anchor_grid[i].to(x[i].device))
                z.append(y.view(B, -1, self.num_classes + 5))

        if self.training:
            return x
        else:
            return x, torch.cat(z, dim=1)

    def make_grid(self, grid_x, grid_y):
        y, x = torch.meshgrid([torch.arange(grid_y), torch.arange(grid_x)], indexing='ij')

        return torch.stack((x, y), dim=2).view((1, 1, grid_y, grid_x, 2)).float()

class Model(nn.Module):
    def __init__(self, anchors, num_classes, stride):
        super(Model, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.num_classes = num_classes
        self.stride = torch.FloatTensor(stride)
        self.backbone = Backbone()
        self.head = Head()
        self.detect = ImplicitDetect(self.anchors, self.num_classes, self.stride)
        # initialize weights
        self.initialize_biases()
        self.initialize_weights()

    def forward(self, x):
        small, medium, large = self.backbone(x)
        out = self.head(small, medium, large)
        if self.training:
            out = self.detect(out)

            return out
        else:
            out, prediction = self.detect(out)

            return out, prediction

    def initialize_biases(self):
        for conv, s in zip(self.detect.conv_out, self.detect.stride):
            b = conv.bias.view(self.detect.num_anchors, -1) # [num_anchors, num_classes + 5]
            b.data[:, 4] += math.log(8 / ((640 / s) ** 2))
            b.data[:, 5:] += math.log(0.6 / (self.detect.num_classes - 0.99))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

