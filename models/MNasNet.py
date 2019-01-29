from torch.autograd import Variable
import torch.nn as nn
import torch
import math


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(inplace=True)
    )

def SepConv_3x3(inp, oup): #input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.LeakyReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.LeakyReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




def make_layers(setting, input_channel, width_mult):
    features = []
    for t, c, n, s, k in setting:
        output_channel = int(c * width_mult)
        for i in range(n):
            if i == 0:
                features.append(InvertedResidual(input_channel, output_channel, s, t, k))
            else:
                features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
            input_channel = output_channel
    return features

class RBB(nn.Module):
    '''
    边界修整模块
    '''
    def __init__(self, in_ch):
        super(RBB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2),
            nn.BatchNorm2d(in_ch)
        )
    def forward(self, input):
        x = self.conv(input)
        x = nn.LeakyReLU()(x + input)
        return x


class up(nn.Module):
    '''
    上升模块，用于上采样和特征融合,提取mask
    '''
    def __init__(self, low_ch, high_ch):
        super(up, self).__init__()
        self.conv_low_1x1 = nn.Conv2d(low_ch, high_ch, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_in = RBB(low_ch)
        self.conv_out = RBB(high_ch)

    def forward(self, low_x, high_x):
        low_x = self.conv_in(low_x)
        low_x = self.conv_low_1x1(low_x)
        low_x = self.upsample(low_x)
        high_x = high_x + low_x
        high_x = self.conv_out(high_x)
        return high_x

class OutPut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutPut, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


class MnasNet(nn.Module):
    def __init__(self, width_mult=1.):
        self.seen = 0
        super(MnasNet, self).__init__()

        # setting of inverted residual blocks
        self.block0 = [
            [3, 32, 3, 2, 3],  # -> 56x56
        ]
        self.block1 = [
            # t, c, n, s, k
            [3, 64, 3, 2, 5],  # -> 28x28
        ]
        self.block2 = [
            # t, c, n, s, k
            [6, 64, 3, 2, 5],  # -> 14x14
            [6, 128, 2, 1, 3],  # -> 14x14
        ]
        self.block3 = [
            # t, c, n, s, k
            [6, 128, 4, 2, 5],  # -> 7x7
            [6, 256, 1, 1, 3],  # -> 7x7
        ]

        # building  layers
        self.input = [Conv_3x3(3, 32, 2), SepConv_3x3(32, 32)]
        self.block0 = make_layers(self.block0, 32, width_mult)
        self.block1 = make_layers(self.block1, 32, width_mult)
        self.block2 = make_layers(self.block2, 64, width_mult)
        self.block3 = make_layers(self.block3, 128, width_mult)

        # make it nn.Sequential
        self.input = nn.Sequential(*self.input)
        self.block0 = nn.Sequential(*self.block0)
        self.block1 = nn.Sequential(*self.block1)
        self.block2 = nn.Sequential(*self.block2)
        self.block3 = nn.Sequential(*self.block3)
        self._initialize_weights()
        self.up26 = up(low_ch=256, high_ch=128)
        self.up52 = up(low_ch=128, high_ch=64)
        self.up104 = up(low_ch=64, high_ch=32)
        self.out13 = OutPut(in_ch=256, out_ch=21)
        self.out26 = OutPut(in_ch=128, out_ch=21)
        self.out52 = OutPut(in_ch=64, out_ch=21)
        self.out104 = OutPut(in_ch=32, out_ch=1)
    def forward(self, x):
        down208 = self.input(x)
        down104 = self.block0(down208)
        down52 = self.block1(down104)
        down26 = self.block2(down52)
        down13 = self.block3(down26)
        up26 = self.up26(down13, down26)
        up52 = self.up52(up26, down52)
        up104 = self.up104(up52, down104)
        out13 = self.out13(down13)
        out26 = self.out26(up26)
        out52 = self.out52(up52)
        out104 = self.out104(up104)
        return out13, out26, out52, out104

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = MnasNet()
    x_image = Variable(torch.randn(1, 3, 416, 416))
    y = net(x_image)
    print(y)