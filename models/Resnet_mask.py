from torch.autograd import Variable
import torch.nn as nn
import torch
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x, x2, x3, x4, x5




def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


class ConvLSTM(nn.Module):

    def __init__(self, in_ch, out_ch, size):
        super(ConvLSTM, self).__init__()
        self.wf = nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1)
        self.wi = nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1)
        self.wc = nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1)
        self.wo = nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1)
        self.h = torch.zeros(size).to('cuda')
        self.c = torch.zeros(size).to('cuda')

    def forward(self, x):
        # print('***', x.shape, self.h.shape)
        ft = nn.Sigmoid()(self.wf(torch.cat((x, self.h), 1)))
        it = nn.Sigmoid()(self.wi(torch.cat((x, self.h), 1)))
        ctt = nn.Tanh()(self.wc(torch.cat((x, self.h), 1)))
        ct = ft * self.c + it * ctt
        ot = self.wo(torch.cat((x, self.h), 1))
        h = ot * ct
        self.h = h.detach()
        self.c = ct.detach()
        return h



class RBB(nn.Module):
    '''
    边界修整模块
    '''
    def __init__(self, in_ch):
        super(RBB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, dilation=2,  padding=2),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, dilation=2,  padding=2),
            nn.BatchNorm2d(in_ch)
        )
    def forward(self, input):
        x = self.conv(input)
        x = nn.ReLU()(x + input)
        return x


class up(nn.Module):
    '''
    上升模块，用于上采样和特征融合,提取mask
    '''
    def __init__(self, low_ch, high_ch, up_scale=2, num_block=2):
        super(up, self).__init__()
        self.conv_low_1x1 = nn.Conv2d(low_ch, high_ch, 1, 1)
        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear')
        self.conv_in = RBB(high_ch)
        self.conv_out = nn.Sequential(RBB(high_ch),
                                      RBB(high_ch))



    def forward(self, low_x, high_x):
        low_x = self.conv_low_1x1(low_x)
        low_x = self.upsample(low_x)
        high_x = self.conv_in(high_x)
        high_x = high_x + low_x
        high_x = self.conv_out(high_x)
        return high_x


class OutPut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutPut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

resnet = resnet34().to('cuda')
# resnet.load_state_dict(torch.load('./backup/pretrain/resnet34.pth'))

class MaskNet(nn.Module):
    def __init__(self):
        self.seen = 0
        super(MaskNet, self).__init__()
        self._initialize_weights()
        self.basenet = resnet

        self.up26 = up(low_ch=512, high_ch=256)
        self.up52 = up(low_ch=256, high_ch=128)
        self.up104 = up(low_ch=128, high_ch=64)
        # self.up208 = up(low_ch=64, high_ch=64)
        self.out104 = OutPut(in_ch=64, out_ch=1)

    def forward(self, image):
        x0, x1, x2, x3, x4 = resnet(image)
        up26 = self.up26(x4, x3)
        up52 = self.up52(up26, x2)
        up104 = self.up104(up52, x1)
        # up208 = self.up208(up104, x0)
        out104 = self.out104(up104)
        return out104

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                torch.nn.init.xavier_normal_(m.weight.data, 1)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
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
    # torchvision.models.resnet34(pretrained=True)
    net = MaskNet()
    resnet = resnet34()
    resnet.load_state_dict(torch.load('./backup/pretrain/resnet34-333f7ec4.pth'))
    tm = resnet
    x_image = Variable(torch.randn(1, 3, 416, 416))
    y = net(x_image)
    print(y)

