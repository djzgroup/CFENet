import torch
import torch.nn as nn
import torch.nn.functional as F
from network.resnet import *

class Feature_Fusion_Module(nn.Module):
    def __init__(self):
        super(Feature_Fusion_Module,self).__init__()
        self.up_cbr2 = _ConvBnReLU(48, 48, 3, 1, 1, 1)
        self.up_cbr3 = _ConvBnReLU(48, 48, 3, 1, 1, 1)
        self.up_cbr4 = _ConvBnReLU(48, 48, 3, 1, 1, 1)
    def forward(self,x1,x2,x3,x4):
        u4 = self.up_cbr4(F.upsample(x4, x3.size()[2:],mode='bilinear', align_corners=True) + x3)
        u3 = self.up_cbr3(F.upsample(u4, x2.size()[2:],mode='bilinear', align_corners=True) + x2)
        u2 = self.up_cbr2(F.upsample(u3, x1.size()[2:],mode='bilinear', align_corners=True) + x1)
        u = torch.cat((F.upsample(x4, x1.size()[2:],mode='bilinear', align_corners=True),F.upsample(u4, x1.size()[2:],mode='bilinear', align_corners=True), \
                       F.upsample(u3, x1.size()[2:],mode='bilinear', align_corners=True),u2),dim = 1)
        
        return u

class Focus_Module(nn.Module):
    def __init__(self, in_channels, out_channels,stride = 1):
        super(Focus_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # SE layers
        self.fc1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1)
        self.DA = DARNetHead(out_channels, out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        out = self.DA(out)


        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
            解释  :
                bmm : 实现batch的叉乘
                Parameter：绑定在层里，所以是可以更新的
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DARNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DARNetHead, self).__init__()
        inter_channels = out_channels
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        sa_feat = self.conv5a(x)
        sa_feat = self.sa(sa_feat)
        sa_feat = self.conv51(sa_feat)

        sc_feat = self.conv5c(x)
        sc_feat = self.sc(sc_feat)
        sc_feat = self.conv52(sc_feat)

        # 两个注意力是相加的
        feat_sum = sa_feat + sc_feat

        output = self.dropout(feat_sum)
        return output

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

class Decoder_v3(nn.Module):
    def __init__(self, num_classes):
        super(Decoder_v3, self).__init__()

        self.conv1 = nn.Conv2d(48, 48, 1, bias=False)  #256->48 1*1
        self.bn1 = nn.BatchNorm2d(48) #nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(144, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder_v2(nn.Module):
    def __init__(self, num_classes):
        super(Decoder_v2, self).__init__()

        self.conv1 = nn.Conv2d(96, 48, 1, bias=False)  #256->48 1*1
        self.bn1 = nn.BatchNorm2d(48) #nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(144, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class ARF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(ARF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=4, dilation=4)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=8, dilation=8)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=16, dilation=16)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(96, 48, 1, bias=False)  #256->48 1*1
        self.bn1 = nn.BatchNorm2d(48) #nn.BatchNorm2d
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(144, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  #插值上采样
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            


class CLNet(nn.Module):
    def __init__(self, nclass = 1):
        super(CLNet, self).__init__()
        self.nclass = nclass
        self.resnet = resnet101(pretrained=True)
        self.skip_cbr1 = nn.Sequential(
            _ConvBnReLU(256, 256, 3, 2, 1, 1),
            ARF(256, 48),
            nn.Dropout(0.5)
        )
        self.skip_cbr2 = nn.Sequential(
            ARF(512, 48),
            nn.Dropout(0.5)
        )
        self.skip_cbr3 = nn.Sequential(
            RF(1024, 48),
            Focus_Module(48, 48),
            nn.Dropout(0.5)
        )
        self.skip_cbr4 = nn.Sequential(
            _ConvBnReLU(2048, 48, 1, 1, 0, 1),
            Focus_Module(48, 48),
            nn.Dropout(0.5)
        )
        self.fm = Focus_Module(96,48)
        self.out = Decoder_v3(self.nclass)

    def forward(self,x):
        layer0, layer1, layer2, layer3 = self.resnet(x)
        layer0 = self.skip_cbr1(layer0)
        layer1 = self.skip_cbr2(layer1)
        layer2 = self.skip_cbr3(layer2)
        layer3 = self.skip_cbr4(layer3)
        layer2_3 = torch.cat((layer3,layer2), dim=1)
        layer0_1 = torch.cat((layer0,layer1),dim=1)
        layer0_1_fm = self.fm(layer0_1)
        out = self.out(layer2_3,layer0_1_fm)

        return F.upsample(out, x.size()[2:],mode='bilinear', align_corners=True)







