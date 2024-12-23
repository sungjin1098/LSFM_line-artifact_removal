import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


'''
    (previous version explanation)
    To obtain mask S: separated encoder of UNet 3+, the latent feature is a vector + sigmoid at the last
                      decoder is Bilinear upsampling to extend the vector + sigmoid at the last
                      discriminator is also used (Purpose: discriminator should not distinguish our generated S and synthetic S.)
    To obtain output X: separated encoder of UNet 3+, the latent feature is 3D, and the decoder of UNet 3+
                        discriminator is also used (Purpose: discriminator should not distinguish X and rotated X.)
                        
    [UNet v13]
    UNet 3+
    To obtain mask S: separated encoder of UNet 3+, the latent feature is a vector + sigmoid at the last
                      decoder is Bilinear upsampling to extend the vector
                      discriminator is also used (Purpose: discriminator should not distinguish our generated S and synthetic S.)
                      [upsample]-[conv]-[bn]-[relu]-[conv] at the end of the decoder.
                      * UNetv8-2: channels {64, 128, 64, 32, 1} / latent bx1x1x256
                      * UNetv8-3: channels {64, 128, 64, 32, 1} / latent bx1x16x256
    To obtain output X: separated encoder of UNet 3+, the latent feature is 3D, and the decoder of UNet 3+
                        * ReLU exists at the end of the UNet in both training and testing.
                        --> Batch normalization option is added.
'''

### 다른건 모두 UNetv15와 같은데, S와 X에서 batch normalization 제거함

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class MLP(nn.Module):
    def __init__(self, nz, patch):
        super(MLP, self).__init__()
        self.fc = nn.Linear(nz, patch*patch)

    def forward(self, x):
        x = self.fc(x)
        return x


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1, relu=True):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm and relu:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     # nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        elif is_batchnorm and not relu:
            for i in range(1, n + 1):
                if i == n:
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), )
                                         # nn.BatchNorm2d(out_size), )
                else:
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                         # nn.BatchNorm2d(out_size),
                                         nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        elif not is_batchnorm and relu:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        elif not is_batchnorm and not relu:
            for i in range(1, n+1):
                if i == n:
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
                else:
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                         nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.activation:
            x = self.activation(x)
        return x



### UNet 3+ to obtain line artifact mask S
class UNet_3Plus_for_S(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_scale=4, is_batchnorm=True):
        super(UNet_3Plus_for_S, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm
        self.kernel = feature_scale

        filters = [64, 128, 64, 32, 1]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm, relu=False)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

        ## -------------Decoder--------------
        self.d5_conv = unetConv2(filters[4], filters[3], self.is_batchnorm)

        self.d5tod4 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d5tod4_conv = unetConv2(filters[3], filters[2], self.is_batchnorm)

        self.d4tod3 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d4tod3_conv = unetConv2(filters[2], filters[1], self.is_batchnorm)

        self.d3tod2 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d3tod2_conv = unetConv2(filters[1], filters[0], self.is_batchnorm)

        self.d2tod1 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d2tod1_conv = unetConv2(filters[0], self.out_channels, self.is_batchnorm, relu=False)

        self.conv11 = nn.Conv2d(filters[0], self.out_channels, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(self.out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.bn22 = nn.BatchNorm2d(self.out_channels)
        # self.d2tod1_conv = unetConv2(filters[0], self.out_channels, self.is_batchnorm, relu=False)

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        # hd5 = F.sigmoid(self.conv5(h5))  # h5->20*20*1024
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        new_d5 = self.d5_conv(hd5)
        new_d4 = self.d5tod4_conv(self.d5tod4(new_d5))
        new_d3 = self.d4tod3_conv(self.d4tod3(new_d4))
        new_d2 = self.d3tod2_conv(self.d3tod2(new_d3))
        # new_d1 = self.conv22(self.relu11(self.bn11(self.conv11(self.d2tod1(new_d2)))))
        new_d1 = self.conv22(self.relu11(self.conv11(self.d2tod1(new_d2))))
        new_d1 = F.sigmoid(new_d1)

        return hd5, new_d1   # return the latent vector of line artifact mask S



### UNet 3+ to obtain cleaned image X
class UNet_3Plus_for_X(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_scale=4, is_batchnorm=True):
        super(UNet_3Plus_for_X, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, out_channels, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        # h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        # h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        # h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        # h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        # hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        # hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
        #     torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels
        #
        # h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        # h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        # h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        # hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        # hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        # hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
        #     torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels
        #
        # h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        # h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        # hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        # hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        # hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        # hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
        #     torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels
        #
        # h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        # hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        # hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        # hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        # hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        # hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
        #     torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_conv(h4))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        hd4 = self.relu4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_conv(h3))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        hd3 = self.relu3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_conv(h2))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        hd2 = self.relu2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_conv(h1))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        hd1 = self.relu1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*out_channels
        d1 = self.relu1d_1(d1)

        return d1 # return cleaned image X


class STN(nn.Module):
    def __init__(self, data_shape, device):
        super(STN, self).__init__()
        self.b, self.w, self.h = data_shape
        self.device = device
        # self.theta = nn.Parameter(torch.zeros(1, dtype=torch.float, device=self.device).requires_grad_(True))

        # localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=11),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(2, 4, kernel_size=9),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10*10*10, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10*10*10)
        self.theta = torch.mean(self.fc_loc(xs))   # the role of 'mean': self.theta.shape (b, 1) --> (1)

        theta = torch.zeros((self.b, 2, 3), dtype=torch.float, device=self.device)
        theta[:, 0, 0] = torch.cos(self.theta)
        theta[:, 1, 1] = torch.cos(self.theta)
        theta[:, 0, 1] = -torch.sin(self.theta)
        theta[:, 1, 0] = torch.sin(self.theta)

        x_reshape = x.view(-1, 1, self.w, self.h)        # (b, 1, w, h) except last batch
        theta = theta[:x_reshape.shape[0]]               # for last batch
        grid = F.affine_grid(theta, x_reshape.size())
        # mask after STN = x_reg
        x_reg = F.grid_sample(x_reshape, grid).view(-1, 1, self.w, self.h)    # (b, inp) except last batch

        return x_reg, self.theta
        

class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, n_layer, dim, activ, pad, num_scales):
        super(Discriminator, self).__init__()
        self.n_layer = n_layer                # the number of conv layers
        self.dim = dim                        # if dim=16, then # of channels become 1 -> 16 -> 32 -> ...
        self.activ = activ                    # {relu, lrelu, prelu, selu, tanh, none} are possible
        self.pad_type = pad                   # {reflect, replicate, zero} are possible
        self.input_dim = input_dim
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())


    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, kernel_size=4, stride=2, padding=1, activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, kernel_size=4, stride=2, padding=1, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        for model in self.cnns:
            x = model(x)
            x = self.downsample(x)
        return F.sigmoid(x)

class Discriminator_revised_previous(nn.Module):
    def __init__(self, input_dim, dim):
        super(Discriminator_revised_previous, self).__init__()
        self.main = nn.Sequential(
            # input is ``(input_dim) x 256 x 256``
            nn.Conv2d(input_dim, dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dim) x 128 x 128``
            nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dim*2) x 64 x 64``
            nn.AvgPool2d(2, stride=2, padding=[0, 0], count_include_pad=False),
            # state size. ``(dim*2) x 32 x 32``
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dim*4) x 16 x 16``
            nn.AvgPool2d(2, stride=2, padding=[0, 0], count_include_pad=False),
            # state size. ``(dim*4) x 8 x 8``
            nn.Conv2d(dim * 4, dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dim*8) x 4 x 4``
            nn.AvgPool2d(2, stride=2, padding=[0, 0], count_include_pad=False),
            # state size. ``(dim*8) x 2 x 2``
            nn.Conv2d(dim * 8, 1, 1, 1, 0, bias=False),
            # state size. ``1 x 2 x 2``
            nn.AvgPool2d(2, stride=2, padding=[0, 0], count_include_pad=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    
class Discriminator_gpt(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator_gpt, self).__init__()
        self.main = nn.Sequential(
            # Input: [batch size, 1, 256, 256]
            # First layer - common feature extraction
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            # Second layer - enhanced feature extraction with larger field of view
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce to 128x128

            # Direction-specific layers
            # Vertical feature detector
            nn.Conv2d(128, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            # Horizontal feature detector
            nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            # Pooling to merge features and reduce dimensions
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce to 64x64

            # Additional feature consolidation
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Reduce to 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            # Final feature reduction to classify
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Reduce to 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512),

            nn.AdaptiveAvgPool2d(1),  # Global average pooling to 1x1
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

    
class Discriminator_revised1(nn.Module):
    def __init__(self, input_dim, dim):
        super(Discriminator_revised1, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim, 7, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(dim, dim * 2, 7, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(dim*2, dim * 2, 7, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(dim * 2, dim * 4,7, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 4, dim * 8, 7, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(dim * 8, dim * 16, 7, 3, 1, bias=False),
            nn.BatchNorm2d(dim * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dim * 16, 1, 1, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator_revised(nn.Module):
    def __init__(self, input_dim, dim):
        super(Discriminator_revised, self).__init__()
        self.main = nn.Sequential(
            # input is ``(input_dim) x 256 x 256``
            nn.Conv2d(input_dim, dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dim) x 256 x 256``
            nn.Conv2d(dim, dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, stride=4, padding=[0, 0], count_include_pad=False),
            # state size. ``(dim * 2) x 64 x 64``
            nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, stride=4, padding=[0, 0], count_include_pad=False),
            # state size. ``(dim * 4) x 16 x 16``
            nn.Conv2d(dim * 4, dim * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, stride=4, padding=[0, 0], count_include_pad=False),
            # state size. ``(dim * 8) x 4 x 4``
            nn.Conv2d(dim * 8, 1, 1, 1, 0, bias=False),
            # state size. ``1 x 4 x 4``
            nn.AvgPool2d(2, stride=2, padding=[0, 0], count_include_pad=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class DiscriminatorWithSkip(nn.Module):
    def __init__(self, input_dim, dim):
        super(DiscriminatorWithSkip, self).__init__()

        # Initial Convolution Block
        self.conv1 = nn.Conv2d(input_dim, dim, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # Convolutional Blocks
        self.conv2 = nn.Conv2d(dim, 2*dim, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d( 2*dim)

        self.conv3 = nn.Conv2d(2*dim, 4*dim, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d( 4*dim)

        self.conv4 = nn.Conv2d(4*dim, 8*dim, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d( 8*dim)

        # Skip Connections
        self.skip_conv1 = nn.Conv2d(dim,  8*dim, kernel_size=4, stride=8, padding=1)
        self.skip_conv2 = nn.Conv2d( 2*dim, 8*dim, kernel_size=4, stride=4, padding=1)

        # Final Convolution
        self.final_conv = nn.Conv2d( 8*dim, 1, kernel_size=4, stride=1, padding=1)

        self.avg_pool =     nn.AvgPool2d(2, stride=2, padding=[0, 0], count_include_pad=False),
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        # Initial Convolution Block
        x1 = self.leaky_relu(self.conv1(x))

        # Convolutional Blocks
        x2 = self.leaky_relu(self.bn2(self.conv2(x1)))
        x3 = self.leaky_relu(self.bn3(self.conv3(x2)))
        x4 = self.leaky_relu(self.bn4(self.conv4(x3)))

        # Adding Skip Connections
        skip1 = self.skip_conv1(x1)
        skip2 = self.skip_conv2(x2)
        x4 = x4 + skip1 + skip2

        # Final Convolution
        out = self.final_conv(x4)

        out = self.global_pool(out)

        return torch.sigmoid(out)

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)
    
class Pix2Pix_Discriminator(nn.Module):
    def __init__(self, in_channels,out_channels,nker=64,norm='bnorm'):
        super(Pix2Pix_Discriminator,self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4,stride=2,
                          padding=1,norm=None,relu=0.2,bias=False)   # 첫번째 D layer에는 batch 적용 X 
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc3 = CBR2d(2*nker, 4 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc4 = CBR2d(4 * nker, 6 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc4_1 = CBR2d(6* nker, 8 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc4_2 = CBR2d(8* nker, 10 * nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc4_3 = CBR2d(10* nker, 12* nker, kernel_size=4,stride=2,
                          padding=1,norm=norm,relu=0.2,bias=False)
        self.enc5 = CBR2d(12* nker, out_channels, kernel_size=4,stride=2,
                          padding=1,norm=None,relu=None,bias=False)

    def forward(self,x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc4_1(x)
        x = self.enc4_2(x)
        x = self.enc4_3(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x
from torch.nn.utils import spectral_norm


    
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
    
    
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='relu'):
    layers = []
    
    # Conv layer
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    
    # Batch Normalization
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    
    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass
    
    return nn.Sequential(*layers)

class Discriminator_patch(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator_patch, self).__init__()
        self.conv1 = conv(1, 64, 4, bn=False, activation='lrelu')
        self.conv2 = conv(64, 128, 4, activation='lrelu')
        self.conv3 = conv(128, 256, 4, activation='lrelu')
        self.conv4 = conv(256, 512, 4, 1, 1, activation='lrelu')
        self.conv5 = conv(512, 1, 4, 1, 1, activation='none')

    # forward method
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = torch.sigmoid(out)

        return out




### UNet 3+ to obtain cleaned image X
class UNet_for_D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_scale=4, is_batchnorm=True):
        super(UNet_for_D, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [32, 64, 128, 256, 512]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.outconv1 = nn.Conv2d(self.UpChannels, out_channels, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024


        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1)))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2)))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3)))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_conv(h4))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5)))
        hd4 = self.relu4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_conv(h3))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5)))
        hd3 = self.relu3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_conv(h2))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5)))
        hd2 = self.relu2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_conv(h1))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5)))
        hd1 = self.relu1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))  # hd1->320*320*UpChannels

        d1 = self.outconv1(hd1)  # d1->320*320*out_channels
        d1 = torch.sigmoid(d1)

        return d1 # return cleaned image X

    





### UNet 3+ to obtain line artifact mask S
class UNet_for_DD(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_scale=3, is_batchnorm=True):
        super(UNet_for_DD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm
        self.kernel = 2

        filters = [64, 128, 64, 32, 1]

        ## -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(self.kernel, 1))

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(4, 1))


        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm, relu=False)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

        ## -------------Decoder--------------
        self.d5_conv = unetConv2(filters[4], filters[3], self.is_batchnorm)

        self.d5tod4 = nn.Upsample(scale_factor=(4,1), mode='bilinear')
        self.d5tod4_conv = unetConv2(filters[3], filters[2], self.is_batchnorm)

        self.d4tod3 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d4tod3_conv = unetConv2(filters[2], filters[1], self.is_batchnorm)

        self.d3tod2 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d3tod2_conv = unetConv2(filters[1], filters[0], self.is_batchnorm)

        self.d2tod1 = nn.Upsample(scale_factor=(self.kernel,1), mode='bilinear')
        self.d2tod1_conv = unetConv2(filters[0], self.out_channels, self.is_batchnorm, relu=False)

        self.conv11 = nn.Conv2d(filters[0], self.out_channels, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(self.out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.bn22 = nn.BatchNorm2d(self.out_channels)
        # self.d2tod1_conv = unetConv2(filters[0], self.out_channels, self.is_batchnorm, relu=False)

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        # hd5 = F.sigmoid(self.conv5(h5))  # h5->20*20*1024
        hd5 = self.conv5(h5)  # h5->20*20*1024
        ## -------------Decoder-------------
        new_d5 = self.d5_conv(hd5)
        new_d4 = self.d5tod4_conv(self.d5tod4(new_d5))
        new_d3 = self.d4tod3_conv(self.d4tod3(new_d4))
        new_d2 = self.d3tod2_conv(self.d3tod2(new_d3))
        
        #new_d1 = self.conv22(self.relu11(self.bn11(self.conv11(self.d2tod1(new_d2)))))
        new_d11 = self.conv22(self.relu11(self.conv11(self.d2tod1(new_d2))))

        new_d1 = torch.sigmoid(new_d11)

        return  new_d1 # return the latent vector of line artifact mask S





