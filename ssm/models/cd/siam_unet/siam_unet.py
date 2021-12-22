import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ssm.models.utils import ConvBNReLU


class SiamUNet(nn.Layer):
    """
    The SiamUNet implementation based on PaddlePaddle.
    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): The channels of input.  Default: 3.
        align_corners (bool, optional): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear.  Default: False.
        siam (bool, optional): Use siam network or not.  Default: True.
        cat (bool, optional): Fuse features using concat, 
            if False, use add after siam or use log difference before encode.  Default: True.
    """
    def __init__(self,
                 num_classes,
                 in_channels=3,
                 align_corners=False,
                 use_deconv=False,
                 siam=True,
                 cat=True):
        super().__init__()
        if (not siam) and cat:
            self.encode = Encoder(2 * in_channels)
        else:
            self.encode = Encoder(in_channels)
            if siam:  # dimensionality reduction
                concat_channels = [128, 256, 512, 1024]
                self.concat_conv = nn.LayerList([
                    ConvBNReLU(cc, cc // 2, 1) for cc in concat_channels])
                self.concat_x_conv = ConvBNReLU(concat_channels[-1], concat_channels[-1] // 2, 1)
        self.decode = Decoder(align_corners, use_deconv=use_deconv)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.siam = siam
        self.cat = cat

    def encoder_forward(self, x1, x2):
        if self.siam:
            x1, short_cuts1 = self.encode(x1)
            x2, short_cuts2 = self.encode(x2)
            if self.cat:
                x = paddle.concat([x1, x2], axis=1)
                x = self.concat_x_conv(x)
                short_cuts = [self.concat_conv[i](paddle.concat([s1, s2], axis=1)) 
                              for i, (s1, s2) in enumerate(zip(short_cuts1, short_cuts2))]
            else:
                x = x1 + x2
                short_cuts = [(s1 + s2) for s1, s2 in zip(short_cuts1, short_cuts2)]
        else:
            if self.cat:
                x = paddle.concat([x1, x2], axis=1)
            else:
                x = paddle.log(x1) - paddle.log(x2)
            x, short_cuts = self.encode(x)
        return x, short_cuts

    def forward(self, x1, x2):
        logit_list = []
        x, short_cuts = self.encoder_forward(x1, x2)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list


class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3), ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(ConvBNReLU(in_channels, out_channels, 3))
        modules.append(ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()
        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1], align_corners, use_deconv)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False):
        super().__init__()
        self.align_corners = align_corners
        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels // 2,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels // 2
        else:
            in_channels *= 2
        self.double_conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, 3),
            ConvBNReLU(out_channels, out_channels, 3))

    def forward(self, x, short_cut):
        if self.use_deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x,
                paddle.shape(short_cut)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.double_conv(x)
        return x