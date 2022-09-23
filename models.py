import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('in', x.shape)
        # print('out', self.model(x).shape)
        return self.model(x)


class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetMid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)


    def forward(self, x, skip_input):
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        x =  nn.functional.pad(x, (1,0,1,0,1,0))

        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print('new')
        # print(x.shape)
        # print(skip_input.shape)
        x = self.model(x)
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)

        return x


class LFFB(nn.Module):
    def __init__(self, v_size, dropout=0.0):
        super(LFFB, self).__init__()
        layers3x3 = [
            nn.Conv3d(v_size, v_size, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(v_size, v_size, kernel_size=3, padding=1, bias=False),
        ]
        layers5x5 = [
            nn.Conv3d(v_size, v_size, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(v_size, v_size, kernel_size=5, padding=2, bias=False),
        ]
        layers7x7 = [
            nn.Conv3d(v_size, v_size, kernel_size=7, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(v_size, v_size, kernel_size=7, padding=3, bias=False),
        ]
        if dropout:
            layers3x3.append(nn.Dropout(dropout))
            layers5x5.append(nn.Dropout(dropout))
            layers7x7.append(nn.Dropout(dropout))

        self.model3x3 = nn.Sequential(*layers3x3)
        self.model5x5 = nn.Sequential(*layers5x5)
        self.model7x7 = nn.Sequential(*layers7x7)

        self.conv1x1 = nn.Conv3d(v_size * 2, v_size, kernel_size=1, bias=False)

    def forward(self, x):

        # print(x.shape)

        x_3x3 = self.model3x3(x)
        # print(x_3x3.shape)
        x_3x3 = torch.cat((x_3x3, x), dim=1)
        # print(x_3x3.shape)
        x_3x3 = self.conv1x1(x_3x3)

        x_5x5 = self.model5x5(x)
        # print(x_5x5.shape)
        x_5x5 = torch.cat((x_5x5, x_3x3), dim=1)
        # print(x_3x3.shape)
        x_5x5 = self.conv1x1(x_5x5)

        x_7x7 = self.model7x7(x)
        # print(x_7x7.shape)
        x_7x7 = torch.cat((x_7x7, x_5x5), dim=1)
        # print(x_7x7.shape)
        x_7x7 = self.conv1x1(x_7x7)

        x = x + x_7x7

        return x


class GFFB(nn.Module):
    def __init__(self, v_size, reduction=1):
        super(GFFB, self).__init__()

        self.cam_gap = nn.AdaptiveAvgPool3d(1)
        self.cam_FC1 = nn.Linear(v_size, v_size // reduction)
        self.cam_relu = nn.ReLU(inplace=True)
        self.cam_FC2 = nn.Linear(v_size // reduction, v_size)
        self.cam_sigmoid = nn.Sigmoid()


        self.f_conv1x1 = nn.Conv3d(v_size, v_size // 8, kernel_size=1, bias=False)
        self.g_conv1x1 = nn.Conv3d(v_size, v_size // 8, kernel_size=1, bias=False)
        self.h_conv1x1 = nn.Conv3d(v_size, v_size, kernel_size=1, bias=False)
        self.final_conv1x1 = nn.Conv3d(v_size, v_size, kernel_size=1, bias=False)

    def forward(self, x):

        # Channel attention module
        # print(x.shape)
        x_cam = self.cam_gap(x)
        x_cam_shape = x_cam.shape
        # print(x_cam.shape)
        x_cam = x_cam.view(-1)
        # print(x_cam.shape)
        x_cam = self.cam_FC1(x_cam)
        # print(x_cam.shape)
        x_cam = self.cam_relu(x_cam)
        x_cam = self.cam_FC2(x_cam)
        x_cam = self.cam_sigmoid(x_cam)
        x_cam = x_cam.reshape(x_cam_shape)
        x_cam = x_cam.expand(x.shape)

        # Convolution feature maps
        f_x = self.f_conv1x1(x)
        g_x = self.g_conv1x1(x)
        h_x = self.h_conv1x1(x)
        print(f_x.shape)
        print(g_x.shape)
        print(h_x.shape)

        f_x = f_x.view([f_x.size(1), f_x.size(2) * f_x.size(3) * f_x.size(4)])
        g_x = g_x.view([g_x.size(2) * g_x.size(3) * g_x.size(4), g_x.size(1)])
        h_x = h_x.view([h_x.size(2) * h_x.size(3) * h_x.size(4), h_x.size(1)])
        print(f_x.shape)
        print(g_x.shape)
        print(h_x.shape)

        attr_map = torch.matmul(g_x, f_x)
        print(attr_map.shape)

        res = attr_map * h_x
        print(res.shape)

        return x_cam


class MSRB(nn.Module):
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Conv3d(n_feats, n_feats, kernel_size_1, padding=(kernel_size_1//2), bias=True)
        self.conv_3_2 = nn.Conv3d(n_feats * 2, n_feats * 2, kernel_size_1, padding=(kernel_size_1//2), bias=True)
        self.conv_5_1 = nn.Conv3d(n_feats, n_feats, kernel_size_2, padding=(kernel_size_2//2), bias=True)
        self.conv_5_2 = nn.Conv3d(n_feats * 2, n_feats * 2, kernel_size_2, padding=(kernel_size_2//2), bias=True)
        self.confusion = nn.Conv3d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        input_1 = x
        # print(input_1.shape)
        output_3_1 = self.relu(self.conv_3_1(input_1))
        # print(output_3_1.shape)
        output_5_1 = self.relu(self.conv_5_1(input_1))
        # print(output_5_1.shape)
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        # print(output_5_1.shape)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.msrb = MSRB(64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.mid1 = UNetMid(1024, 512, dropout=0.2)
        self.mid2 = UNetMid(1024, 512, dropout=0.2)
        self.mid3 = UNetMid(1024, 512, dropout=0.2)
        '''
        self.mid3_1 = UNetMid(1024, 512, dropout=0.2)
        self.mid3_2 = UNetMid(1024, 512, dropout=0.2)
        self.mid3_3 = UNetMid(1024, 512, dropout=0.2)
        self.mid3_4 = UNetMid(1024, 512, dropout=0.2)
        self.mid3_5 = UNetMid(1024, 512, dropout=0.2)
        self.mid3_6 = UNetMid(1024, 512, dropout=0.2)
        '''
        self.mid4 = UNetMid(1024, 256, dropout=0.2)
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        # self.us =   nn.Upsample(scale_factor=2)

        self.lffb = LFFB(128)

        self.final = nn.Sequential(
            # nn.Conv3d(128, out_channels, 4, padding=1),
            # nn.Tanh(),
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        # print(d1.shape)
        d1_l = self.msrb(d1)
        # print(d1_l.shape)
        d2 = self.down2(d1_l)
        # print(d2.shape)
        d3 = self.down3(d2)
        # print(d3.shape)
        d4 = self.down4(d3)
        # print(d4.shape)
        m1 = self.mid1(d4, d4)
        # print(m1.shape)
        m2 = self.mid2(m1, m1)
        # print(m2.shape)
        m3 = self.mid3(m2, m2)
        # print(m3.shape)
        # m3_1 = self.mid3_1(m3, m3)
        # m3_2 = self.mid3_2(m3_1, m3_1)
        # m3_3 = self.mid3_3(m3_2, m3_2)
        # m3_4 = self.mid3_4(m3_3, m3_3)
        # m3_5 = self.mid3_5(m3_4, m3_4)
        # m3_6 = self.mid3_6(m3_5, m3_5)
        # m4 = self.mid4(m3_4, m3_4)
        m4 = self.mid4(m3, m3)
        # print(m4.shape)
        u1 = self.up1(m4, d3)
        # print(u1.shape)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        u3_g = self.lffb(u3)

        return self.final(u3_g)


class GeneratorUNet_2048(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet_2048, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 1024)
        self.mid1 = UNetMid(2048, 1024, dropout=0.2)
        self.mid2 = UNetMid(2048, 1024, dropout=0.2)
        self.mid3 = UNetMid(2048, 1024, dropout=0.2)
        self.mid3_1 = UNetMid(2048, 1024, dropout=0.2)
        self.mid3_2 = UNetMid(2048, 1024, dropout=0.2)
        self.mid3_3 = UNetMid(2048, 1024, dropout=0.2)
        self.mid3_4 = UNetMid(2048, 1024, dropout=0.2)
        self.mid4 = UNetMid(2048, 512, dropout=0.2)
        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)
        # self.us =   nn.Upsample(scale_factor=2)

        self.final = nn.Sequential(
            # nn.Conv3d(128, out_channels, 4, padding=1),
            # nn.Tanh(),
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        # print(d1.shape)
        d2 = self.down2(d1)
        # print(d2.shape)
        d3 = self.down3(d2)
        # print(d3.shape)
        d4 = self.down4(d3)
        # print(d4.shape)
        d5 = self.down5(d4)
        # print(d5.shape)
        m1 = self.mid1(d5, d5)
        # print(m1.shape)
        m2 = self.mid2(m1, m1)
        # print(m2.shape)
        m3 = self.mid3(m2, m2)
        # print(m3.shape)
        m3_1 = self.mid3_1(m3, m3)
        # print(m3_1.shape)
        m3_2 = self.mid3_2(m3_1, m3_1)
        # print(m3_2.shape)
        m3_3 = self.mid3_3(m3_2, m3_2)
        # print(m3_3.shape)
        m3_4 = self.mid3_4(m3_3, m3_3)
        # print(m3_4.shape)
        m4 = self.mid4(m3_4, m3_4)
        # print(m4.shape)
        u1 = self.up1(m4, d4)
        # print(u1.shape)
        u2 = self.up2(u1, d3)
        # print(u2.shape)
        u3 = self.up3(u2, d2)
        # print(u3.shape)
        u4 = self.up4(u3, d1)
        # print(u4.shape)
        return self.final(u4)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(512, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        intermediate = self.model(img_input)
        pad = nn.functional.pad(intermediate, pad=(1,0,1,0,1,0))
        return self.final(pad)
