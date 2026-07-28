import math
from xml.sax.expatreader import AttributesNSImpl

import torch

class Resnet(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Resnet, self).__init__()

        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        )

        self.res = None

        if dim_in != dim_out:
            # why dim of res need use padding=0 and kernel=1 depthwise kernel
            self.res = torch.nn.Conv2d(dim_in,
                                       dim_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        # x -> [1, 128, 10, 10]
        res = x
        if self.res is not None:
            res = self.res(x)

        return res + self.s(x)


class Atten(torch.nn.Module):

    def __init__(self, ):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=32,
                                       num_channels=512,
                                       eps=1e-6,
                                       affine=True)
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        # x -> [1, 512, 64, 64]
        res = x
        x = self.norm(x)
        # [1, 512, 64, 64] -> [1, 512, 4096] -> [1, 4096, 512]
        x = x.flatten(start_dim=2, end_dim=3).transpose(1, 2)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # [1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        k = k.transpose(1, 2)
        attn = torch.bmm(q, k) / math.sqrt(512)
        # print("Attn Score Shape", attn.shape)

        # 0.044194173824159216 = 1 / 512**0.5
        # atten = q.bmm(k) * 0.044194173824159216

        # 照理来说应该是等价的,但是却有很小的误差
        # atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
        #                       q,
        #                       k,
        #                       beta=0,
        #                       alpha=0.044194173824159216)

        # [1, 4096, 4096] -> [1, 4096]
        attn = torch.softmax(attn, dim=2)

        attn = torch.bmm(attn, v)

        attn = self.out(attn)

        attn = attn.transpose(1, 2).reshape(-1, 512, 64, 64)
        # print("Attn Output Shape", attn.shape)

        attn = attn + res

        return attn

class Pad(torch.nn.Module):
    def forward(self, x):
        # 作用是在最后两维上面填充
        # padding = (左, 右, 上, 下)
        #         = (0,  1,  0,  1)
        return torch.nn.functional.pad(x, (0, 1, 0, 1),
                                       mode='constant',
                                       value=0)

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            # in
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            # down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            # mid
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            # out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512,
                                   num_groups=32,
                                   eps=1e-6,
                                   ),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            # 正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = torch.nn.Sequential(
            # 正态分布层
            torch.nn.Conv2d(4, 4, 1),

            # in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            # middle
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            # up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode="nearest"),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),

            # out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128,
                                   num_groups=32,
                                   eps=1e-6,),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 3, 3, padding=1),
            ),
        )

    def sample(self, h):
        # h -> [1, 8, 64, 64]
        # 先将h分成两半,前4个通道是均值,后4个通道是方差
        mean, logvar = torch.chunk(h, 2, dim=1)
        std = logvar.exp() ** 2

        # 使用均值和方差采样
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + h * std

        return h

    def forward(self, x):
        # x -> [1, 3, 512, 512]

        # [1, 3, 512, 512] -> [1, 8, 64, 64]
        h = self.encoder(x)

        # [1, 8, 64, 64] -> [1, 4, 64, 64]
        h = self.sample(h)

        # [1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)

        return h


# print(Resnet(128, 256)(torch.randn(1, 128, 10, 10)).shape)
# print(Atten()(torch.randn(1, 512, 64, 64)).shape)
# print(Pad()(torch.ones(1, 2, 5, 5)))
# print(VAE()(torch.randn(1, 3, 512, 512)).shape)