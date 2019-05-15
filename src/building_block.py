import torch
import torch.nn as nn


def init_conv(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-2)


class LayerNormConv(nn.Module):

    def __init__(self, num_planes, plane_size):
        super(LayerNormConv, self).__init__()
        self.layer_norm = nn.LayerNorm([num_planes, plane_size, plane_size], elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(1, num_planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_planes, 1, 1))

    def forward(self, x):
        x = self.layer_norm(x)
        x = x * self.weight + self.bias
        return x


class EncoderConvLayer(nn.Module):

    def __init__(self, num_planes_in, num_planes_out, plane_size_out, downsample):
        super(EncoderConvLayer, self).__init__()
        self.conv = nn.Conv2d(num_planes_in, num_planes_out, kernel_size=3, padding=1, stride=(2 if downsample else 1))
        self.activation = nn.Sequential(
            LayerNormConv(num_planes_out, plane_size_out),
            nn.ReLU(inplace=True),
        )
        init_conv(self.modules())

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderConvLayer(nn.Module):

    def __init__(self, num_planes_in, num_planes_out, plane_size_out, upsample, is_last=False):
        super(DecoderConvLayer, self).__init__()
        self.upsample = None
        if upsample:
            self.upsample = lambda x: nn.functional.interpolate(x, scale_factor=2)
        self.conv = nn.Conv2d(num_planes_in, num_planes_out, kernel_size=3, padding=1)
        self.activation = None
        if not is_last:
            self.activation = nn.Sequential(
                LayerNormConv(num_planes_out, plane_size_out),
                nn.ReLU(inplace=True),
            )
        init_conv(self.modules())

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x


class FCLayer(nn.Module):

    def __init__(self, num_features_in, num_features_out):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(num_features_in, num_features_out)
        self.activation = nn.Sequential(
            nn.LayerNorm(num_features_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class EncoderConv(nn.Module):

    def __init__(self, config_channels, config_resample, num_planes_in, plane_size_in):
        super(EncoderConv, self).__init__()
        layers = []
        for num_planes_out, resample in zip(config_channels, config_resample):
            plane_size_out = plane_size_in // 2 if resample else plane_size_in
            layers.append(EncoderConvLayer(num_planes_in, num_planes_out, plane_size_out, downsample=resample))
            num_planes_in = num_planes_out
            plane_size_in = plane_size_out
        self.net = nn.Sequential(*layers)
        self.num_planes = num_planes_in
        self.plane_size = plane_size_in
        self.num_features = self.num_planes * self.plane_size * self.plane_size

    def forward(self, x):
        return self.net(x)


class DecoderConv(nn.Module):

    def __init__(self, config_channels, config_resample, num_planes_out, plane_size_out):
        super(DecoderConv, self).__init__()
        layers_rev = []
        for idx, (num_planes_in, resample) in enumerate(zip(config_channels, config_resample)):
            layers_rev.append(DecoderConvLayer(num_planes_in, num_planes_out, plane_size_out, upsample=resample,
                                               is_last=(idx == 0)))
            num_planes_out = num_planes_in
            plane_size_out = plane_size_out // 2 if resample else plane_size_out
        self.net = nn.Sequential(*reversed(layers_rev))
        self.num_planes = num_planes_out
        self.plane_size = plane_size_out
        self.num_features = self.num_planes * self.plane_size * self.plane_size

    def forward(self, x):
        return self.net(x)
