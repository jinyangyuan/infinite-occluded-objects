import torch
import torch.nn as nn
import torch.nn.functional


def init_conv(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 1e-2)


class LayerNormConv(nn.Module):

    def __init__(self, num_planes):
        super(LayerNormConv, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_planes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_planes, 1, 1))

    def forward(self, x):
        weight = self.weight.expand(-1, *x.shape[-2:])
        bias = self.bias.expand(-1, *x.shape[-2:])
        return nn.functional.layer_norm(x, x.shape[-3:], weight, bias)


class Upsample(nn.Module):

    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)


class FC(nn.Module):

    def __init__(self, hidden_list, num_features_in, num_features_out, last_activation):
        super(FC, self).__init__()
        layers = []
        for num_features in hidden_list:
            layers = layers + self.create_layers(num_features_in, num_features)
            num_features_in = num_features
        layers = layers + self.create_layers(num_features_in, num_features_out, last_activation)
        self.net = nn.Sequential(*layers)

    @staticmethod
    def create_layers(num_features_in, num_features_out, use_activation=True):
        layers = [nn.Linear(num_features_in, num_features_out)]
        if use_activation:
            layers += [
                nn.LayerNorm(num_features_out),
                nn.ReLU(inplace=True),
            ]
        return layers

    def forward(self, x):
        x = self.net(x)
        return x


class EncoderConv(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, num_planes_in, plane_height_in,
                 plane_width_in, num_features_out, last_activation):
        super(EncoderConv, self).__init__()
        assert len(channel_list) == len(kernel_list)
        assert len(channel_list) == len(stride_list)
        layers = []
        for num_planes_out, kernel_size, stride in zip(channel_list, kernel_list, stride_list):
            assert plane_height_in % stride == 0
            assert plane_width_in % stride == 0
            layers = layers + self.create_layers(num_planes_in, num_planes_out, kernel_size, stride)
            num_planes_in = num_planes_out
            plane_height_in //= stride
            plane_width_in //= stride
        self.conv = nn.Sequential(*layers)
        self.fc = FC(
            hidden_list=hidden_list,
            num_features_in=num_planes_in * plane_height_in * plane_width_in,
            num_features_out=num_features_out,
            last_activation=last_activation,
        ).net
        init_conv(self.modules())

    @staticmethod
    def create_layers(num_planes_in, num_planes_out, kernel_size, stride):
        layers = [torch.nn.ZeroPad2d((0, 1, 0, 1))] if kernel_size % 2 == 0 else []
        padding = (kernel_size - 1) // 2
        layers += [
            nn.Conv2d(num_planes_in, num_planes_out, kernel_size, stride=stride, padding=padding),
            LayerNormConv(num_planes_out),
            nn.ReLU(inplace=True),
        ]
        return layers

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class DecoderConv(nn.Module):

    def __init__(self, channel_list_rev, kernel_list_rev, stride_list_rev, hidden_list_rev, num_features_in,
                 num_planes_out, plane_height_out, plane_width_out):
        super(DecoderConv, self).__init__()
        assert len(channel_list_rev) == len(kernel_list_rev)
        assert len(channel_list_rev) == len(stride_list_rev)
        layers = []
        use_activation = False
        for num_planes_in, kernel_size, stride in zip(channel_list_rev, kernel_list_rev, stride_list_rev):
            assert plane_height_out % stride == 0
            assert plane_width_out % stride == 0
            layers = self.create_layers(num_planes_in, num_planes_out, kernel_size, stride, use_activation) + layers
            num_planes_out = num_planes_in
            plane_height_out //= stride
            plane_width_out //= stride
            use_activation = True
        self.num_planes = num_planes_out
        self.plane_height = plane_height_out
        self.plane_width = plane_width_out
        self.fc = FC(
            hidden_list=reversed(hidden_list_rev),
            num_features_in=num_features_in,
            num_features_out=self.num_planes * self.plane_height * self.plane_width,
            last_activation=True,
        ).net
        self.conv = nn.Sequential(*layers)
        init_conv(self.modules())

    @staticmethod
    def create_layers(num_planes_in, num_planes_out, kernel_size, stride, use_activation):
        layers = [] if stride == 1 else [Upsample(stride)]
        if kernel_size % 2 == 0:
            layers.append(torch.nn.ZeroPad2d((0, 1, 0, 1)))
        padding = (kernel_size - 1) // 2
        layers.append(nn.Conv2d(num_planes_in, num_planes_out, kernel_size, padding=padding))
        if use_activation:
            layers += [
                LayerNormConv(num_planes_out),
                nn.ReLU(inplace=True),
            ]
        return layers

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.num_planes, self.plane_height, self.plane_width)
        x = self.conv(x)
        return x
