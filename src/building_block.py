import torch
import torch.nn as nn
import torch.nn.functional as nn_func


def get_linear_ln(in_features, out_features, activation=None):
    layers = [nn.Linear(in_features, out_features)]
    nonlinearity='linear'
    if activation is not None:
        layers += [
            nn.LayerNorm(out_features),
            activation(inplace=True),
        ]
        nonlinearity='relu'
    nn.init.kaiming_normal_(layers[0].weight, mode='fan_in', nonlinearity=nonlinearity)
    nn.init.zeros_(layers[0].bias)
    return layers


def get_enc_conv_ln(in_channels, out_channels, kernel_size, stride, activation=None):
    pad_1 = (kernel_size - 1) // 2
    pad_2 = kernel_size - 1 - pad_1
    layers = [
        nn.ZeroPad2d((pad_1, pad_2, pad_1, pad_2)),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
    ]
    nonlinearity = 'linear'
    if activation is not None:
        layers += [
            LayerNormConv(out_channels),
            activation(inplace=True),
        ]
        nonlinearity = 'relu'
    nn.init.kaiming_normal_(layers[1].weight, mode='fan_in', nonlinearity=nonlinearity)
    nn.init.zeros_(layers[1].bias)
    return layers


def get_dec_conv_ln(in_channels, out_channels, kernel_size, in_size, out_size, activation=None):
    layers = []
    if in_size != out_size:
        layers.append(Interpolate(out_size))
    layers += get_enc_conv_ln(in_channels, out_channels, kernel_size, stride=1, activation=activation)
    return layers


class LayerNormConv(nn.Module):

    def __init__(self, num_channels):
        super(LayerNormConv, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        weight = self.weight.expand(-1, *x.shape[-2:])
        bias = self.bias.expand(-1, *x.shape[-2:])
        return nn_func.layer_norm(x, x.shape[-3:], weight, bias)


class Interpolate(nn.Module):

    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.size = size

    def forward(self, x):
        return nn_func.interpolate(x, size=self.size)


class LinearBlock(nn.Sequential):

    def __init__(self, hidden_list, in_features, out_features, activation: nn.Module=nn.ReLU):
        layers = []
        for num_features in hidden_list:
            layers += get_linear_ln(in_features, num_features, activation=activation)
            in_features = num_features
        self.out_features = in_features
        if out_features is not None:
            layers += get_linear_ln(in_features, out_features)
            self.out_features = out_features
        super(LinearBlock, self).__init__(*layers)


class EncoderBlock(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, in_shape, out_features,
                 activation: nn.Module=nn.ReLU):
        super(EncoderBlock, self).__init__()
        assert len(channel_list) == len(kernel_list)
        assert len(channel_list) == len(stride_list)
        layers = []
        in_channels, in_height, in_width = in_shape
        for num_channels, kernel_size, stride in zip(channel_list, kernel_list, stride_list):
            layers += get_enc_conv_ln(in_channels, num_channels, kernel_size, stride, activation=activation)
            in_channels = num_channels
            in_height = (in_height - 1) // stride + 1
            in_width = (in_width - 1) // stride + 1
        self.conv = nn.Sequential(*layers)
        self.linear = LinearBlock(
            hidden_list=hidden_list,
            in_features=in_channels * in_height * in_width,
            out_features=out_features,
            activation=activation,
        )
        self.out_features = self.linear.out_features

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, channel_list_rev, kernel_list_rev, stride_list_rev, hidden_list_rev, in_features, out_shape,
                 activation: nn.Module=nn.ReLU):
        super(DecoderBlock, self).__init__()
        assert len(channel_list_rev) == len(kernel_list_rev)
        assert len(channel_list_rev) == len(stride_list_rev)
        layers = []
        out_channels, out_height, out_width = out_shape
        layer_act = None
        for num_channels, kernel_size, stride in zip(channel_list_rev, kernel_list_rev, stride_list_rev):
            in_height = (out_height - 1) // stride + 1
            in_width = (out_width - 1) // stride + 1
            sub_layers = get_dec_conv_ln(num_channels, out_channels, kernel_size, in_size=[in_height, in_width],
                                         out_size=[out_height, out_width], activation=layer_act)
            layers = sub_layers + layers
            out_channels = num_channels
            out_height = in_height
            out_width = in_width
            layer_act = activation
        self.conv = nn.Sequential(*layers)
        if layer_act is None:
            self.linear = LinearBlock(
                hidden_list=[*reversed(hidden_list_rev)],
                in_features=in_features,
                out_features=out_channels * out_height * out_width,
                activation=activation,
            )
        else:
            self.linear = LinearBlock(
                hidden_list=[*reversed(hidden_list_rev)] + [out_channels * out_height * out_width],
                in_features=in_features,
                out_features=None,
                activation=activation,
            )
        self.in_shape = [out_channels, out_height, out_width]

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], *self.in_shape)
        x = self.conv(x)
        return x
