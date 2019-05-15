import torch
import torch.nn as nn
from building_block import EncoderConv, DecoderConv, FCLayer


def normalize_image(x):
    return x * 2 - 1


def restore_image(x):
    return (x + 1) * 0.5


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class UpdaterBase(nn.Module):

    def __init__(self, num_planes_in, plane_size_in, config_channels, config_resample, config_hidden, state_size):
        super(UpdaterBase, self).__init__()
        self.conv = EncoderConv(config_channels, config_resample, num_planes_in, plane_size_in)
        layers = []
        num_features_in = self.conv.num_features
        for num_features_out in config_hidden:
            layers.append(FCLayer(num_features_in, num_features_out))
            num_features_in = num_features_out
        self.fc = nn.Sequential(*layers)
        self.lstm = nn.LSTMCell(num_features_in, state_size)

    def forward(self, x, states=None):
        x = normalize_image(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        states = self.lstm(x, states)
        return states


class VAEEncoder(nn.Module):

    def __init__(self, num_features_in, config_hidden, vae_size, vae_groups):
        super(VAEEncoder, self).__init__()
        self.vae_size = vae_size
        layers = []
        for num_features_out in config_hidden:
            layers.append(FCLayer(num_features_in, num_features_out))
            num_features_in = num_features_out
        layers.append(nn.Linear(num_features_in, vae_size * vae_groups))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x).split(self.vae_size, dim=-1)


class VAEDecoderSimple(nn.Module):

    def __init__(self, num_planes_out, plane_size_out, config_hidden, vae_size):
        super(VAEDecoderSimple, self).__init__()
        layers = []
        num_features_in = vae_size
        for num_features_out in config_hidden:
            layers.append(FCLayer(num_features_in, num_features_out))
            num_features_in = num_features_out
        layers.append(nn.Linear(num_features_in, num_planes_out))
        self.fc = nn.Sequential(*layers)
        self.plane_size_out = plane_size_out

    def forward(self, x):
        x = self.fc(x)
        x = x[..., None, None].expand(-1, -1, self.plane_size_out, self.plane_size_out)
        return x


class VAEDecoderComplex(nn.Module):

    def __init__(self, num_planes_out, plane_size_out, config_channels, config_resample, config_hidden, vae_size):
        super(VAEDecoderComplex, self).__init__()
        conv = DecoderConv(config_channels, config_resample, num_planes_out, plane_size_out)
        layers = []
        num_features_in = vae_size
        for num_features_out in config_hidden:
            layers.append(FCLayer(num_features_in, num_features_out))
            num_features_in = num_features_out
        layers.append(FCLayer(num_features_in, conv.num_features))
        self.fc = nn.Sequential(*layers)
        self.conv = conv

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], self.conv.num_planes, self.conv.plane_size, self.conv.plane_size)
        x = self.conv(x)
        return x


class InitializerWhere(nn.Module):

    def __init__(self, args):
        super(InitializerWhere, self).__init__()
        num_planes_in = args.image_channels * 2
        plane_size_in = args.image_full_size
        config_channels = args.init_where_channels
        config_resample = args.init_where_resample
        config_hidden_1 = args.init_where_hidden_1
        config_hidden_2 = args.init_where_hidden_2
        state_size_base = args.state_size_base
        state_size_where = args.state_size_where
        self.conv = EncoderConv(config_channels, config_resample, num_planes_in, plane_size_in)
        layers = []
        num_features_in = self.conv.num_features
        for num_features_out in config_hidden_1:
            layers.append(FCLayer(num_features_in, num_features_out))
            num_features_in = num_features_out
        self.fc_1 = nn.Sequential(*layers)
        self.lstm_base = nn.LSTMCell(num_features_in, state_size_base)
        layers = []
        num_features_in = state_size_base
        for num_features_out in config_hidden_2:
            layers.append(FCLayer(num_features_in, num_features_out))
            num_features_in = num_features_out
        self.fc_2 = nn.Sequential(*layers)
        self.lstm_where = nn.LSTMCell(num_features_in, state_size_where)

    def forward(self, x, states_base):
        x = normalize_image(x)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        states_base = self.lstm_base(x, states_base)
        x = self.fc_2(states_base[0])
        states_where = self.lstm_where(x)
        return states_where, states_base


class InitializerWhat(UpdaterBase):

    def __init__(self, args):
        super(InitializerWhat, self).__init__(
            num_planes_in=args.image_channels * 2,
            plane_size_in=args.image_crop_size,
            config_channels=args.init_what_channels,
            config_resample=args.init_what_resample,
            config_hidden=args.init_what_hidden,
            state_size=args.state_size_what,
        )


class InitializerBack(UpdaterBase):

    def __init__(self, args):
        super(InitializerBack, self).__init__(
            num_planes_in=args.image_channels,
            plane_size_in=args.image_full_size,
            config_channels=args.init_back_channels,
            config_resample=args.init_back_resample,
            config_hidden=args.init_back_hidden,
            state_size=args.state_size_back,
        )


class UpdaterWhere(UpdaterBase):

    def __init__(self, args):
        super(UpdaterWhere, self).__init__(
            num_planes_in=args.image_channels * 4,
            plane_size_in=args.image_full_size,
            config_channels=args.upd_where_channels,
            config_resample=args.upd_where_resample,
            config_hidden=args.upd_where_hidden,
            state_size=args.state_size_where,
        )


class UpdaterWhat(UpdaterBase):

    def __init__(self, args):
        super(UpdaterWhat, self).__init__(
            num_planes_in=args.image_channels * 4,
            plane_size_in=args.image_crop_size,
            config_channels=args.upd_what_channels,
            config_resample=args.upd_what_resample,
            config_hidden=args.upd_what_hidden,
            state_size=args.state_size_what,
        )


class UpdaterBack(UpdaterBase):

    def __init__(self, args):
        super(UpdaterBack, self).__init__(
            num_planes_in=args.image_channels * 3,
            plane_size_in=args.image_full_size,
            config_channels=args.upd_back_channels,
            config_resample=args.upd_back_resample,
            config_hidden=args.upd_back_hidden,
            state_size=args.state_size_back,
        )


class VAEPres(nn.Module):

    def __init__(self, args):
        super(VAEPres, self).__init__()
        self.enc = VAEEncoder(
            num_features_in=args.state_size_where + args.state_size_what,
            config_hidden=args.enc_pres_hidden,
            vae_size=1,
            vae_groups=3,
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        logits_tau1, logits_tau2, logits_zeta = self.enc(x)
        tau1 = nn.functional.softplus(logits_tau1)
        tau2 = nn.functional.softplus(logits_tau2)
        zeta = torch.sigmoid(logits_zeta)
        return tau1, tau2, zeta, logits_zeta


class VAEScaleTrans(nn.Module):

    def __init__(self, args):
        super(VAEScaleTrans, self).__init__()
        self.enc = VAEEncoder(
            num_features_in=args.state_size_where,
            config_hidden=args.enc_scl_trs_hidden,
            vae_size=4,
            vae_groups=2,
        )

    def forward(self, x):
        mu, logvar = self.enc(x)
        sample = reparameterize_normal(mu, logvar)
        scale = torch.sigmoid(sample[..., :2])
        trans = torch.tanh(sample[..., 2:])
        return scale, trans, mu, logvar


class VAEShapeAppear(nn.Module):

    def __init__(self, args):
        super(VAEShapeAppear, self).__init__()
        vae_size_shape = args.vae_size_shp
        vae_size_appear = args.vae_size_apc
        vae_size = vae_size_shape + vae_size_appear
        self.vae_size_split = vae_size_shape
        # Encoder
        self.enc_0 = VAEEncoder(
            num_features_in=args.state_size_what,
            config_hidden=args.enc_shp_apc_hidden,
            vae_size=vae_size,
            vae_groups=2,
        )
        self.enc_1 = VAEEncoder(
            num_features_in=args.state_size_what,
            config_hidden=args.enc_shp_apc_hidden,
            vae_size=vae_size,
            vae_groups=2,
        )
        # Decoder
        self.dec_shape = VAEDecoderComplex(
            num_planes_out=1,
            plane_size_out=args.image_crop_size,
            config_channels=args.dec_shp_channels,
            config_resample=args.dec_shp_resample,
            config_hidden=args.dec_shp_hidden,
            vae_size=vae_size_shape,
        )
        self.dec_appear = VAEDecoderSimple(
            num_planes_out=args.image_channels,
            plane_size_out=args.image_crop_size,
            config_hidden=args.dec_apc_hidden,
            vae_size=vae_size_appear,
        )

    def forward(self, x, zeta):
        mu_0, logvar_0 = self.enc_0(x)
        mu_1, logvar_1 = self.enc_1(x)
        mu = zeta * mu_1 + (1 - zeta) * mu_0
        logvar = zeta * logvar_1 + (1 - zeta) * logvar_0
        sample = reparameterize_normal(mu, logvar)
        sample_shape = sample[..., :self.vae_size_split]
        sample_appear = sample[..., self.vae_size_split:]
        logits_shape_crop = self.dec_shape(sample_shape)
        shape_crop = torch.sigmoid(logits_shape_crop)
        unbiased_appear_crop = self.dec_appear(sample_appear)
        appear_crop = restore_image(unbiased_appear_crop)
        return shape_crop, appear_crop, mu, logvar


class VAEBack(nn.Module):

    def __init__(self, args):
        super(VAEBack, self).__init__()
        vae_size = args.vae_size_back
        # Encoder
        self.enc = VAEEncoder(
            num_features_in=args.state_size_back,
            config_hidden=args.enc_back_hidden,
            vae_size=vae_size,
            vae_groups=2,
        )
        # Decoder
        self.dec = VAEDecoderSimple(
            num_planes_out=args.image_channels,
            plane_size_out=args.image_full_size,
            config_hidden=args.dec_back_hidden,
            vae_size=vae_size,
        )

    def forward(self, x):
        mu, logvar = self.enc(x)
        sample = reparameterize_normal(mu, logvar)
        unbiased_back = self.dec(sample)
        back = restore_image(unbiased_back)
        return back, mu, logvar
