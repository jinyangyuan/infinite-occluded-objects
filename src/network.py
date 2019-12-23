import torch
import torch.nn as nn
import torch.nn.functional
from building_block import FC, EncoderConv, DecoderConv


def normalize_image(x):
    return x * 2 - 1


def restore_image(x):
    return (x + 1) * 0.5


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise


class UpdaterBase(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, num_planes_in, plane_height_in,
                 plane_width_in, num_features_out, state_size):
        super(UpdaterBase, self).__init__()
        self.net = EncoderConv(
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            hidden_list=hidden_list,
            num_planes_in=num_planes_in,
            plane_height_in=plane_height_in,
            plane_width_in=plane_width_in,
            num_features_out=num_features_out,
            last_activation=True,
        )
        self.lstm = nn.LSTMCell(num_features_out, state_size)

    def forward(self, inputs, states=None):
        x = normalize_image(inputs)
        x = self.net(x)
        states = self.lstm(x, states)
        return states


class InitializerBack(UpdaterBase):

    def __init__(self, args):
        super(InitializerBack, self).__init__(
            channel_list=args.init_back_channel_list,
            kernel_list=args.init_back_kernel_list,
            stride_list=args.init_back_stride_list,
            hidden_list=args.init_back_hidden_list,
            num_planes_in=args.image_planes,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.init_back_size,
            state_size=args.state_back_size,
        )


class InitializerFull(nn.Module):

    def __init__(self, args):
        super(InitializerFull, self).__init__()
        self.upd_main = UpdaterBase(
            channel_list=args.init_full_channel_list,
            kernel_list=args.init_full_kernel_list,
            stride_list=args.init_full_stride_list,
            hidden_list=args.init_full_main_hidden_list,
            num_planes_in=args.image_planes * 2 + 1,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.init_full_main_size,
            state_size=args.state_main_size,
        )
        self.net_full = FC(
            hidden_list=args.init_full_full_hidden_list,
            num_features_in=args.state_main_size,
            num_features_out=args.init_full_full_size,
            last_activation=True,
        )
        self.lstm_full = nn.LSTMCell(args.init_full_full_size, args.state_full_size)

    def forward(self, inputs, states_main):
        states_main = self.upd_main(inputs, states_main)
        x = self.net_full(states_main[0])
        states_full = self.lstm_full(x)
        return states_full, states_main


class InitializerCrop(UpdaterBase):

    def __init__(self, args):
        super(InitializerCrop, self).__init__(
            channel_list=args.init_crop_channel_list,
            kernel_list=args.init_crop_kernel_list,
            stride_list=args.init_crop_stride_list,
            hidden_list=args.init_crop_hidden_list,
            num_planes_in=args.image_planes * 2 + 1,
            plane_height_in=args.image_crop_height,
            plane_width_in=args.image_crop_width,
            num_features_out=args.init_crop_size,
            state_size=args.state_crop_size,
        )


class UpdaterBack(UpdaterBase):

    def __init__(self, args):
        super(UpdaterBack, self).__init__(
            channel_list=args.upd_back_channel_list,
            kernel_list=args.upd_back_kernel_list,
            stride_list=args.upd_back_stride_list,
            hidden_list=args.upd_back_hidden_list,
            num_planes_in=args.image_planes * 3 + 1,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.upd_back_size,
            state_size=args.state_back_size,
        )


class UpdaterFull(UpdaterBase):

    def __init__(self, args):
        super(UpdaterFull, self).__init__(
            channel_list=args.upd_full_channel_list,
            kernel_list=args.upd_full_kernel_list,
            stride_list=args.upd_full_stride_list,
            hidden_list=args.upd_full_hidden_list,
            num_planes_in=args.image_planes * 4 + 2,
            plane_height_in=args.image_full_height,
            plane_width_in=args.image_full_width,
            num_features_out=args.upd_full_size,
            state_size=args.state_full_size,
        )


class UpdaterCrop(UpdaterBase):

    def __init__(self, args):
        super(UpdaterCrop, self).__init__(
            channel_list=args.upd_crop_channel_list,
            kernel_list=args.upd_crop_kernel_list,
            stride_list=args.upd_crop_stride_list,
            hidden_list=args.upd_crop_hidden_list,
            num_planes_in=args.image_planes * 4 + 2,
            plane_height_in=args.image_crop_height,
            plane_width_in=args.image_crop_width,
            num_features_out=args.upd_crop_size,
            state_size=args.state_crop_size,
        )


class EncDecBack(nn.Module):

    def __init__(self, args):
        super(EncDecBack, self).__init__()
        self.enc = FC(
            hidden_list=args.enc_back_hidden_list,
            num_features_in=args.state_back_size,
            num_features_out=args.latent_back_size * 2,
            last_activation=False,
        )
        self.dec_color = FC(
            hidden_list=args.dec_back_color_hidden_list,
            num_features_in=args.latent_back_size,
            num_features_out=args.image_planes,
            last_activation=False,
        )
        self.dec_diff = DecoderConv(
            channel_list_rev=args.dec_back_diff_channel_list_rev,
            kernel_list_rev=args.dec_back_diff_kernel_list_rev,
            stride_list_rev=args.dec_back_diff_stride_list_rev,
            hidden_list_rev=args.dec_back_diff_hidden_list_rev,
            num_features_in=args.latent_back_size,
            num_planes_out=args.image_planes,
            plane_height_out=args.image_full_height,
            plane_width_out=args.image_full_width,
        )

    def encode(self, x):
        back_mu, back_logvar = self.enc(x).chunk(2, dim=-1)
        return back_mu, back_logvar

    def decode(self, back_mu, back_logvar):
        sample = reparameterize_normal(back_mu, back_logvar)
        back_color = self.dec_color(sample)[..., None, None]
        back_diff = self.dec_diff(sample)
        back = restore_image(back_color + back_diff)
        return back, back_diff

    def forward(self, x):
        back_mu, back_logvar = self.encode(x)
        back, back_diff = self.decode(back_mu, back_logvar)
        result = {'back': back, 'back_diff': back_diff, 'back_mu': back_mu, 'back_logvar': back_logvar}
        return result


class EncDecFull(nn.Module):

    def __init__(self, args):
        super(EncDecFull, self).__init__()
        self.enc_pres = FC(
            hidden_list=args.enc_pres_hidden_list,
            num_features_in=args.state_full_size,
            num_features_out=3,
            last_activation=False,
        )
        self.enc_where_0 = FC(
            hidden_list=args.enc_where_hidden_list,
            num_features_in=args.state_full_size,
            num_features_out=8,
            last_activation=False,
        )
        self.enc_where_1 = FC(
            hidden_list=args.enc_where_hidden_list,
            num_features_in=args.state_full_size,
            num_features_out=8,
            last_activation=False,
        )

    def encode(self, x):
        logits_tau1, logits_tau2, logits_zeta = self.enc_pres(x).chunk(3, dim=-1)
        tau1 = nn.functional.softplus(logits_tau1)
        tau2 = nn.functional.softplus(logits_tau2)
        zeta = torch.sigmoid(logits_zeta)
        where_mu_0, where_logvar_0 = self.enc_where_0(x).chunk(2, dim=-1)
        where_mu_1, where_logvar_1 = self.enc_where_1(x).chunk(2, dim=-1)
        where_mu = zeta * where_mu_1 + (1 - zeta) * where_mu_0
        where_logvar = zeta * where_logvar_1 + (1 - zeta) * where_logvar_0
        return tau1, tau2, zeta, logits_zeta, where_mu, where_logvar

    @staticmethod
    def decode(where_mu, where_logvar):
        sample = reparameterize_normal(where_mu, where_logvar)
        scl = torch.sigmoid(sample[..., :2])
        trs = torch.tanh(sample[..., 2:])
        return scl, trs

    def forward(self, x):
        tau1, tau2, zeta, logits_zeta, where_mu, where_logvar = self.encode(x)
        scl, trs = self.decode(where_mu, where_logvar)
        result = {
            'scl': scl, 'trs': trs,
            'tau1': tau1, 'tau2': tau2, 'zeta': zeta, 'logits_zeta': logits_zeta,
            'where_mu': where_mu, 'where_logvar': where_logvar,
        }
        return result


class EncDecCrop(nn.Module):

    def __init__(self, args):
        super(EncDecCrop, self).__init__()
        self.enc_0 = FC(
            hidden_list=args.enc_what_hidden_list,
            num_features_in=args.state_crop_size,
            num_features_out=args.latent_what_size * 2,
            last_activation=False,
        )
        self.enc_1 = FC(
            hidden_list=args.enc_what_hidden_list,
            num_features_in=args.state_crop_size,
            num_features_out=args.latent_what_size * 2,
            last_activation=False,
        )
        self.dec_apc_color = FC(
            hidden_list=args.dec_apc_color_hidden_list,
            num_features_in=args.latent_what_size,
            num_features_out=args.image_planes,
            last_activation=False,
        )
        self.dec_apc_diff = DecoderConv(
            channel_list_rev=args.dec_apc_diff_channel_list_rev,
            kernel_list_rev=args.dec_apc_diff_kernel_list_rev,
            stride_list_rev=args.dec_apc_diff_stride_list_rev,
            hidden_list_rev=args.dec_apc_diff_hidden_list_rev,
            num_features_in=args.latent_what_size,
            num_planes_out=args.image_planes,
            plane_height_out=args.image_crop_height,
            plane_width_out=args.image_crop_width,
        )
        self.dec_shp = DecoderConv(
            channel_list_rev=args.dec_shp_channel_list_rev,
            kernel_list_rev=args.dec_shp_kernel_list_rev,
            stride_list_rev=args.dec_shp_stride_list_rev,
            hidden_list_rev=args.dec_shp_hidden_list_rev,
            num_features_in=args.latent_what_size,
            num_planes_out=1,
            plane_height_out=args.image_crop_height,
            plane_width_out=args.image_crop_width,
        )

    def encode(self, x, zeta):
        what_mu_0, what_logvar_0 = self.enc_0(x).chunk(2, dim=-1)
        what_mu_1, what_logvar_1 = self.enc_1(x).chunk(2, dim=-1)
        what_mu = zeta * what_mu_1 + (1 - zeta) * what_mu_0
        what_logvar = zeta * what_logvar_1 + (1 - zeta) * what_logvar_0
        return what_mu, what_logvar

    def decode(self, what_mu, what_logvar, grid_full):
        sample = reparameterize_normal(what_mu, what_logvar)
        apc_crop_color = self.dec_apc_color(sample)[..., None, None]
        apc_crop_diff = self.dec_apc_diff(sample)
        apc_crop = restore_image(apc_crop_color + apc_crop_diff)
        logits_shp_crop = self.dec_shp(sample)
        shp_crop = torch.sigmoid(logits_shp_crop)
        apc = nn.functional.grid_sample(apc_crop, grid_full)
        shp = nn.functional.grid_sample(shp_crop, grid_full)
        return apc, shp, apc_crop, apc_crop_diff, shp_crop

    def forward(self, x, zeta, grid_full):
        what_mu, what_logvar = self.encode(x, zeta)
        apc, shp, apc_crop, apc_crop_diff, shp_crop = self.decode(what_mu, what_logvar, grid_full)
        result = {
            'apc': apc, 'shp': shp, 'apc_crop': apc_crop, 'apc_crop_diff': apc_crop_diff, 'shp_crop': shp_crop,
            'what_mu': what_mu, 'what_logvar': what_logvar,
        }
        return result
