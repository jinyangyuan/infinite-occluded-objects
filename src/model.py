import math
import torch
import torch.nn as nn
from network import InitializerWhere, InitializerWhat, InitializerBack, UpdaterWhere, UpdaterWhat, UpdaterBack, \
    VAEPres, VAEScaleTrans, VAEShapeAppear, VAEBack


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        # Hyperparameters
        self.register_buffer('vae_scl_trs_mu', torch.FloatTensor(args.vae_scl_trs_mu)[None])
        self.register_buffer('vae_scl_trs_invvar', 1 / torch.FloatTensor(args.vae_scl_trs_std)[None].pow(2))
        self.register_buffer('vae_scl_trs_logvar', -self.vae_scl_trs_invvar.log())
        self.num_steps = args.num_steps
        self.max_objects = args.max_objects
        self.image_channels = args.image_channels
        self.image_crop_size = args.image_crop_size
        self.gaussian_invvar = 1 / pow(args.gaussian_std, 2)
        self.gaussian_const = math.log(2 * math.pi) - math.log(self.gaussian_invvar)
        self.vae_pres_alpha = args.vae_pres_alpha
        self.vae_pres_log_alpha = math.log(self.vae_pres_alpha)
        # Storage
        self.zeta = None
        self.shape = None
        self.appear = None
        self.saves_dict = {
            key: [None for _ in range(args.max_objects)] for key in
            [
                'tau1', 'tau2', 'scl_trs_mu', 'scl_trs_logvar', 'shp_apc_mu', 'shp_apc_logvar',
                'scale', 'trans', 'logits_zeta', 'shape_crop', 'appear_crop',
            ]
        }
        self.states_dict = {key: [None for _ in range(args.max_objects)] for key in ['states_where', 'states_what']}
        # Neural networks
        self.init_where = InitializerWhere(args)
        self.init_what = InitializerWhat(args)
        self.init_back = InitializerBack(args)
        self.upd_where = UpdaterWhere(args)
        self.upd_what = UpdaterWhat(args)
        self.upd_back = UpdaterBack(args)
        self.vae_pres = VAEPres(args)
        self.vae_scl_trs = VAEScaleTrans(args)
        self.vae_shp_apc = VAEShapeAppear(args)
        self.vae_back = None
        if args.predict_back:
            self.vae_back = VAEBack(args)

    @staticmethod
    def compute_gamma(zeta, shape):
        x = zeta[..., None, None] * shape
        padded_ones = x.new_ones(1, *x.shape[1:])
        return torch.cat([x, padded_ones]) * torch.cat([padded_ones, 1 - x]).cumprod(0)

    def compute_init_obj_inputs_full(self, images, back, idx):
        gamma = self.compute_gamma(self.zeta[:idx], self.shape[:idx])
        recon = (gamma * torch.cat([self.appear[:idx], back[None]])).sum(0)
        return torch.cat([images, recon], dim=1).detach()

    def compute_upd_back_inputs(self, images, back):
        gamma = self.compute_gamma(self.zeta, self.shape)
        recon = (gamma * torch.cat([self.appear, back[None]])).sum(0)
        return torch.cat([images, recon, back], dim=1).detach()

    def compute_upd_obj_inputs_full(self, images, back, idx):
        gamma1 = self.compute_gamma(self.zeta[:idx], self.shape[:idx])
        gamma2 = self.compute_gamma(self.zeta[idx:idx + 1], self.shape[idx:idx + 1])
        gamma3 = self.compute_gamma(self.zeta[idx + 1:], self.shape[idx + 1:])
        recon1 = (gamma1 * torch.cat([self.appear[:idx], back[None]])).sum(0)
        recon2 = (gamma2 * torch.cat([self.appear[idx:idx + 1], back[None]])).sum(0)
        recon3 = (gamma3 * torch.cat([self.appear[idx + 1:], back[None]])).sum(0)
        return torch.cat([images, recon1, recon2, recon3], dim=1).detach()

    def compute_upd_obj_inputs_crop(self, inputs_full, back, grid_crop, idx):
        shape_crop = self.saves_dict['shape_crop'][idx]
        appear_crop = self.saves_dict['appear_crop'][idx]
        back_crop = nn.functional.grid_sample(back, grid_crop)
        gamma2_crop = self.compute_gamma(self.zeta[idx:idx + 1], shape_crop[None])
        recon2_crop = (gamma2_crop * torch.cat([appear_crop[None], back_crop[None]])).sum(0)
        images_crop = nn.functional.grid_sample(inputs_full[:, :self.image_channels], grid_crop)
        recon1_crop = nn.functional.grid_sample(inputs_full[:, self.image_channels:2 * self.image_channels], grid_crop)
        recon3_crop = nn.functional.grid_sample(inputs_full[:, 3 * self.image_channels:], grid_crop)
        return torch.cat([images_crop, recon1_crop, recon2_crop, recon3_crop], dim=1).detach()

    def compute_grid(self, scale, trans, image_shape_restore):
        theta_crop = torch.stack([
            torch.stack([scale[:, 0], torch.zeros_like(scale[:, 0]), trans[:, 0]], dim=1),
            torch.stack([torch.zeros_like(scale[:, 1]), scale[:, 1], trans[:, 1]], dim=1),
        ], dim=1)
        theta_restore = torch.stack([
            torch.stack([1 / scale[:, 0], torch.zeros_like(scale[:, 0]), -trans[:, 0] / scale[:, 0]], dim=1),
            torch.stack([torch.zeros_like(scale[:, 1]), 1 / scale[:, 1], -trans[:, 1] / scale[:, 1]], dim=1),
        ], dim=1)
        image_shape_crop = (*image_shape_restore[:-2], self.image_crop_size, self.image_crop_size)
        grid_crop = nn.functional.affine_grid(theta_crop, image_shape_crop)
        grid_restore = nn.functional.affine_grid(theta_restore, image_shape_restore)
        return grid_crop, grid_restore

    def update_storage(self, update_dict, idx):
        for key in ['zeta', 'shape', 'appear']:
            x = getattr(self, key)
            setattr(self, key, torch.cat([x[:idx], update_dict[key][None], x[idx + 1:]]))
        for key, val in self.saves_dict.items():
            self.saves_dict[key] = val[:idx] + [update_dict[key]] + val[idx + 1:]
        for key, val in self.states_dict.items():
            self.states_dict[key] = val[:idx] + [update_dict[key]] + val[idx + 1:]

    def forward(self, images, backgrounds):
        ###################
        # Initializations #
        ###################
        self.zeta = images.new_zeros(self.max_objects, images.shape[0], 1)
        self.shape = images.new_zeros(self.max_objects, images.shape[0], 1, *images.shape[2:])
        self.appear = images.new_zeros(self.max_objects, *images.shape)
        # Background
        states_back = self.init_back(images)
        if self.vae_back is not None:
            back, back_mu, back_logvar = self.vae_back(states_back[0])
        else:
            back, back_mu, back_logvar = backgrounds, None, None
        # Objects
        states_base = None
        for idx_obj in range(self.max_objects):
            # States where
            inputs_full = self.compute_init_obj_inputs_full(images, back, idx_obj)
            states_where, states_base = self.init_where(inputs_full, states_base)
            # Scale and translation
            scale, trans, scl_trs_mu, scl_trs_logvar = self.vae_scl_trs(states_where[0])
            grid_crop, grid_restore = self.compute_grid(scale, trans, images.shape)
            # States what
            inputs_crop = nn.functional.grid_sample(inputs_full, grid_crop)
            states_what = self.init_what(inputs_crop)
            # Presence
            tau1, tau2, zeta, logits_zeta = self.vae_pres(states_where[0], states_what[0])
            # Shape and appearance
            shape_crop, appear_crop, shp_apc_mu, shp_apc_logvar = self.vae_shp_apc(states_what[0], zeta)
            shape = nn.functional.grid_sample(shape_crop, grid_restore)
            appear = nn.functional.grid_sample(appear_crop, grid_restore)
            # Update storage
            update_dict = {
                'zeta': zeta, 'shape': shape, 'appear': appear,
                'tau1': tau1, 'tau2': tau2,
                'scl_trs_mu': scl_trs_mu, 'scl_trs_logvar': scl_trs_logvar,
                'shp_apc_mu': shp_apc_mu, 'shp_apc_logvar': shp_apc_logvar,
                'scale': scale, 'trans': trans,
                'logits_zeta': logits_zeta, 'shape_crop': shape_crop, 'appear_crop': appear_crop,
                'states_where': states_where, 'states_what': states_what,
            }
            self.update_storage(update_dict, idx_obj)
        # Result
        result = {
            'zeta': self.zeta, 'shape': self.shape, 'appear': self.appear,
            'back': back, 'back_mu': back_mu, 'back_logvar': back_logvar,
        }
        result.update({key: torch.stack(val) for key, val in self.saves_dict.items()})
        results = [result]
        ###############
        # Refinements #
        ###############
        for _ in range(self.num_steps):
            # Background
            if self.vae_back is not None:
                inputs_back = self.compute_upd_back_inputs(images, back)
                states_back = self.upd_back(inputs_back, states_back)
                back, back_mu, back_logvar = self.vae_back(states_back[0])
            # Objects
            for idx_obj in range(self.max_objects):
                # States where
                inputs_full = self.compute_upd_obj_inputs_full(images, back, idx_obj)
                states_where = self.upd_where(inputs_full, self.states_dict['states_where'][idx_obj])
                # Scale and translation
                scale, trans, scl_trs_mu, scl_trs_logvar = self.vae_scl_trs(states_where[0])
                grid_crop, grid_restore = self.compute_grid(scale, trans, images.shape)
                # States what
                inputs_crop = self.compute_upd_obj_inputs_crop(inputs_full, back, grid_crop, idx_obj)
                states_what = self.upd_what(inputs_crop, self.states_dict['states_what'][idx_obj])
                # Presence
                tau1, tau2, zeta, logits_zeta = self.vae_pres(states_where[0], states_what[0])
                # Shape and appearance
                shape_crop, appear_crop, shp_apc_mu, shp_apc_logvar = self.vae_shp_apc(states_what[0], zeta)
                shape = nn.functional.grid_sample(shape_crop, grid_restore)
                appear = nn.functional.grid_sample(appear_crop, grid_restore)
                # Update storage
                update_dict = {
                    'zeta': zeta, 'shape': shape, 'appear': appear,
                    'tau1': tau1, 'tau2': tau2,
                    'scl_trs_mu': scl_trs_mu, 'scl_trs_logvar': scl_trs_logvar,
                    'shp_apc_mu': shp_apc_mu, 'shp_apc_logvar': shp_apc_logvar,
                    'scale': scale, 'trans': trans,
                    'logits_zeta': logits_zeta, 'shape_crop': shape_crop, 'appear_crop': appear_crop,
                    'states_where': states_where, 'states_what': states_what,
                }
                self.update_storage(update_dict, idx_obj)
            # Result
            result = {
                'zeta': self.zeta, 'shape': self.shape, 'appear': self.appear,
                'back': back, 'back_mu': back_mu, 'back_logvar': back_logvar,
            }
            result.update({key: torch.stack(val) for key, val in self.saves_dict.items()})
            results.append(result)
        return results

    def compute_loss_recon(self, images, zeta, shape, appear, back, ratio):
        gamma = self.compute_gamma(zeta, shape)
        appear_combine = torch.cat([appear, back[None]])
        part_diff = (images - (gamma * appear_combine).sum(0)).pow(2)
        part_elbo = (gamma * (images[None] - appear_combine).pow(2)).sum(0)
        part_opt = (1 - ratio) * part_diff + ratio * part_elbo
        loss = 0.5 * (self.gaussian_const + self.gaussian_invvar * (part_opt - part_opt.detach() + part_elbo.detach()))
        return loss.sum()

    def compute_kld_pres(self, tau1, tau2, zeta, logits_zeta):
        psi1 = torch.digamma(tau1)
        psi2 = torch.digamma(tau2)
        psi12 = torch.digamma(tau1 + tau2)
        # Beta
        loss_beta_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - torch.lgamma(tau2) - self.vae_pres_log_alpha
        loss_beta_2 = (tau1 - self.vae_pres_alpha) * psi1
        loss_beta_3 = (tau2 - 1) * psi2
        loss_beta_4 = -(tau1 + tau2 - self.vae_pres_alpha - 1) * psi12
        loss_beta = loss_beta_1 + loss_beta_2 + loss_beta_3 + loss_beta_4
        # Bernoulli
        log_zeta = nn.functional.logsigmoid(logits_zeta)
        log_one_minus_zeta = log_zeta - logits_zeta
        psi1_le_sum = psi1.cumsum(0)
        psi12_le_sum = psi12.cumsum(0)
        kappa1 = psi1_le_sum - psi12_le_sum
        psi1_lt_sum = torch.cat([psi1_le_sum.new_zeros(1, *psi1_le_sum.shape[1:]), psi1_le_sum[:-1]])
        logits_coef = psi2 + psi1_lt_sum - psi12_le_sum
        kappa2_list = []
        for idx in range(logits_coef.shape[0]):
            coef = torch.softmax(logits_coef[:idx + 1], dim=0)
            log_coef = nn.functional.log_softmax(logits_coef[:idx + 1], dim=0)
            coef_le_sum = coef.cumsum(0)
            coef_lt_sum = torch.cat([coef_le_sum.new_zeros(1, *coef_le_sum.shape[1:]), coef_le_sum[:-1]])
            part1 = (coef * psi2[:idx + 1]).sum(0)
            part2 = ((1 - coef_le_sum[:-1]) * psi1[:idx]).sum(0)
            part3 = -((1 - coef_lt_sum) * psi12[:idx + 1]).sum(0)
            part4 = -(coef * log_coef).sum(0)
            kappa2_list.append(part1 + part2 + part3 + part4)
        kappa2 = torch.stack(kappa2_list)
        loss_bernoulli = zeta * (log_zeta - kappa1) + (1 - zeta) * (log_one_minus_zeta - kappa2)
        return loss_beta.sum() + loss_bernoulli.sum()

    @staticmethod
    def compute_kld_normal(mu, logvar, mu_prior, logvar_prior, invvar_prior):
        loss = 0.5 * (logvar_prior - logvar + invvar_prior * ((mu - mu_prior).pow(2) + logvar.exp()) - 1)
        return loss.sum()

    def compute_reg_back(self, images, back, coef):
        images_mean = images.mean(-1, keepdim=True).mean(-2, keepdim=True)
        reg = coef * 0.5 * self.gaussian_invvar * (back - images_mean).pow(2)
        return reg.sum()

    def compute_loss(self, images, result, ratio_recon, coef_back):
        # Load result
        tau1 = result['tau1']
        tau2 = result['tau2']
        zeta = result['zeta']
        logits_zeta = result['logits_zeta']
        shape = result['shape']
        appear = result['appear']
        back = result['back']
        scl_trs_mu, scl_trs_logvar = result['scl_trs_mu'], result['scl_trs_logvar']
        shp_apc_mu, shp_apc_logvar = result['shp_apc_mu'], result['shp_apc_logvar']
        back_mu, back_logvar = result['back_mu'], result['back_logvar']
        # Compute loss
        loss_recon = self.compute_loss_recon(images, zeta, shape, appear, back, ratio_recon)
        loss_pres = self.compute_kld_pres(tau1, tau2, zeta, logits_zeta)
        loss_scl_trs = self.compute_kld_normal(
            scl_trs_mu, scl_trs_logvar, self.vae_scl_trs_mu, self.vae_scl_trs_logvar, self.vae_scl_trs_invvar)
        loss_shp_apc = self.compute_kld_normal(shp_apc_mu, shp_apc_logvar, 0, 0, 1)
        loss_back = 0
        if back_mu is not None and back_logvar is not None:
            loss_back = self.compute_kld_normal(back_mu, back_logvar, 0, 0, 1) + \
                        self.compute_reg_back(images, back, coef_back)
        loss = (loss_recon + loss_pres + loss_scl_trs + loss_shp_apc + loss_back) / images.shape[0]
        return loss


def get_model(args, path=None):
    model = Model(args)
    if path is not None:
        model.load_state_dict(torch.load(path))
    return model
