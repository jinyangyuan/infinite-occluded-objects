import math
import torch
import torch.nn as nn
import torch.nn.functional
from network import InitializerBack, InitializerFull, InitializerCrop, UpdaterBack, UpdaterFull, UpdaterCrop, \
    EncDecBack, EncDecFull, EncDecCrop


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        # Hyperparameters
        self.register_buffer('prior_where_mu', torch.tensor(args.prior_where_mu)[None, None])
        self.register_buffer('prior_where_invvar', 1 / torch.tensor(args.prior_where_std)[None, None].pow(2))
        self.register_buffer('prior_where_logvar', -self.prior_where_invvar.log())
        self.num_steps = args.num_steps
        self.max_objects = args.max_objects
        self.image_full_height = args.image_full_height
        self.image_full_width = args.image_full_width
        self.image_crop_height = args.image_crop_height
        self.image_crop_width = args.image_crop_width
        self.gaussian_invvar = 1 / pow(args.gaussian_std, 2)
        self.gaussian_const = math.log(2 * math.pi / self.gaussian_invvar)
        self.prior_pres_alpha = args.prior_pres_alpha
        self.prior_pres_log_alpha = math.log(self.prior_pres_alpha)
        # Neural networks
        self.init_back = InitializerBack(args)
        self.init_full = InitializerFull(args)
        self.init_crop = InitializerCrop(args)
        self.upd_back = UpdaterBack(args)
        self.upd_full = UpdaterFull(args)
        self.upd_crop = UpdaterCrop(args)
        self.enc_dec_back = EncDecBack(args)
        self.enc_dec_full = EncDecFull(args)
        self.enc_dec_crop = EncDecCrop(args)

    def compute_grid(self, scl, trs, batch_size):
        theta_crop = torch.stack([
            torch.stack([scl[:, 0], torch.zeros_like(scl[:, 0]), trs[:, 0]], dim=1),
            torch.stack([torch.zeros_like(scl[:, 1]), scl[:, 1], trs[:, 1]], dim=1),
        ], dim=1)
        theta_full = torch.stack([
            torch.stack([1 / scl[:, 0], torch.zeros_like(scl[:, 0]), -trs[:, 0] / scl[:, 0]], dim=1),
            torch.stack([torch.zeros_like(scl[:, 1]), 1 / scl[:, 1], -trs[:, 1] / scl[:, 1]], dim=1),
        ], dim=1)
        grid_crop = nn.functional.affine_grid(
            theta_crop, [batch_size, 1, self.image_crop_height, self.image_crop_width])
        grid_full = nn.functional.affine_grid(
            theta_full, [batch_size, 1, self.image_full_height, self.image_full_width])
        return grid_crop, grid_full

    @staticmethod
    def compute_gamma(shp, zeta):
        x = shp * zeta[..., None, None]
        padded_ones = x.new_ones(1, *x.shape[1:])
        return torch.cat([x, padded_ones]) * torch.cat([padded_ones, 1 - x]).cumprod(0)

    def compute_init_full_inputs(self, images, result_back, result_obj):
        gamma = self.compute_gamma(result_obj['shp'], result_obj['zeta'])
        recon = (gamma * torch.cat([result_obj['apc'], result_back['back'][None]])).sum(0)
        mask = 1 - gamma[-1]
        return torch.cat([images, recon, mask], dim=1).detach()

    def compute_upd_back_inputs(self, images, result_back, result_obj):
        inputs_exclude = self.compute_init_full_inputs(images, result_back, result_obj)
        return torch.cat([inputs_exclude, result_back['back']], dim=1).detach()

    def compute_upd_full_inputs(self, images, result_back, result_obj, idx):
        gamma_above = self.compute_gamma(result_obj['shp'][:idx], result_obj['zeta'][:idx])
        gamma_below = self.compute_gamma(result_obj['shp'][idx + 1:], result_obj['zeta'][idx + 1:])
        gamma_cur = self.compute_gamma(result_obj['shp'][idx:idx + 1], result_obj['zeta'][idx:idx + 1])
        recon_above = (gamma_above * torch.cat([result_obj['apc'][:idx], result_back['back'][None]])).sum(0)
        recon_below = (gamma_below * torch.cat([result_obj['apc'][idx + 1:], result_back['back'][None]])).sum(0)
        mask_above = 1 - gamma_above[-1]
        recon_cur = (gamma_cur * torch.cat([result_obj['apc'][idx:idx + 1], result_back['back'][None]])).sum(0)
        mask_cur = 1 - gamma_cur[-1]
        return torch.cat([images, recon_above, recon_below, mask_above, recon_cur, mask_cur], dim=1).detach()

    def compute_upd_crop_inputs(self, images, result_back, result_obj, grid_crop, idx):
        inputs_full = self.compute_upd_full_inputs(images, result_back, result_obj, idx)
        return nn.functional.grid_sample(inputs_full, grid_crop)

    @staticmethod
    def initialize_storage(result_obj, states_dict, update_dict):
        for key, val in result_obj.items():
            if val is None:
                result_obj[key] = update_dict[key][None]
            else:
                result_obj[key] = torch.cat([val, update_dict[key][None]])
        for key in states_dict:
            states_dict[key].append(update_dict[key])
        return

    @staticmethod
    def update_storage(result_obj, states_dict, update_dict, idx):
        for key, val in result_obj.items():
            if states_dict is not None or key in update_dict:
                result_obj[key] = torch.cat([val[:idx], update_dict[key][None], val[idx + 1:]])
        if states_dict is not None:
            for key, val in states_dict.items():
                states_dict[key] = val[:idx] + [update_dict[key]] + val[idx + 1:]
        return

    def compute_indices(self, images, result_obj, eps=1e-5):
        diffs_sq = (result_obj['apc'] - images[None]).pow(2).sum(-3, keepdim=True).detach()
        masks = result_obj['shp'].clone().detach()
        coefs = masks.new_ones(masks.shape[:-2])
        indices_list = []
        for _ in range(diffs_sq.shape[0]):
            vis_diffs_sq = (masks * diffs_sq).view(*masks.shape[:-2], -1).sum(-1)
            vis_areas = masks.view(*masks.shape[:-2], -1).sum(-1)
            vis_max_vals = masks.view(*masks.shape[:-2], -1).max(-1).values
            scores = coefs * vis_max_vals * result_obj['zeta'] * \
                     torch.exp(-0.5 * self.gaussian_invvar * vis_diffs_sq / (vis_areas + eps))
            indices = torch.argmax(scores, dim=0)
            indices_list.append(indices)
            mask = torch.gather(masks, 0, indices[None, ..., None, None].expand(-1, -1, *masks.shape[2:]))
            masks *= 1 - mask
            coefs.scatter_(0, indices[None], -1)
        indices = torch.stack(indices_list)
        return indices

    @staticmethod
    def adjust_order_sub(x, indices):
        if x.dim() == 3:
            x = torch.gather(x, 0, indices.expand(-1, -1, *x.shape[2:]))
        elif x.dim() == 5:
            x = torch.gather(x, 0, indices[..., None, None].expand(-1, -1, *x.shape[2:]))
        else:
            raise AssertionError
        return x

    def adjust_order(self, images, result_obj, states_dict):
        indices = self.compute_indices(images, result_obj)
        for key, val in result_obj.items():
            result_obj[key] = self.adjust_order_sub(val, indices)
        for key, val in states_dict.items():
            states_0 = self.adjust_order_sub(torch.stack([n[0] for n in val]), indices)
            states_1 = self.adjust_order_sub(torch.stack([n[1] for n in val]), indices)
            states_dict[key] = [(n_0, n_1) for n_0, n_1 in zip(states_0, states_1)]
        return

    @staticmethod
    def transform_result(result):
        for key, val in result.items():
            if key not in ['back', 'back_diff', 'back_mu', 'back_logvar']:
                result[key] = val.transpose(0, 1)
        return result

    def forward(self, images):
        ###################
        # Initializations #
        ###################
        # Background
        states_back = self.init_back(images)
        result_back = self.enc_dec_back(states_back[0])
        # Objects
        result_obj = {
            'apc': images.new_empty(0, *images.shape),
            'shp': images.new_zeros(0, images.shape[0], 1, *images.shape[2:]),
            'zeta': images.new_zeros(0, images.shape[0], 1),
        }
        result_obj.update({
            key: None for key in
            [
                'tau1', 'tau2', 'logits_zeta', 'where_mu', 'where_logvar', 'what_mu', 'what_logvar',
                'scl', 'trs', 'apc_crop', 'apc_crop_diff', 'shp_crop',
            ]
        })
        states_dict = {key: [] for key in ['states_full', 'states_crop']}
        states_main = None
        for _ in range(self.max_objects):
            # Full
            inputs_full = self.compute_init_full_inputs(images, result_back, result_obj)
            states_full, states_main = self.init_full(inputs_full, states_main)
            result_full = self.enc_dec_full(states_full[0])
            grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'], images.shape[0])
            # Crop
            inputs_crop = nn.functional.grid_sample(inputs_full, grid_crop)
            states_crop = self.init_crop(inputs_crop)
            result_crop = self.enc_dec_crop(states_crop[0], result_full['zeta'], grid_full)
            # Update storage
            update_dict = {**result_full, **result_crop, 'states_full': states_full, 'states_crop': states_crop}
            self.initialize_storage(result_obj, states_dict, update_dict)
        # Adjust order
        self.adjust_order(images, result_obj, states_dict)
        # Result
        result = {**result_back, **result_obj}
        results = [result]
        ###############
        # Refinements #
        ###############
        for _ in range(self.num_steps):
            # Background
            inputs_back = self.compute_upd_back_inputs(images, result_back, result_obj)
            states_back = self.upd_back(inputs_back, states_back)
            result_back = self.enc_dec_back(states_back[0])
            # Objects
            for idx_obj in range(self.max_objects):
                # Full
                inputs_full = self.compute_upd_full_inputs(images, result_back, result_obj, idx_obj)
                states_full = self.upd_full(inputs_full, states_dict['states_full'][idx_obj])
                result_full = self.enc_dec_full(states_full[0])
                grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'], images.shape[0])
                apc = nn.functional.grid_sample(result_obj['apc_crop'][idx_obj], grid_full)
                shp = nn.functional.grid_sample(result_obj['shp_crop'][idx_obj], grid_full)
                update_dict = {'apc': apc, 'shp': shp, 'zeta': result_full['zeta']}
                self.update_storage(result_obj, None, update_dict, idx_obj)
                # Crop
                inputs_crop = self.compute_upd_crop_inputs(images, result_back, result_obj, grid_crop, idx_obj)
                states_crop = self.upd_crop(inputs_crop, states_dict['states_crop'][idx_obj])
                result_crop = self.enc_dec_crop(states_crop[0], result_full['zeta'], grid_full)
                # Update storage
                update_dict = {**result_full, **result_crop, 'states_full': states_full, 'states_crop': states_crop}
                self.update_storage(result_obj, states_dict, update_dict, idx_obj)
            # Adjust order
            self.adjust_order(images, result_obj, states_dict)
            # Result
            result = {**result_back, **result_obj}
            results.append(result)
        results = [self.transform_result(n) for n in results]
        return results

    def compute_loss_recon(self, images, result, ratio):
        apc_n_back = torch.cat([result['apc'], result['back'][None]])
        gamma = self.compute_gamma(result['shp'], result['zeta'])
        part_diff = (images - (gamma * apc_n_back).sum(0)).pow(2)
        part_elbo = (gamma * (images[None] - apc_n_back).pow(2)).sum(0)
        part_opt = ratio * part_diff + (1 - ratio) * part_elbo
        loss = 0.5 * (self.gaussian_const + self.gaussian_invvar * (part_opt - part_opt.detach() + part_elbo.detach()))
        return loss.sum()

    @staticmethod
    def compute_kld_normal(mu, logvar, prior_mu, prior_logvar, prior_invvar):
        loss = 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).pow(2) + logvar.exp()) - 1)
        return loss.sum()

    def compute_kld_pres(self, result):
        tau1 = result['tau1']
        tau2 = result['tau2']
        zeta = result['zeta']
        logits_zeta = result['logits_zeta']
        psi1 = torch.digamma(tau1)
        psi2 = torch.digamma(tau2)
        psi12 = torch.digamma(tau1 + tau2)
        # Beta
        loss_beta_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - torch.lgamma(tau2) - self.prior_pres_log_alpha
        loss_beta_2 = (tau1 - self.prior_pres_alpha) * psi1
        loss_beta_3 = (tau2 - 1) * psi2
        loss_beta_4 = -(tau1 + tau2 - self.prior_pres_alpha - 1) * psi12
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

    def compute_loss_back_prior(self, images, back):
        back_prior = images.view(*images.shape[:-2], -1).median(-1).values[..., None, None]
        loss = 0.5 * self.gaussian_invvar * (back - back_prior).pow(2).sum()
        return loss - loss.detach()

    def compute_loss_diff(self, x):
        loss = 0.5 * self.gaussian_invvar * x.pow(2).sum()
        return loss - loss.detach()

    def compute_batch_loss(self, images, result, coef_dict):
        loss_recon = self.compute_loss_recon(images, result, coef_dict['recon'])
        loss_kld_back = self.compute_kld_normal(result['back_mu'], result['back_logvar'], 0, 0, 1)
        loss_kld_pres = self.compute_kld_pres(result)
        loss_kld_where = self.compute_kld_normal(result['where_mu'], result['where_logvar'], self.prior_where_mu,
                                                 self.prior_where_logvar, self.prior_where_invvar)
        loss_kld_what = self.compute_kld_normal(result['what_mu'], result['what_logvar'], 0, 0, 1)
        loss_back_prior = coef_dict['back_prior'] * self.compute_loss_back_prior(images, result['back'])
        loss_back_diff = coef_dict['back_diff'] * self.compute_loss_diff(result['back_diff'])
        loss_apc_diff = coef_dict['apc_diff'] * self.compute_loss_diff(result['apc_crop_diff'])
        loss = loss_recon + loss_kld_back + loss_kld_pres + loss_kld_where + loss_kld_what + \
               loss_back_prior + loss_back_diff + loss_apc_diff
        return loss

    def compute_log_likelihood(self, images, result, segre, recon_scene, eps=1e-10):
        diff_mixture = torch.cat([result['apc'], result['back'][None]]) - images[None]
        raw_ll_mixture = -0.5 * (self.gaussian_const + self.gaussian_invvar * diff_mixture.pow(2)).sum(-3, keepdim=True)
        ll_mixture = torch.logsumexp(segre.clamp(min=eps).log() + raw_ll_mixture, dim=0)
        ll_mixture = ll_mixture.view(ll_mixture.shape[0], -1).sum(-1)
        diff_single = recon_scene - images
        ll_single = -0.5 * (self.gaussian_const + self.gaussian_invvar * diff_single.pow(2))
        ll_single = ll_single.view(ll_single.shape[0], -1).sum(-1)
        return ll_mixture, ll_single


def get_model(args, path=None):
    model = Model(args)
    if path is not None:
        load_dict = torch.load(path)
        model_dict = model.state_dict()
        for key in model_dict:
            if key in load_dict and model_dict[key].shape == load_dict[key].shape:
                model_dict[key] = load_dict[key]
            else:
                print('"{}" not loaded'.format(key))
        model.load_state_dict(model_dict)
    return nn.DataParallel(model)
