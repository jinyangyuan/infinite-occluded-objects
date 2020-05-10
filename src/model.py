import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from network import InitializerBack, InitializerFull, InitializerCrop, UpdaterBack, UpdaterFull, UpdaterCrop, \
    NetworkBack, NetworkFull, NetworkCrop


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.register_buffer('prior_stn_mu', torch.tensor(config['prior_stn_mu'])[None, None])
        self.register_buffer('prior_stn_invvar', 1 / torch.tensor(config['prior_stn_std']).pow(2)[None, None])
        self.register_buffer('prior_stn_logvar', -self.prior_stn_invvar.log())
        self.obj_slots = config['num_slots'] - 1
        self.num_steps = config['num_steps']
        self.image_shape = config['image_shape']
        self.crop_shape = config['crop_shape']
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.prior_pres_alpha = config['prior_pres_alpha']
        self.prior_pres_log_alpha = math.log(self.prior_pres_alpha)
        self.seg_bck = config['seg_bck']
        # Neural networks
        self.init_back = InitializerBack(config)
        self.init_full = InitializerFull(config)
        self.init_crop = InitializerCrop(config)
        self.upd_back = UpdaterBack(config)
        self.upd_full = UpdaterFull(config)
        self.upd_crop = UpdaterCrop(config)
        self.net_back = NetworkBack(config)
        self.net_full = NetworkFull(config)
        self.net_crop = NetworkCrop(config)

    def forward(self, images, labels, step_wt):
        ###################
        # Initializations #
        ###################
        # Background
        states_back = self.init_back(images)
        result_bck = self.net_back(states_back[0])
        # Objects
        result_obj = {
            'apc': torch.zeros([0, *images.shape], device=images.device),
            'shp': torch.zeros([0, images.shape[0], 1, *images.shape[2:]], device=images.device),
            'zeta': torch.zeros([0, images.shape[0], 1], device=images.device),
        }
        result_obj.update({
            key: None for key in
            [
                'tau1', 'tau2', 'logits_zeta', 'stn_mu', 'stn_logvar', 'obj_mu', 'obj_logvar',
                'scl', 'trs', 'apc_crop', 'apc_crop_res', 'shp_crop',
            ]
        })
        states_dict = {key: [] for key in ['states_full', 'states_crop']}
        states_main = None
        for _ in range(self.obj_slots):
            # Full
            init_full_in = self.compute_init_full_in(images, result_bck, result_obj)
            states_full, states_main = self.init_full(init_full_in, states_main)
            result_full = self.net_full(states_full[0])
            # Crop
            grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'])
            init_crop_in = nn_func.grid_sample(init_full_in, grid_crop, align_corners=False)
            states_crop = self.init_crop(init_crop_in)
            result_crop = self.net_crop(states_full[0], states_crop[0], grid_full)
            # Update storage
            update_dict = {**result_full, **result_crop, 'states_full': states_full, 'states_crop': states_crop}
            self.initialize_storage(result_obj, states_dict, update_dict)
        # Adjust order
        self.adjust_order(images, result_obj, states_dict)
        # Losses
        mask = self.compute_mask(result_obj['shp'], result_obj['zeta'])
        raw_pixel_ll = self.compute_raw_pixel_ll(images, result_bck['bck'], result_obj['apc'])
        step_losses = self.compute_step_losses(images, result_bck, result_obj, mask, raw_pixel_ll)
        losses = {key: [val] for key, val in step_losses.items()}
        ###############
        # Refinements #
        ###############
        for _ in range(self.num_steps):
            # Background
            upd_back_in = self.compute_upd_back_in(images, result_bck, result_obj)
            states_back = self.upd_back(upd_back_in, states_back)
            result_bck = self.net_back(states_back[0])
            # Objects
            for idx_obj in range(self.obj_slots):
                # Full
                upd_full_in = self.compute_upd_full_in(images, result_bck, result_obj, idx_obj)
                states_full = self.upd_full(upd_full_in, states_dict['states_full'][idx_obj])
                result_full = self.net_full(states_full[0])
                # Crop
                grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'])
                upd_crop_in = self.compute_upd_crop_in(result_bck, result_obj, upd_full_in, grid_crop, idx_obj)
                states_crop = self.upd_crop(upd_crop_in, states_dict['states_crop'][idx_obj])
                result_crop = self.net_crop(states_full[0], states_crop[0], grid_full)
                # Update storage
                update_dict = {**result_full, **result_crop, 'states_full': states_full, 'states_crop': states_crop}
                self.update_storage(result_obj, states_dict, update_dict, idx_obj)
            # Adjust order
            self.adjust_order(images, result_obj, states_dict)
            # Losses
            mask = self.compute_mask(result_obj['shp'], result_obj['zeta'])
            raw_pixel_ll = self.compute_raw_pixel_ll(images, result_bck['bck'], result_obj['apc'])
            step_losses = self.compute_step_losses(images, result_bck, result_obj, mask, raw_pixel_ll)
            for key, val in step_losses.items():
                losses[key].append(val)
        ###########
        # Outputs #
        ###########
        # Losses
        sum_step_wt = step_wt.sum(1)
        losses = {key: torch.stack(val, dim=1) for key, val in losses.items()}
        losses = {key: (step_wt * val).sum(1) / sum_step_wt for key, val in losses.items()}
        # Results
        apc_all = torch.cat([result_obj['apc'], result_bck['bck'][None]]).transpose(0, 1)
        shp = result_obj['shp']
        shp_all = torch.cat([shp, torch.ones([1, *shp.shape[1:]], device=shp.device)]).transpose(0, 1)
        pres = torch.bernoulli(result_obj['zeta'])
        mask = self.compute_mask(shp, pres).transpose(0, 1)
        pres = pres.squeeze(-1).transpose(0, 1)
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1], device=images.device)], dim=1)
        recon = (mask * apc_all).sum(1)
        scl = result_obj['scl'].transpose(0, 1)
        trs = result_obj['trs'].transpose(0, 1)
        raw_pixel_ll = raw_pixel_ll.transpose(0, 1)
        segment_all = torch.argmax(mask, dim=1, keepdim=True)
        segment_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
        mask_oh_all = torch.zeros_like(mask)
        mask_oh_obj = torch.zeros_like(mask[:, :-1])
        mask_oh_all.scatter_(1, segment_all, 1)
        mask_oh_obj.scatter_(1, segment_obj, 1)
        results = {'apc': apc_all, 'shp': shp_all, 'pres': pres_all, 'scl': scl, 'trs': trs, 'recon': recon,
                   'mask': mask, 'segment_all': segment_all, 'segment_obj': segment_obj}
        # Metrics
        metrics = self.compute_metrics(images, labels, pres, mask, mask_oh_all, mask_oh_obj, recon, raw_pixel_ll)
        losses['compare'] = -metrics['ll'] + step_losses['kld']
        return results, metrics, losses

    def compute_grid(self, scl, trs):
        batch_size = scl.shape[0]
        zeros = torch.zeros_like(scl[:, 0])
        theta_crop = torch.stack([
            torch.stack([scl[:, 0], zeros, trs[:, 0]], dim=1),
            torch.stack([zeros, scl[:, 1], trs[:, 1]], dim=1),
        ], dim=1)
        theta_full = torch.stack([
            torch.stack([1 / scl[:, 0], zeros, -trs[:, 0] / scl[:, 0]], dim=1),
            torch.stack([zeros, 1 / scl[:, 1], -trs[:, 1] / scl[:, 1]], dim=1),
        ], dim=1)
        grid_crop = nn_func.affine_grid(theta_crop, [batch_size, 1, *self.crop_shape[1:]], align_corners=False)
        grid_full = nn_func.affine_grid(theta_full, [batch_size, 1, *self.image_shape[1:]], align_corners=False)
        return grid_crop, grid_full

    @staticmethod
    def compute_mask(shp, zeta):
        x = shp * zeta[..., None, None]
        ones = torch.ones([1, *x.shape[1:]], device=x.device)
        return torch.cat([x, ones]) * torch.cat([ones, 1 - x]).cumprod(0)

    def compute_init_full_in(self, images, result_bck, result_obj):
        mask = self.compute_mask(result_obj['shp'], result_obj['zeta'])
        recon = (mask * torch.cat([result_obj['apc'], result_bck['bck'][None]])).sum(0)
        return torch.cat([images, recon, mask[-1]], dim=1).detach()

    def compute_upd_back_in(self, images, result_bck, result_obj):
        inputs_excl = self.compute_init_full_in(images, result_bck, result_obj)
        return torch.cat([inputs_excl, result_bck['bck']], dim=1).detach()

    def compute_upd_full_in(self, images, result_bck, result_obj, idx):
        mask_above = self.compute_mask(result_obj['shp'][:idx], result_obj['zeta'][:idx])
        mask_below = self.compute_mask(result_obj['shp'][idx + 1:], result_obj['zeta'][idx + 1:])
        mask_cur = self.compute_mask(result_obj['shp'][idx:idx + 1], result_obj['zeta'][idx:idx + 1])
        recon_above = (mask_above * torch.cat([result_obj['apc'][:idx], result_bck['bck'][None]])).sum(0)
        recon_below = (mask_below * torch.cat([result_obj['apc'][idx + 1:], result_bck['bck'][None]])).sum(0)
        recon_cur = (mask_cur * torch.cat([result_obj['apc'][idx:idx + 1], result_bck['bck'][None]])).sum(0)
        return torch.cat([images, recon_above, recon_below, mask_above[-1], recon_cur, mask_cur[-1]], dim=1).detach()

    def compute_upd_crop_in(self, result_bck, result_obj, upd_full_in, grid_crop, idx):
        excl_dims = self.image_shape[0] + 1
        inputs_excl = nn_func.grid_sample(upd_full_in[:, :-excl_dims], grid_crop, align_corners=False)
        bck_crop = nn_func.grid_sample(result_bck['bck'], grid_crop, align_corners=False)
        mask_cur = self.compute_mask(result_obj['shp_crop'][idx:idx + 1], result_obj['zeta'][idx:idx + 1])
        recon_cur = (mask_cur * torch.cat([result_obj['apc_crop'][idx:idx + 1], bck_crop[None]])).sum(0)
        return torch.cat([inputs_excl, recon_cur, mask_cur[-1]], dim=1).detach()

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
            result_obj[key] = torch.cat([val[:idx], update_dict[key][None], val[idx + 1:]])
        for key, val in states_dict.items():
            states_dict[key] = val[:idx] + [update_dict[key]] + val[idx + 1:]
        return

    def adjust_order(self, images, result_obj, states_dict, eps=1e-5):
        def permute(x):
            if x.dim() == 3:
                indices_expand = indices
            else:
                indices_expand = indices[..., None, None]
            x = torch.gather(x, 0, indices_expand.expand(-1, -1, *x.shape[2:]))
            return x
        sq_diffs = (result_obj['apc'] - images[None]).pow(2).sum(-3, keepdim=True).detach()
        visibles = result_obj['shp'].clone().detach()
        zeta = result_obj['zeta'].detach()
        coefs = torch.ones(visibles.shape[:-2], device=visibles.device)
        indices_list = []
        for _ in range(self.obj_slots):
            vis_sq_diffs = (visibles * sq_diffs).sum([-2, -1])
            vis_areas = visibles.sum([-2, -1])
            vis_max_vals = visibles.reshape(*visibles.shape[:-2], -1).max(-1).values
            scores = torch.exp(-0.5 * self.normal_invvar * vis_sq_diffs / (vis_areas + eps))
            scaled_scores = coefs * (vis_max_vals * zeta * scores + 1)
            indices = torch.argmax(scaled_scores, dim=0, keepdim=True)
            indices_list.append(indices)
            vis = torch.gather(visibles, 0, indices[..., None, None].expand(-1, -1, *visibles.shape[2:]))
            visibles *= 1 - vis
            coefs.scatter_(0, indices, -1)
        indices = torch.cat(indices_list)
        for key, val in result_obj.items():
            result_obj[key] = permute(val)
        for key, val in states_dict.items():
            states_0 = permute(torch.stack([n[0] for n in val]))
            states_1 = permute(torch.stack([n[1] for n in val]))
            states_dict[key] = [(n_0, n_1) for n_0, n_1 in zip(states_0, states_1)]
        return

    def compute_raw_pixel_ll(self, images, bck, apc):
        diff = torch.cat([apc, bck[None]]) - images[None]
        raw_pixel_ll = -0.5 * (self.normal_const + self.normal_invvar * diff.pow(2)).sum(-3, keepdim=True)
        return raw_pixel_ll

    def compute_kld(self, result_bck, result_obj):
        def compute_kld_normal(mu, logvar, prior_mu, prior_logvar, prior_invvar):
            return 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).pow(2) + logvar.exp()) - 1)
        # Presence
        tau1 = result_obj['tau1']
        tau2 = result_obj['tau2']
        zeta = result_obj['zeta']
        logits_zeta = result_obj['logits_zeta']
        psi1 = torch.digamma(tau1)
        psi2 = torch.digamma(tau2)
        psi12 = torch.digamma(tau1 + tau2)
        loss_pres_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - torch.lgamma(tau2) - self.prior_pres_log_alpha
        loss_pres_2 = (tau1 - self.prior_pres_alpha) * psi1
        loss_pres_3 = (tau2 - 1) * psi2
        loss_pres_4 = -(tau1 + tau2 - self.prior_pres_alpha - 1) * psi12
        log_zeta = nn_func.logsigmoid(logits_zeta)
        log1m_zeta = log_zeta - logits_zeta
        psi1_le_sum = psi1.cumsum(0)
        psi12_le_sum = psi12.cumsum(0)
        kappa1 = psi1_le_sum - psi12_le_sum
        psi1_lt_sum = torch.cat([torch.zeros([1, *psi1_le_sum.shape[1:]], device=zeta.device), psi1_le_sum[:-1]])
        logits_coef = psi2 + psi1_lt_sum - psi12_le_sum
        kappa2_list = []
        for idx in range(self.obj_slots):
            coef = torch.softmax(logits_coef[:idx + 1], dim=0)
            log_coef = nn_func.log_softmax(logits_coef[:idx + 1], dim=0)
            coef_le_sum = coef.cumsum(0)
            coef_lt_sum = torch.cat([torch.zeros([1, *coef_le_sum.shape[1:]], device=zeta.device), coef_le_sum[:-1]])
            part1 = (coef * psi2[:idx + 1]).sum(0)
            part2 = ((1 - coef_le_sum[:-1]) * psi1[:idx]).sum(0)
            part3 = -((1 - coef_lt_sum) * psi12[:idx + 1]).sum(0)
            part4 = -(coef * log_coef).sum(0)
            kappa2_list.append(part1 + part2 + part3 + part4)
        kappa2 = torch.stack(kappa2_list)
        loss_pres_5 = zeta * (log_zeta - kappa1) + (1 - zeta) * (log1m_zeta - kappa2)
        loss_pres = loss_pres_1 + loss_pres_2 + loss_pres_3 + loss_pres_4 + loss_pres_5
        loss_pres = loss_pres.sum([0, *range(2, loss_pres.dim())])
        # Back
        loss_bck = compute_kld_normal(result_bck['bck_mu'], result_bck['bck_logvar'], 0, 0, 1)
        loss_bck = loss_bck.sum([*range(1, loss_bck.dim())])
        # STN
        loss_stn = compute_kld_normal(result_obj['stn_mu'], result_obj['stn_logvar'],
                                      self.prior_stn_mu, self.prior_stn_logvar, self.prior_stn_invvar)
        loss_stn = loss_stn.sum([0, *range(2, loss_stn.dim())])
        # Objects
        loss_obj = compute_kld_normal(result_obj['obj_mu'], result_obj['obj_logvar'], 0, 0, 1)
        loss_obj = loss_obj.sum([0, *range(2, loss_obj.dim())])
        return loss_pres + loss_bck + loss_stn + loss_obj

    def compute_step_losses(self, images, result_bck, result_obj, mask, raw_pixel_ll):
        # Loss NLL
        loss_nll = -(mask * raw_pixel_ll).sum([0, *range(2, mask.dim())])
        # Loss reconstruction
        apc_all = torch.cat([result_obj['apc'], result_bck['bck'][None]])
        recon = (mask * apc_all).sum(0)
        loss_recon = 0.5 * (self.normal_const + self.normal_invvar * (recon - images).pow(2))
        loss_recon = loss_recon.sum([*range(1, loss_recon.dim())])
        # Loss KLD
        loss_kld = self.compute_kld(result_bck, result_obj)
        # Loss back prior
        bck_prior = images.reshape(*images.shape[:-2], -1).median(-1).values[..., None, None]
        sq_diff = (result_bck['bck'] - bck_prior).pow(2)
        loss_bck_prior = 0.5 * self.normal_invvar * sq_diff.sum([*range(1, sq_diff.dim())])
        # Loss back variance
        bck_var = result_bck['bck_res'].pow(2)
        loss_bck_var = 0.5 * self.normal_invvar * bck_var.sum([*range(1, bck_var.dim())])
        # Loss apc variance
        apc_var = result_obj['apc_crop_res'].pow(2)
        loss_apc_var = 0.5 * self.normal_invvar * apc_var.sum([0, *range(2, apc_var.dim())])
        # Losses
        losses = {'nll': loss_nll, 'recon': loss_recon, 'kld': loss_kld, 'bck_prior': loss_bck_prior,
                  'bck_var': loss_bck_var, 'apc_var': loss_apc_var}
        return losses

    @staticmethod
    def compute_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.dim() > 1:
                x = x.sum([*range(1, x.dim())])
            return x
        num_pixels = mask_true.sum([*range(1, mask_true.dim())])
        mask_true = mask_true.reshape(
            [mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[-2] * mask_true.shape[-1]])
        mask_pred = mask_pred.reshape(
            [mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[-2] * mask_pred.shape[-1]])
        mat = (mask_true * mask_pred).sum(-1)
        sum_row = mat.sum(1)
        sum_col = mat.sum(2)
        comb_mat = comb2(mat)
        comb_row = comb2(sum_row)
        comb_col = comb2(sum_col)
        comb_num = comb2(num_pixels)
        comb_prod = (comb_row * comb_col) / comb_num
        comb_mean = 0.5 * (comb_row + comb_col)
        diff = comb_mean - comb_prod
        score = (comb_mat - comb_prod) / diff
        invalid = ((comb_num == 0) + (diff == 0)) > 0
        score = torch.where(invalid, torch.ones_like(score), score)
        return score

    def compute_metrics(self, images, labels, pres, mask, mask_oh_all, mask_oh_obj, recon, raw_pixel_ll, eps=1e-10):
        # ARI
        ari_all = self.compute_ari(labels, mask_oh_all)
        ari_obj = self.compute_ari(labels, mask_oh_obj)
        # MSE
        sq_diff = (recon - images).pow(2)
        mse = sq_diff.mean([*range(1, sq_diff.dim())])
        # Log-likelihood
        pixel_ll = torch.logsumexp(mask.clamp(min=eps).log() + raw_pixel_ll, dim=1)
        ll = pixel_ll.sum([*range(1, pixel_ll.dim())])
        # Count
        pres_true = labels.reshape(*labels.shape[:-3], -1).max(-1).values
        if self.seg_bck:
            pres_true = pres_true[:, 1:]
        count_true = pres_true.sum(1)
        count_pred = pres.sum(1)
        count_acc = (count_true == count_pred).to(dtype=torch.float)
        metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics

    def compute_overview(self, images, results):
        def convert_single(x_in, color=None):
            x = nn_func.pad(x_in, [boarder_size] * 4, value=0)
            if color is not None:
                boarder = nn_func.pad(torch.zeros_like(x_in), [boarder_size] * 4, value=1) * color
                x += boarder
            x = nn_func.pad(x, [boarder_size] * 4, value=1)
            return x
        def convert_multiple(x, color=None):
            batch_size, num_slots = x.shape[:2]
            x = x.reshape(batch_size * num_slots, *x.shape[2:])
            if color is not None:
                color = color.reshape(batch_size * num_slots, *color.shape[2:])
            x = convert_single(x, color=color)
            x = x.reshape(batch_size, num_slots, *x.shape[1:])
            x = torch.cat(torch.unbind(x, dim=1), dim=-1)
            return x
        boarder_size = round(min(images.shape[-2:]) / 32)
        images = images.expand(-1, 3, -1, -1)
        recon = results['recon'].expand(-1, 3, -1, -1)
        apc = results['apc'].expand(-1, -1, 3, -1, -1)
        shp = results['shp'].expand(-1, -1, 3, -1, -1)
        pres = results['pres'][..., None, None, None].expand(-1, -1, 3, -1, -1)
        scl = results['scl'].reshape(-1, results['scl'].shape[-1])
        trs = results['trs'].reshape(-1, results['trs'].shape[-1])
        _, grid_full = self.compute_grid(scl, trs)
        white_crop = torch.ones([scl.shape[0], 3, *self.crop_shape[1:]], device=images.device)
        shp_obj_mask = nn_func.grid_sample(white_crop, grid_full, align_corners=False)
        shp_obj_mask = shp_obj_mask.reshape(*results['scl'].shape[:2], 3, *self.image_shape[1:])
        area_color = 1 - shp_obj_mask
        area_color[..., 0, :, :] *= 0.5
        area_color[..., 2, :, :] *= 0.5
        shp = torch.cat([shp[:, :-1] + area_color, shp[:, -1:]], dim=1)
        color_0 = torch.zeros_like(pres)
        color_1 = torch.zeros_like(pres)
        color_0[..., 1, :, :] = 0.5
        color_0[..., 2, :, :] = 1
        color_1[..., 0, :, :] = 1
        color_1[..., 1, :, :] = 0.5
        boarder_color = pres * color_1 + (1 - pres) * color_0
        boarder_color[:, -1] = 0
        row1 = torch.cat([convert_single(images), convert_multiple(apc)], dim=-1)
        row2 = torch.cat([convert_single(recon), convert_multiple(shp, color=boarder_color)], dim=-1)
        overview = torch.cat([row1, row2], dim=-2)
        overview = nn_func.pad(overview, [boarder_size * 4] * 4, value=1)
        overview = (overview.clamp_(0, 1) * 255).to(dtype=torch.uint8)
        return overview


def get_model(config):
    net = Model(config).cuda()
    return nn.DataParallel(net)
