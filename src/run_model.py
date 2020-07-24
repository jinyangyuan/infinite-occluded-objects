import h5py
import math
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score


def compute_loss_coef(config, epoch):
    loss_coef = {'nll': 1, 'kld_bck': 1, 'kld_obj': 1, 'kld_stn': 1, 'kld_pres': 1, 'kld_mask': 1, 'recon': 0}
    for key, val in config['loss_coef'].items():
        epoch_list = [0] + val['epoch'] + [config['num_epochs'] - 1]
        assert len(epoch_list) == len(val['value'])
        assert len(epoch_list) == len(val['linear']) + 1
        assert epoch_list == sorted(epoch_list)
        for idx in range(len(epoch_list) - 1):
            if epoch <= epoch_list[idx + 1]:
                ratio = (epoch - epoch_list[idx]) / (epoch_list[idx + 1] - epoch_list[idx])
                val_1 = val['value'][idx]
                val_2 = val['value'][idx + 1]
                if val['linear'][idx]:
                    loss_coef[key] = (1 - ratio) * val_1 + ratio * val_2
                else:
                    loss_coef[key] = math.exp((1 - ratio) * math.log(val_1) + ratio * math.log(val_2))
                assert math.isfinite(loss_coef[key])
                break
        else:
            raise ValueError
    return loss_coef


def get_step_wt(config):
    if config['step_wt'] is None:
        step_wt = torch.ones([1, config['num_steps'] + 1])
    else:
        step_wt = torch.tensor([config['step_wt']]).reshape(1, config['num_steps'] + 1)
    return step_wt.cuda()


def train_model(config, data_loaders, net):
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    phase_list = ['train']
    save_phase = 'train'
    if 'valid' in data_loaders:
        phase_list.append('valid')
        save_phase = 'valid'
    path_ckpt = os.path.join(config['folder_out'], config['file_ckpt'])
    path_model = os.path.join(config['folder_out'], config['file_model'])
    if config['resume']:
        checkpoint = torch.load(path_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Resume training from epoch {}'.format(start_epoch))
    else:
        start_epoch = 0
        best_epoch = -1
        best_loss = float('inf')
        print('Start training')
    print()
    with SummaryWriter(log_dir=config['folder_log'], purge_step=start_epoch) as writer:
        for epoch in range(start_epoch, config['num_epochs']):
            loss_coef = compute_loss_coef(config, epoch)
            print('Epoch: {}/{}'.format(epoch, config['num_epochs'] - 1))
            for phase in phase_list:
                phase_param = config['phase_param'][phase]
                step_wt_base = get_step_wt(phase_param)
                net.train(phase == 'train')
                sum_losses, sum_metrics = {}, {}
                num_data = 0
                for idx_batch, data in enumerate(data_loaders[phase]):
                    data = {key: val.cuda(non_blocking=True) for key, val in data.items()}
                    batch_size = data['image'].shape[0]
                    step_wt = step_wt_base.expand(batch_size, -1)
                    if phase == 'train':
                        enable_grad = True
                        temp = loss_coef['temp']
                        hard = False
                    else:
                        enable_grad = False
                        temp = None
                        hard = True
                    with torch.set_grad_enabled(enable_grad):
                        results, metrics, losses = net(data['image'], data['label'], phase_param['num_slots'],
                                                       phase_param['num_steps'], step_wt, temp, hard)
                    for key, val in losses.items():
                        if key in sum_losses:
                            sum_losses[key] += val.sum().item()
                        else:
                            sum_losses[key] = val.sum().item()
                    for key, val in metrics.items():
                        if key in sum_metrics:
                            sum_metrics[key] += val.sum().item()
                        else:
                            sum_metrics[key] = val.sum().item()
                    num_data += batch_size
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss_opt = torch.stack(
                            [loss_coef[key] * val.mean() for key, val in losses.items() if key != 'compare']).sum()
                        loss_opt.backward()
                        optimizer.step()
                    if idx_batch == 0 and epoch % config['summ_image_intvl'] == 0:
                        overview = net.module.compute_overview(data['image'], results)
                        overview = torch.cat(torch.unbind(overview[:config['summ_image_count']], dim=0), dim=-2)
                        writer.add_image(phase.capitalize(), overview, global_step=epoch)
                        writer.flush()
                mean_losses = {key: val / num_data for key, val in sum_losses.items()}
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                if epoch % config['summ_scalar_intvl'] == 0:
                    for key, val in mean_losses.items():
                        writer.add_scalar('{}/loss_{}'.format(phase.capitalize(), key), val, global_step=epoch)
                    for key, val in mean_metrics.items():
                        writer.add_scalar('{}/metric_{}'.format(phase.capitalize(), key), val, global_step=epoch)
                    writer.flush()
                print(phase.capitalize())
                print((' ' * 4).join([
                    'ARI_A: {:.3f}'.format(mean_metrics['ari_all']),
                    'ARI_O: {:.3f}'.format(mean_metrics['ari_obj']),
                    'MSE: {:.2e}'.format(mean_metrics['mse']),
                    'LL: {:.1f}'.format(mean_metrics['ll']),
                    'Count: {:.3f}'.format(mean_metrics['count']),
                ]))
                if phase == save_phase:
                    if mean_losses['compare'] < best_loss:
                        best_loss = mean_losses['compare']
                        best_epoch = epoch
                        torch.save(net.state_dict(), path_model)
            save_dict = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_dict, path_ckpt)
            print('Best Epoch: {}'.format(best_epoch))
            print()
    return


def test_model(config, data_loaders, net):
    def get_path_save():
        return os.path.join(config['folder_out'], '{}.h5'.format(phase))
    def compute_ami(seg_true_list, seg_pred_list, seg_valid_list):
        ami_list = []
        for seg_true, seg_pred, seg_valid in zip(seg_true_list, seg_pred_list, seg_valid_list):
            ami_list.append(adjusted_mutual_info_score(seg_true[seg_valid], seg_pred[seg_valid], average_method='max'))
        return torch.tensor(ami_list)
    def compute_order(cost_list):
        order_list = []
        for cost in cost_list:
            _, cols = linear_sum_assignment(cost)
            order_list.append(cols)
        return np.array(order_list)
    def compute_ooa(layers_list, order_list):
        objects_rgb, objects_a = layers_list[:, 1:, :-1], layers_list[:, 1:, -1]
        weights = np.zeros((objects_a.shape[0], objects_a.shape[1], objects_a.shape[1]))
        for i in range(objects_a.shape[1] - 1):
            for j in range(i + 1, objects_a.shape[1]):
                sq_diffs = np.square(objects_rgb[:, i] - objects_rgb[:, j]).sum(-3)
                sq_diffs *= objects_a[:, i] * objects_a[:, j]
                weights[:, i, j] = sq_diffs.reshape(sq_diffs.shape[0], -1).sum(-1)
        binary_mat = np.zeros(weights.shape)
        for i in range(order_list.shape[1] - 1):
            for j in range(i + 1, order_list.shape[1]):
                binary_mat[:, i, j] = order_list[:, i] > order_list[:, j]
        sum_scores = (binary_mat * weights).sum().astype(np.float32)
        sum_weights = weights.sum().astype(np.float32)
        return sum_scores, sum_weights
    def compute_layer_mse(layers_list, apc_list, order_list):
        objects_rgb, objects_a = layers_list[:, 1:, :-1], layers_list[:, 1:, -1]
        apc_sel = np.array([val[idx] for val, idx in zip(apc_list, order_list)])
        sq_diffs = np.square(apc_sel - objects_rgb).mean(-3)
        sum_scores = (sq_diffs * objects_a).sum().astype(np.float32)
        sum_weights = objects_a.sum().astype(np.float32)
        return sum_scores, sum_weights
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    for phase in phase_list:
        path_save = get_path_save()
        if os.path.exists(path_save):
            raise FileExistsError(path_save)
    path_model = os.path.join(config['folder_out'], config['file_model'])
    net.load_state_dict(torch.load(path_model))
    net.train(False)
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        step_wt_base = get_step_wt(phase_param)
        path_save = get_path_save()
        data_key = phase_param['key'] if 'key' in phase_param else phase
        with h5py.File(path_save, 'a') as f:
            if config['save_detail']:
                f.create_group('detail')
            all_metrics = {}
            for idx_run in range(config['num_tests']):
                sum_metrics, sum_metrics_extra = {}, {}
                num_data = 0
                for data in data_loaders[data_key]:
                    data = {key: val if key == 'layers' else val.cuda(non_blocking=True) for key, val in data.items()}
                    batch_size = data['image'].shape[0]
                    step_wt = step_wt_base.expand(batch_size, -1)
                    with torch.set_grad_enabled(False):
                        results, metrics, _ = net(
                            data['image'], data['label'], phase_param['num_slots'], phase_param['num_steps'], step_wt)
                    segment_true = data['label'].argmax(1).data.cpu().numpy()
                    segment_valid = (data['label'].sum(1) != 0).data.cpu().numpy()
                    segment_pred_all = results['segment_all'].squeeze(1).data.cpu().numpy()
                    segment_pred_obj = results['segment_obj'].squeeze(1).data.cpu().numpy()
                    metrics['ami_all'] = compute_ami(segment_true, segment_pred_all, segment_valid)
                    metrics['ami_obj'] = compute_ami(segment_true, segment_pred_obj, segment_valid)
                    mask_true = data['label']
                    if config['seg_bck']:
                        mask_true = mask_true[:, 1:]
                    mask_pred = results['mask'][:, :-1]
                    metrics_extra = {}
                    if 'layers' in data and mask_true.shape[1] <= mask_pred.shape[1]:
                        order_cost = -(mask_true[:, :, None] * mask_pred[:, None]).sum([-3, -2, -1]).data.cpu().numpy()
                        order = compute_order(order_cost)
                        layers = data['layers'].numpy()
                        metrics_extra['order'] = compute_ooa(layers, order)
                        apc = results['apc'].data.cpu().numpy()
                        metrics_extra['layer_mse'] = compute_layer_mse(layers, apc, order)
                    for key, val in metrics.items():
                        if key in sum_metrics:
                            sum_metrics[key] += val.sum().item()
                        else:
                            sum_metrics[key] = val.sum().item()
                    for key, val in metrics_extra.items():
                        if key in sum_metrics_extra:
                            sum_metrics_extra[key][0] += val[0]
                            sum_metrics_extra[key][1] += val[1]
                        else:
                            sum_metrics_extra[key] = list(val)
                    num_data += batch_size
                    if idx_run == 0 and config['save_detail']:
                        for key in ['apc', 'shp', 'pres']:
                            val = (results[key].data.clamp_(0, 1).mul_(255)).to(dtype=torch.uint8).cpu().numpy()
                            if key in f['detail']:
                                f['detail'][key].resize(f['detail'][key].shape[0] + val.shape[0], axis=0)
                                f['detail'][key][-val.shape[0]:] = val
                            else:
                                f['detail'].create_dataset(
                                    key, data=val, maxshape=[None, *val.shape[1:]], compression='gzip')
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                mean_metrics.update({key: val[0] / val[1] for key, val in sum_metrics_extra.items()})
                for key, val in mean_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(val)
                    else:
                        all_metrics[key] = [val]
            f.create_group('metric')
            for key, val in all_metrics.items():
                f['metric'].create_dataset(key, data=np.array(val, dtype=np.float32))
            metrics_mean = {key: np.mean(val) for key, val in all_metrics.items()}
            metrics_std = {key: np.std(val) for key, val in all_metrics.items()}
            format_list = [
                ('ARI_All', '3f'), ('ARI_Obj', '3f'), ('AMI_All', '3f'), ('AMI_Obj', '3f'),
                ('Count', '3f'), ('Order', '3f'), ('LL', '1f'), ('MSE', '2e'), ('Layer_MSE', '2e'),
            ]
            print(phase)
            for name, mean_fmt in format_list:
                key = name.lower()
                if key in all_metrics:
                    print(name)
                    print(('Mean: {:.' + mean_fmt + '}').format(metrics_mean[key]))
                    print('Std:  {:.2e}'.format(metrics_std[key]))
                    print()
    return
