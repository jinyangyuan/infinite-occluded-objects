import h5py
import math
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from metric import compute_ami, compute_order, select_by_order, compute_ooa, compute_layer_mse, compute_iou_f1


def compute_loss_coef(config, epoch):
    loss_coef = {'nll': 1, 'kld_bck': 1, 'kld_obj': 1, 'kld_stn': 1, 'kld_pres': 1, 'kld_mask': 1}
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
                        ratio_mixture = loss_coef['ratio_mixture']
                        temp_pres = loss_coef['temp_pres']
                        temp_shp = loss_coef['temp_shp']
                        hard = False
                    else:
                        enable_grad = False
                        ratio_mixture = 1
                        temp_pres = None
                        temp_shp = None
                        hard = True
                    with torch.set_grad_enabled(enable_grad):
                        results, metrics, losses = net(
                            data['image'], data['segment'], data['overlap'], phase_param['num_slots'],
                            phase_param['num_steps'], step_wt, ratio_mixture, temp_pres, temp_shp, hard,
                        )
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
        with h5py.File(path_save, 'w') as f:
            all_metrics = {}
            for idx_run in range(config['num_tests']):
                if idx_run == 0 and config['save_detail']:
                    details = {key: [] for key in ['apc', 'shp', 'pres']}
                else:
                    details = None
                sum_metrics, sum_metrics_extra = {}, {}
                num_data = 0
                for data in data_loaders[data_key]:
                    data = {key: val if key == 'layers' else val.cuda(non_blocking=True) for key, val in data.items()}
                    batch_size = data['image'].shape[0]
                    step_wt = step_wt_base.expand(batch_size, -1)
                    with torch.set_grad_enabled(False):
                        results, metrics, _ = net(data['image'], data['segment'], data['overlap'],
                                                  phase_param['num_slots'], phase_param['num_steps'], step_wt)
                    data = {key: val.data.cpu().numpy() for key, val in data.items()}
                    results = {key: val.data.cpu().numpy() for key, val in results.items()}
                    segment_true = data['segment'][:, :-1].argmax(1)
                    if config['seg_overlap']:
                        segment_valid = data['segment'][:, -1] == 0
                    else:
                        segment_valid = (data['segment'][:, -1] + data['overlap']) == 0
                    segment_pred_all = results['segment_all'].squeeze(1)
                    segment_pred_obj = results['segment_obj'].squeeze(1)
                    metrics['ami_all'] = compute_ami(segment_true, segment_pred_all, segment_valid)
                    metrics['ami_obj'] = compute_ami(segment_true, segment_pred_obj, segment_valid)
                    metrics_extra = {}
                    if 'layers' in data:
                        shp_true = data['layers'][:, :, -1:]
                        part_cumprod = np.concatenate([
                            np.ones((shp_true.shape[0], 1, *shp_true.shape[2:]), dtype=shp_true.dtype),
                            np.cumprod(1 - shp_true[:, :-1], 1),
                        ], axis=1)
                        mask_true = shp_true * part_cumprod
                    else:
                        mask_true = data['segment']
                    mask_pred = results['mask']
                    if not config['seg_overlap']:
                        mask_true *= 1 - data['overlap'][:, None]
                        mask_pred *= 1 - data['overlap'][:, None]
                    if 'layers' in data:
                        region_order_true = shp_true
                        region_order_pred = results['shp']
                    else:
                        region_order_true = mask_true
                        region_order_pred = mask_pred
                    order_cost = -(region_order_true[:, :-1, None] * region_order_pred[:, None, :-1])
                    order_cost = order_cost.reshape(*order_cost.shape[:-3], -1).sum(-1)
                    order = compute_order(order_cost)
                    mask_sel = select_by_order(mask_pred, order)
                    metrics_extra['iou_part'], metrics_extra['f1_part'] = compute_iou_f1(mask_true, mask_sel)
                    if 'layers' in data:
                        layers = data['layers']
                        apc_sel = select_by_order(results['apc'], order)
                        shp_sel = select_by_order(results['shp'], order)
                        metrics_extra['iou_full'], metrics_extra['f1_full'] = compute_iou_f1(shp_true, shp_sel)
                        metrics_extra['order'] = compute_ooa(layers, order)
                        metrics_extra['layer_mse'] = compute_layer_mse(layers, apc_sel, shp_sel)
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
                    if details is not None:
                        for key, val in details.items():
                            val.append((np.clip(results[key], 0, 1) * 255).astype(np.uint8))
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                mean_metrics.update({key: val[0] / val[1] for key, val in sum_metrics_extra.items()})
                for key, val in mean_metrics.items():
                    if key in all_metrics:
                        all_metrics[key].append(val)
                    else:
                        all_metrics[key] = [val]
                if details is not None:
                    f.create_group('detail')
                    for key, val in details.items():
                        f['detail'].create_dataset(key, data=np.concatenate(val), compression='gzip')
            f.create_group('metric')
            for key, val in all_metrics.items():
                f['metric'].create_dataset(key, data=np.array(val, dtype=np.float32))
            metrics_mean = {key: np.mean(val) for key, val in all_metrics.items()}
            metrics_std = {key: np.std(val) for key, val in all_metrics.items()}
            format_list = [
                ('ARI_All', '3f'), ('ARI_Obj', '3f'), ('AMI_All', '3f'), ('AMI_Obj', '3f'),
                ('IOU_Full', '3f'), ('IOU_Part', '3f'), ('F1_Full', '3f'), ('F1_Part', '3f'),
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
