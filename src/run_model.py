import h5py
import math
import os
import torch
import torch.optim as optim


def print_file(f, data=''):
    print(data)
    print(data, file=f)
    f.flush()


def compute_coef(epoch, num_epochs, epoch_list, ratio_list, linear_list):
    epoch_list = [0] + epoch_list + [num_epochs - 1]
    assert len(epoch_list) == len(ratio_list)
    assert len(epoch_list) == len(linear_list) + 1
    assert epoch_list == sorted(epoch_list)
    coef = None
    for idx in range(len(epoch_list) - 1):
        if epoch <= epoch_list[idx + 1]:
            ratio = (epoch - epoch_list[idx]) / (epoch_list[idx + 1] - epoch_list[idx])
            if linear_list[idx]:
                coef = (1 - ratio) * ratio_list[idx] + ratio * ratio_list[idx + 1]
            else:
                coef = math.exp((1 - ratio) * math.log(ratio_list[idx]) + ratio * math.log(ratio_list[idx + 1]))
            break
    return coef


def train_model(args, net, dataloaders):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_loss, best_epoch = float('inf'), -1
    with open(os.path.join(args.folder, args.file_log), 'w') as f:
        for epoch in range(args.num_epochs):
            print_file(f, 'Epoch {}/{}'.format(epoch, args.num_epochs - 1))
            print_file(f, '-' * 10)
            coef_dict = {key: compute_coef(epoch, args.num_epochs, val['epochs'], val['values'], val['linears'])
                         for key, val in args.coef_param_dict.items()}
            for phase in ['train', 'valid']:
                net.train(phase == 'train')
                sum_loss, num_data = 0, 0
                for images in dataloaders[phase]:
                    images = images.cuda()
                    batch_size = images.shape[0]
                    with torch.set_grad_enabled(phase == 'train'):
                        results = net(images)
                    results = [net.module.transform_result(n) for n in results]
                    batch_loss, weight = 0, 0
                    assert len(results) == len(args.loss_weights)
                    for result, sub_weight in zip(results, args.loss_weights):
                        sub_loss = net.module.compute_batch_loss(images, result, coef_dict)
                        batch_loss = batch_loss + sub_weight * sub_loss
                        weight += sub_weight
                    batch_loss /= weight
                    loss = batch_loss / batch_size
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    sum_loss += batch_loss.item()
                    num_data += batch_size
                mean_loss = sum_loss / num_data
                print_file(f, '{}\tLoss: {:.2f}'.format(phase.capitalize(), mean_loss))
                if phase == 'valid':
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        best_epoch = epoch
                        torch.save(net.module.state_dict(), os.path.join(args.folder, args.file_model))
                    print_file(f, 'Best\tLoss: {:.2f}\tEpoch: {}'.format(best_loss, best_epoch))
            print_file(f)


def test_model(args, net, dataloaders):
    net.train(False)
    for model_id in range(args.num_tests):
        result_list = {key: [] for key in ['scl', 'trs', 'pres', 'segre', 'recon_objects', 'recon_scene', 'back',
                                           'll_mixture', 'll_single']}
        for images in dataloaders['test']:
            images = images.cuda()
            with torch.set_grad_enabled(False):
                result = net(images)[-1]
            result = net.module.transform_result(result)
            pres = torch.bernoulli(result['zeta'])
            segre = net.module.compute_gamma(result['shp'], pres)
            shp_mul_pres = result['shp'] * pres[..., None, None]
            back = result['back']
            recon_objects = shp_mul_pres * result['apc'] + (1 - shp_mul_pres) * back[None]
            recon_scene = (segre * torch.cat([result['apc'], back[None]])).sum(0)
            ll_mixture, ll_single = net.module.compute_log_likelihood(images, result, segre, recon_scene)
            result_list['scl'].append(result['scl'].transpose(0, 1).cpu())
            result_list['trs'].append(result['trs'].transpose(0, 1).cpu())
            result_list['pres'].append(pres.squeeze(-1).transpose(0, 1).cpu())
            result_list['segre'].append(segre.transpose(0, 1).cpu())
            result_list['recon_objects'].append(recon_objects.transpose(0, 1).cpu())
            result_list['recon_scene'].append(recon_scene.cpu())
            result_list['back'].append(back.cpu())
            result_list['ll_mixture'].append(ll_mixture.cpu())
            result_list['ll_single'].append(ll_single.cpu())
        with h5py.File(os.path.join(args.folder, args.file_result_base.format(model_id)), 'w') as f:
            for key, val in result_list.items():
                f.create_dataset(key, data=torch.cat(val).numpy(), compression='gzip')
