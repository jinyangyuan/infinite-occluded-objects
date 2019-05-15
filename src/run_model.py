import h5py
import os
import torch
import torch.optim as optim


def train_model(args, net, dataloader):
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        if epoch < args.epoch_split_recon:
            ratio_recon = 0
        else:
            ratio_recon = (epoch - args.epoch_split_recon) / (args.num_epochs - args.epoch_split_recon - 1)
        if epoch < args.epoch_split_back:
            coef_back = args.coef_back * (1 - epoch / (args.epoch_split_back - 1))
        else:
            coef_back = 0
        for phase in ['train', 'valid']:
            net.train(phase == 'train')
            sum_loss, num_data = 0, 0
            for images, backgrounds in dataloader[phase]:
                images = images.cuda()
                backgrounds = backgrounds.cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    results = net(images, backgrounds)
                    loss, weight = 0, 0
                    for result, sub_weight in zip(results, args.loss_weights):
                        sub_loss = net.compute_loss(images, result, ratio_recon, coef_back)
                        loss = loss + sub_weight * sub_loss
                        weight += sub_weight
                    loss /= weight
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                sum_loss += loss.item() * images.shape[0]
                num_data += images.shape[0]
            mean_loss = sum_loss / num_data
            print('{}\tLoss: {:.4f}'.format(phase.capitalize(), mean_loss))
            if phase == 'valid' and mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(net.state_dict(), os.path.join(args.folder, args.file_model))
        print()
    print('Best Validation\tLoss: {:.4f}'.format(best_loss))


def test_model(args, net, dataloader):
    net.train(False)
    for model_id in range(args.num_tests):
        results_last = {key: [] for key in ['gamma', 'pres', 'shape', 'appear_combine']}
        for images, backgrounds in dataloader['test']:
            images = images.cuda()
            backgrounds = backgrounds.cuda()
            with torch.set_grad_enabled(False):
                sub_results_last = net(images, backgrounds)[-1]
            zeta = sub_results_last['zeta']
            shape = sub_results_last['shape']
            appear = sub_results_last['appear']
            back = sub_results_last['back']
            gamma = net.compute_gamma(zeta, shape)
            pres = torch.bernoulli(zeta)
            appear_combine = torch.cat([appear, back[None]])
            results_last['gamma'].append(gamma.data.transpose(0, 1).cpu())
            results_last['pres'].append(pres.data.transpose(0, 1).cpu())
            results_last['shape'].append(shape.data.transpose(0, 1).cpu())
            results_last['appear_combine'].append(appear_combine.data.transpose(0, 1).cpu())
        with h5py.File(os.path.join(args.folder, args.file_result_base.format(model_id)), 'w') as f:
            for key, val in results_last.items():
                f.create_dataset(key, data=torch.cat(val).numpy(), compression='gzip')
