import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
import torch
import os
import numpy

from src.dataset.data_loader import GMDataset, get_dataloader
from src.loss import construct_supervised_loss, construct_permutation_chain, \
                     construct_cycle_loss, compute_gradient_update
from src.evaluation_metric import matching_accuracy
from src.utils.general import load_model, save_model, data_to_cuda
from eval import eval_model
from src.lap.hungarian import hungarian



from src.utils.config import cfg
import ops

is_cuda = torch.cuda.is_available()
def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x


def train_and_eval(model,
                     optimizer,
                     dataloader,
                     num_epochs=25,
                     start_epoch=0):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)

    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    alphas = torch.tensor(cfg.EVAL.PCK_ALPHAS, dtype=torch.float32, device=device)  # for evaluation
    checkpoint_path = Path(cfg.OUTPUT_PATH) / ('params'+'_'+str(cfg.MATCHING_PROBLEM) + '_' + str(cfg.source_partial_kpt_len)+'_'+str(cfg.target_partial_kpt_len)+'_GConvNorma_'+str(cfg.crossgraph_s_normalization)+'_sample_'+str(cfg.samples_per_num_train)+now_time+'_'+str(cfg.PROBLEM.TYPE))
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path = '',''
    if start_epoch > 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    if len(cfg.PRETRAINED_PATH) > 0:
        model_path = cfg.PRETRAINED_PATH
    if len(model_path) > 0:
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path, strict=False)
    if len(optim_path) > 0:
        print('Loading optimizer state from {}'.format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        print("sigma_noise= ", str(cfg.sigma_norm))

        if cfg.PROBLEM.TYPE in ['MGM']:
            # if cfg.OPTIMIZATION_METHOD == 'Direct':
            # if cfg.penalty_method_on_cycle:
            cfg.lagrange_multiplier += cfg.penalty_epoch_increase
            print("lagrange_multiplier= ", str(cfg.lagrange_multiplier))

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader['train']:

            if model.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)

            '''
            inputs include:
            - Ps - list of C point-sets of size [B, N, 2]
            - ns - list of number of points of size [B] for each of the C point-sets.
            - Gs - list of connectivity matrices of size [B, N, 86]  (See fig.3 of NGMv2 paper)
            - Hs - list of connectivity matrices of size [B, N, 86]  (See fig.3 of NGMv2 paper)
            - As - list of adjacency matrices of size [B, N, N]      (See fig.3 of NGMv2 paper)
            For all Gs, Hs 1 indicates the presence of directed edge.
            Gs_tgt, Hs_tgt, As_tgt similar to Gs, Hs, As respectively
            - pyg_graphs - torch_geometric graph:
                -- x 
                -- edge_index
                -- edge_attr
                -- hyperedge_index
                -- batch
                -- ptr
            pyg_graphs_tgt similar to pyg_graphs
            - cls - contains class information of shape for each of the C graphs of shape [B]
            - univ_size - contains maximum number of nodes for each of the C graphs
            - KGHs - is a dictionary that contains sparse matrices for all pairs of graphs (including self)
            - 
            '''
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)

                if cfg.PROBLEM.TYPE == '2GM':
                    assert 'ds_mat' in outputs
                    assert 'perm_mat' in outputs
                    assert 'gt_perm_mat' in outputs


                    # compute loss
                    if cfg.TRAIN.LOSS_FUNC in ['perm']:

                        # if cfg.OPTIMIZATION_METHOD == 'Direct':  # direct optimization
                        pos_weight = torch.tensor(cfg.pos_weight)

                        if cfg.train_noise_factor:
                            sigma_tmp = to_var(torch.ones([outputs['ds_mat'].size()[0], 1], dtype=torch.float)) / cfg.sigma_norm
                            outputs['ds_mat'], _ = ops.my_phi_and_gamma_sigma_unbalanced(outputs['ds_mat'], cfg.samples_per_num_train,
                                                                                    cfg.train_noise_factor,
                                                                                    sigma_tmp)

                            # Solve a matching problem for a batch of matrices, if noise is added.
                            # tiled variables, to compare to many permutations
                            if cfg.samples_per_num_train > 1:
                                outputs['gt_perm_mat'] = outputs['gt_perm_mat'].repeat(cfg.samples_per_num_train, 1, 1)
                                outputs['ns'][0] = outputs['ns'][0].repeat(cfg.samples_per_num_train)
                                outputs['ns'][1] = outputs['ns'][1].repeat(cfg.samples_per_num_train)


                            outputs['perm_mat'] = hungarian(outputs['ds_mat'], outputs['ns'][0], outputs['ns'][1])

                        # calculate weighted bce loss without reduction
                        train_wbce_loss = construct_supervised_loss(outputs['perm_mat'], outputs['gt_perm_mat'], pos_weight)


                        gradient_update = compute_gradient_update(
                            outputs['ds_mat'], outputs['perm_mat'], train_wbce_loss, 1,  outputs['ns'][0], outputs['ns'][1])

                        # calculate loss to optimize encoder
                        gradient_update = (1. / 1.) * gradient_update

                        loss = torch.sum(outputs['ds_mat'] * to_var(gradient_update))

                    # compute accuracy
                    acc, _, __ = matching_accuracy(outputs['perm_mat'], outputs['gt_perm_mat'], outputs['ns'][0])


                elif cfg.PROBLEM.TYPE in ['MGM']:

                    # compute loss & accuracy
                    if cfg.TRAIN.LOSS_FUNC in ['perm']:

                        # if cfg.OPTIMIZATION_METHOD == 'Direct':  # direct optimization
                        pos_weight = torch.tensor(cfg.pos_weight)

                        loss = torch.zeros(1, device=model.device)
                        ns = outputs['ns'] #number of sampled nodes, each is (#keypoints at source graph, #keypoints at target graph)

                        graph_cycleparts_preds_all = []
                        for s_pred, perm_mat_pred, x_gt, (idx_src, idx_tgt) in \
                                zip(outputs['ds_mat_list'], outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):

                            if cfg.train_noise_factor:
                                sigma_tmp = to_var(torch.ones([s_pred.size()[0], 1],dtype=torch.float)) / cfg.sigma_norm
                                # s_pred, _ = my_ops.my_phi_and_gamma_sigma_unbalanced(s_pred, cfg.samples_per_num_train, cfg.train_noise_factor, sigma_tmp)
                                s_pred, _ = ops.perturb_scoreMatrix_unbalanced(s_pred, cfg.samples_per_num_train, cfg.train_noise_factor, sigma_tmp)

                                if cfg.samples_per_num_train > 1:
                                    x_gt = x_gt.repeat(cfg.samples_per_num_train, 1, 1)
                                    ns_src = ns[idx_src].repeat(cfg.samples_per_num_train)
                                    ns_trg = ns[idx_tgt].repeat(cfg.samples_per_num_train)

                                else:
                                    ns_src = ns[idx_src][0]
                                    ns_trg = ns[idx_tgt][0]

                                perm_mat = hungarian(s_pred, ns_src, ns_trg)

                            #no noise situation
                            else:
                                ns_src = ns[idx_src]
                                ns_trg = ns[idx_tgt]
                                perm_mat = perm_mat_pred

                            # calculate weighted bce loss without reduction
                            train_wbce_loss = construct_supervised_loss(perm_mat, x_gt, pos_weight)

                            # calculate 2step cycle consistency loss without reduction

                            graph_cycleparts_indices, graph_cycleparts_preds = construct_permutation_chain((idx_src, idx_tgt), outputs['graph_indices'], outputs['perm_mat_list'])
                            for p in range(len(graph_cycleparts_preds)):
                                graph_cycleparts_preds[p] = graph_cycleparts_preds[p].repeat(cfg.samples_per_num_train, 1, 1)

                            graph_cycleparts_preds_all.append(graph_cycleparts_preds[0])
                            cycle_consistency_loss = construct_cycle_loss(perm_mat, graph_cycleparts_preds, cfg.lagrange_multiplier)

                            if cfg.PROBLEM.UNSUPERVISED:
                                total_loss = cycle_consistency_loss
                            else:
                                total_loss = cycle_consistency_loss + train_wbce_loss

                            gradient_update = compute_gradient_update(s_pred, perm_mat, total_loss, 1, ns_src, ns_trg)

                            # calculate loss to optimize encoder
                            gradient_update = 1. * gradient_update

                            l = torch.sum(s_pred * to_var(gradient_update))
                            loss += l
                        loss /= len(outputs['ds_mat_list'])

                    else:
                        raise ValueError('Unsupported loss function {} for problem type {}'.format(cfg.TRAIN.LOSS_FUNC, cfg.PROBLEM.TYPE))

                    # compute accuracy
                    acc = torch.zeros(1, device=model.device)
                    for x_pred, x_gt, (idx_src, idx_tgt) in \
                            zip(outputs['perm_mat_list'], outputs['gt_perm_mat_list'], outputs['graph_indices']):
                        a, _, __ = matching_accuracy(x_pred, x_gt, ns[idx_src])
                        if cfg.PROBLEM.TYPE in ['MGM']:
                            acc += torch.mean(a)
                        else:
                            acc += torch.sum(a)
                    acc /= len(outputs['perm_mat_list'])

                    # compute cycle-consistency
                    # if cfg.OPTIMIZATION_METHOD == "Direct":
                    if cfg.PROBLEM.TYPE in ['MGM']:
                        cyc_const = torch.zeros(1, device=model.device)
                        for x_pred, x_2stepcycle, (idx_src, idx_tgt) in \
                                zip(outputs['perm_mat_list'], graph_cycleparts_preds_all, outputs['graph_indices']):
                            c, _, __ = matching_accuracy(x_pred, x_2stepcycle, ns[idx_src])
                            cyc_const += torch.mean(c)

                        cyc_const /= len(outputs['perm_mat_list'])

                else:
                    raise ValueError('Unknown problem type {}'.format(cfg.PROBLEM.TYPE))

                # backward + optimize
                loss.backward()

                optimizer.step()

                batch_num = inputs['batch_size']

                # tfboard writer
                loss_dict = dict()
                loss_dict['loss'] = loss.item()

                accdict = dict()
                accdict['matching accuracy'] = torch.mean(acc)

                # if cfg.OPTIMIZATION_METHOD == "Direct":
                if cfg.PROBLEM.TYPE in ['MGM']:
                    cycle_consistency_dict = dict()
                    cycle_consistency_dict['cycle_consistency accuracy'] = torch.mean(cyc_const)

                # statistics
                running_loss += loss.item() * batch_num
                epoch_loss += loss.item() * batch_num

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num))

                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size

        save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.8f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        accs = eval_model(model, alphas, dataloader['test'])
        acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs)}
        acc_dict['average'] = torch.mean(accs)
        # wb.save(wb.__save_path)

        scheduler.step()

        cfg.sigma_norm = cfg.sigma_norm*(1+cfg.sigma_decay)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    from src.utils.general import Logger, parse_args, print_easydict, count_parameters
    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    #cfg.PROBLEM.TYPE = 'MGM'
    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}

    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     #cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     obj_resize=cfg.PROBLEM.RESCALE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = Net()

    model.to(device)

    if cfg.TRAIN.SEPARATE_BACKBONE_LR:

        backbone_ids = [id(item) for item in model.backbone_params]
        other_params = [param for param in model.parameters() if id(param) not in backbone_ids]

        model_params = [
            #{'params': other_params, 'lr': 1.5*cfg.TRAIN.LR, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY},
            {'params': other_params},
            {'params': model.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
        ]
    else:
        model_params = model.parameters()

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(model_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(model_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

    model = model.cuda()

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_path = Path(cfg.OUTPUT_PATH) / ('logs'+'_'+str(cfg.MATCHING_PROBLEM)+'_'+str(cfg.source_partial_kpt_len)+'_'+str(cfg.target_partial_kpt_len)+'_GConv_normalization_'+str(cfg.crossgraph_s_normalization)+'_sample_'+str(cfg.samples_per_num_train)+'_'+str(cfg.PROBLEM.TYPE))
    if not log_path.exists():
        log_path.mkdir(parents=True)


    with Logger(os.path.join(log_path, 'train_log_' + now_time + '.log')) as _:
        print_easydict(cfg)
        print('Number of parameters: {:.2f}M'.format(count_parameters(model) / 1e6))
        model = train_and_eval(model, optimizer, dataloader,
                                 #num_epochs=10,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH)
