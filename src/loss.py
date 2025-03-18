import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import cfg
import numpy
from src.lap.hungarian import hungarian


def to_var(x):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        x = x.cuda()
    return x

def construct_supervised_loss(matching_permutation_matrix, matching_gt_tiled, pos_weight):
    loss = torch.nn.BCELoss(reduction='none')
    loss_output = loss(to_var(matching_permutation_matrix.float()), to_var(matching_gt_tiled.float()))
    loss_output += pos_weight * loss_output * to_var(matching_gt_tiled.float())
    return loss_output

def construct_permutation_chain(current_graph, graph_indices, pred_perm_mats):

    current_source_graph = current_graph[0]
    current_target_graph = current_graph[1]

    graph_indices_transpose = [tuple(numpy.flip(ind)) for ind in graph_indices]
    pred_perm_mat_transpose = [torch.transpose(pred_perm_mat,1,2) for pred_perm_mat in pred_perm_mats]

    graph_cycleparts = [((g_ind_tup_a, g_ind_tup_b), torch.matmul(pred_perm_mat_a,pred_perm_mat_b)) for g_ind_tup_a,pred_perm_mat_a in zip(graph_indices,pred_perm_mats) for g_ind_tup_b,pred_perm_mat_b in zip(graph_indices,pred_perm_mats) if g_ind_tup_a[0] == current_source_graph and g_ind_tup_b[1] == current_target_graph and g_ind_tup_a[1] == g_ind_tup_b[0] and g_ind_tup_a[1] != current_target_graph]
    graph_cycleparts_t_b = [((g_ind_tup_a, g_ind_tup_b),torch.matmul(pred_perm_mat_a,pred_perm_mat_b))  for g_ind_tup_a,pred_perm_mat_a in zip(graph_indices, pred_perm_mats) for g_ind_tup_b,pred_perm_mat_b in zip(graph_indices_transpose,pred_perm_mat_transpose) if g_ind_tup_a[0] == current_source_graph and g_ind_tup_b[1] == current_target_graph and g_ind_tup_a[1] == g_ind_tup_b[0] and g_ind_tup_a[1] != current_target_graph]
    graph_cycleparts_t_a = [((g_ind_tup_a, g_ind_tup_b),torch.matmul(pred_perm_mat_a,pred_perm_mat_b)) for g_ind_tup_a, pred_perm_mat_a in zip(graph_indices_transpose,pred_perm_mat_transpose) for g_ind_tup_b, pred_perm_mat_b in zip(graph_indices,pred_perm_mats) if g_ind_tup_a[0] == current_source_graph and g_ind_tup_b[1] == current_target_graph and g_ind_tup_a[1] == g_ind_tup_b[0] and g_ind_tup_a[1] != current_target_graph]

    graph_cycleparts_indices_tuples = [t[0] for t in graph_cycleparts] + [t[0] for t in graph_cycleparts_t_b] + [t[0] for t in graph_cycleparts_t_a]
    graph_cycleparts_preds = [t[1] for t in graph_cycleparts] + [t[1] for t in graph_cycleparts_t_b] + [t[1] for t in graph_cycleparts_t_a]

    return graph_cycleparts_indices_tuples, graph_cycleparts_preds

def construct_cycle_loss(focal_perm_mat_pred, graph_cycleparts_preds, lagrange_multiplier):
    all_cycles_loss = to_var(torch.zeros(focal_perm_mat_pred.size()))
    for cyc in graph_cycleparts_preds:
        cycle_loss_c = lagrange_multiplier*torch.where(focal_perm_mat_pred < cyc, cyc, to_var(torch.zeros(1)))
        all_cycles_loss += cycle_loss_c

    return all_cycles_loss

def compute_gradient_update(perturbed_score_matrix, log_alpha_w_noise_permutation_matrix, train_wbce_loss, samples_per_num_train, n1_gt, n2_gt):
    with torch.no_grad():
        perturbed_score_matrix_w_bce = perturbed_score_matrix.clone()

        perturbed_score_matrix_minus_bce = perturbed_score_matrix.clone()  ###two sided

        reattempt = True
        while reattempt:
            # associate the perturbation to its correlated position in log_alpha_w_noise according to the ground truth permutation
            perturbed_score_matrix_w_bce += cfg.loss_epsilon * train_wbce_loss
            perturbed_score_matrix_minus_bce -= cfg.loss_epsilon * train_wbce_loss  ###two sided

            # Solve a matching problem for a batch of matrices.
            perturbed_perm_matrix_w_bce = hungarian(perturbed_score_matrix_w_bce, n1_gt, n2_gt)
            perturbed_perm_matrix_minus_bce = hungarian(perturbed_score_matrix_minus_bce, n1_gt, n2_gt) ###two sided

            #gradient_update = (-1)*perturbed_perm_matrix_w_bce + perturbed_perm_matrix_minus_bce
            gradient_update = (-1)*perturbed_perm_matrix_w_bce + perturbed_perm_matrix_minus_bce ###two sided
            gradient_update = gradient_update.type(torch.float)
            batch_size = perturbed_score_matrix.size()[0]
            if torch.all(torch.eq(gradient_update, to_var(torch.zeros([batch_size, gradient_update.size()[1], gradient_update.size()[2]])))) and torch.sum(train_wbce_loss) > 0.:
                cfg.loss_epsilon *= 1.1

                print("*************************zero gradients loss positive")
                print("*********increasing epsilon by 10%")
                reattempt = False
            else:
                reattempt = False

        return gradient_update
