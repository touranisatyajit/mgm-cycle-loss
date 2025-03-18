import torch
import itertools

from models.affinity_layer import InnerProductWithWeightsAffinity
from models.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from models.transformer.Layers import EncoderLayer, DecoderLayer

from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat
from src.utils.general import pad_tensor
from models.gnn import GNNLayer
from src.lap.sinkhorn import Sinkhorn
from src.lap.hungarian import hungarian
import torch_geometric
import numpy
from scipy.spatial import distance


from src.utils.config import cfg
import torch.nn.functional

from src.backbone import *


is_cuda = torch.cuda.is_available()
def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x
CNN = eval(cfg.BACKBONE)

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)

def _normalize_coordinates(points: torch.Tensor):
    """
    Normalize coordinates in a batched manner.
    
    Args:
        points (torch.Tensor): Input tensor of shape [B, N, 2].
    
    Returns:
        torch.Tensor: Normalized coordinates of shape [B, N, 2].
    """
    # Compute center for each batch
    center = points.mean(dim=1, keepdim=True)  # Shape: [B, 1, 2]
    norm_points = points - center  # Centered points
    
    # Compute pairwise distance for each batch using broadcasting
    B, N, _ = points.shape
    max_dists = []
    for b in range(B):
        dist_matrix = distance.cdist(points[b].cpu().numpy(), points[b].cpu().numpy())
        max_dists.append(dist_matrix.max())
    
    max_dists = torch.tensor(max_dists, device=points.device).view(B, 1, 1)  # Shape: [B, 1, 1]
    
    norm_points = norm_points / max_dists  # Normalize by max distance
    
    return norm_points


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=self.mgm_tau)
        self.gnn_layer = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(1, 1,
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)

            self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)


        self.encoder = EncoderLayer(d_model=1024, d_inner=1024, n_head=8, d_k=1024, d_v=1024,
                                    atten_mode="softmax")
        # self.decoder = DecoderLayer(d_model=1024, d_inner=1024, n_head=8, d_k=1024, d_v=1024,
        #                             atten_mode="sinkhorn")
        self.pos_embedding_layer = nn.Linear(2, 1024)

    def _normalize_coordinates(self, points: torch.Tensor):
        """
        Normalize coordinates in a batched manner.
        
        Args:
            points (torch.Tensor): Input tensor of shape [B, N, 2].
        
        Returns:
            torch.Tensor: Normalized coordinates of shape [B, N, 2].
        """
        # Compute center for each batch
        center = points.mean(dim=1, keepdim=True)  # Shape: [B, 1, 2]
        norm_points = points - center  # Centered points
        
        # Compute pairwise distance for each batch using broadcasting
        B, N, _ = points.shape
        max_dists = []
        for b in range(B):
            dist_matrix = distance.cdist(points[b].cpu().numpy(), points[b].cpu().numpy())
            max_dists.append(dist_matrix.max())
        
        max_dists = torch.tensor(max_dists, device=points.device).view(B, 1, 1).to(torch.float32)  # Shape: [B, 1, 1]
        
        norm_points = norm_points / max_dists  # Normalize by max distance
        
        return norm_points
    
    def NGM_QAP(self, unary_affs_list, quadratic_affs_list, num_graphs, data_dict, \
                n_points, points):
        s_list, x_list, indices = [], [], []
        # row_x_list = []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
            Kp = torch.stack(pad_tensor(unary_affs), dim=0)
            Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
            K = construct_aff_mat(Ke, Kp, kro_G, kro_H)


            if num_graphs == 2: data_dict['aff_mat'] = K

            if cfg.NGM.FIRST_ORDER:
                emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
            else:
                emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

            if cfg.NGM.POSITIVE_EDGES:
                A = (K > 0).to(K.dtype)
            else:
                A = (K != 0).to(K.dtype)

            emb_K = K.unsqueeze(-1)

            # NGM qap solver
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)

            if cfg.normalize_gnn_outputs_channles:
                v = normalize_over_channels(v)

            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)


            if cfg.OPTIMIZATION_METHOD == 'Direct':

                if cfg.MATCHING_PROBLEM == 'Balanced':
                    ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
                elif cfg.MATCHING_PROBLEM == 'Unbalanced':
                    ss = s

            x = hungarian(ss, n_points[idx1], n_points[idx2])

            s_list.append(ss)
            x_list.append(x)

            indices.append((idx1, idx2))

        return s_list, x_list, indices

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['batch_size']
        num_graphs = len(images)

        #Normalizing coordinates
        points = [self._normalize_coordinates(cur_points) for cur_points in points]

        pos_embeddings = [self.pos_embedding_layer(cur_points) for cur_points in points]

        if cfg.PROBLEM.TYPE == '2GM' and 'gt_perm_mat' in data_dict:
            gt_perm_mats = [data_dict['gt_perm_mat']]
        elif cfg.PROBLEM.TYPE == 'MGM' and 'gt_perm_mat' in data_dict:

            if cfg.MATCHING_PROBLEM == "Balanced":
                perm_mat_list = data_dict['gt_perm_mat']
                gt_perm_mats = [torch.bmm(pm_src, pm_tgt.transpose(1, 2)) for pm_src, pm_tgt in lexico_iter(perm_mat_list)]
            elif cfg.MATCHING_PROBLEM == "Unbalanced":
                gt_perm_mats = data_dict['gt_perm_mat']


        else:
            raise ValueError('Ground truth information is required during training.')

        global_list = []
        orig_graph_list = []
        for image, p, n_p, graph, pos_embed in zip(images, points, n_points, graphs, pos_embeddings):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))

            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features(old code)
            # U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            # F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            # node_features = torch.cat((U, F), dim=1)
            # graph.x = node_features

            #Added node transformer here
            max_elements = torch.max(n_p)
            atten_mat_list = torch.zeros((nodes.shape[0], max_elements, max_elements))
            for idx in range(len(n_p)):
                atten_mat_list[idx, :n_p[idx], :n_p[idx]] = 1 
            U = feature_align(nodes, p, n_p, self.rescale)
            F = feature_align(edges, p, n_p, self.rescale)
            node_features = torch.cat((U, F), dim=1)
            node_features = self.encoder(enc_input=node_features.permute(0, 2, 1).cuda(), slf_attn_mask=atten_mat_list.cuda())[0]
            node_features = node_features + pos_embed
            graph.x = concat_features(node_features.permute(0, 2, 1), n_p)
            

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)


            #Added edge transformer here
            # edge_sizes = []
            # edge_matrices = []
            
            # for idx in range(len(orig_graph)):
            #     edge_matrices.append(orig_graph[idx].edge_attr)
            #     edge_sizes.append(orig_graph[idx].edge_attr.shape[0])
            # max_edges = max(edge_sizes)
            # atten_mat_list_edges = torch.zeros((len(orig_graph), max_edges, max_edges))
            # for idx in range(len(orig_graph)):
            #     atten_mat_list_edges[idx, :orig_graph[idx].edge_attr.shape[0], :orig_graph[idx].edge_attr.shape[0]] = 1
            # padded_tensors = [
            # torch.nn.functional.pad(tensor, (0, 0, 0, max_edges - tensor.shape[0])) for tensor in edge_matrices
            # ]
            # edge_matrices = torch.stack(padded_tensors)
            # edge_features = self.encoder(enc_input=edge_matrices.cuda(), slf_attn_mask=atten_mat_list_edges.cuda())[0]
            # for idx in range(len(orig_graph)):
            #     orig_graph[idx].edge_attr = edge_features[idx, :edge_sizes[idx], :]
            #orig_graph_list.append(orig_graph) #comment the same call made above if this is uncommented

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        #computing unaries like BBGM
        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        #computing edge costs like BBGM
        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []
        #shifting the below for loop to a seperate function
        s_list, x_list, indices = self.NGM_QAP(unary_affs_list, quadratic_affs_list, \
                                               num_graphs, data_dict, n_points, points)


        if num_graphs > 2:
            
            if cfg.OPTIMIZATION_METHOD == 'Direct':
                mgm_x_list = x_list
                mgm_s_list = s_list
                

        if cfg.PROBLEM.TYPE == '2GM':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0]
            })
            
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
                'gt_perm_mat_list': gt_perm_mats
            })

        return data_dict
