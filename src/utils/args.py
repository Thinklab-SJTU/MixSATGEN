from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    
    # basic model
    # mode: test == generate, train == finetune.
    parser.add_argument('--mode', dest='mode', default='test',
                        type=str, help='If use matching model, train is needed for finetune. Options: test, train')
    parser.add_argument('--matching', dest='matching', default='satbench',
                        type=str, help='Model / algorithm used for matching. Options: rrwm, satbench (default)')
    parser.add_argument('--mixing', dest='mixing', default='clause',
                        type=str, help='Method used for mixing. Options: clause (default)')
    parser.add_argument('--graph', dest='graph', default='lig',
                        type=str, help='The type of graph used in matching. Options: lig (default), vig')
    
    # global detail
    parser.add_argument('--dtype', dest='dtype', default='float32',
                        type=str, help='Options: float16, float32, float64 (exist in both torch and numpy)')
    parser.add_argument('--device', dest='device', default='cuda:0',
                        type=str, help='Options: cpu, cuda:0, cuda:1, ... (will be check by torch.cuda.is_available)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # file
    parser.add_argument('--dataset', dest='dataset', default='easy',
                        type=str, help='directory name under ./dataset/')
    parser.add_argument('--load_K', dest='load_K', action='store_true',
                        help='whether load K i.e.affinity matrix. If there is no existing affinity matrix, a new one will be built')
    
    # matching finetune
    parser.add_argument('--optim', dest='optim', default='Adam',
                        type=str, help='Optimizer for graph matching nn solver. Options: Adam, RMSprop, SGD')
    parser.add_argument('--epoch', dest='epoch', default=5, type=int,
                        help='max epoch num for matching nn solver training (finetune)')
    parser.add_argument('--lr', dest='lr', default=0.01, type=float,
                        help='learning rate for matching nn solver training')
    parser.add_argument('--pretrain', type=str, default=None, 
                    help='Pretrained model path to be finetuned')
    parser.add_argument('--sinkhorn_tau', dest='sinkhorn_tau', default=0.1, type=float,
                        help='param tau for pygmtools.sinkhorn in xKx loss')
    parser.add_argument('--sinkhorn_iter', dest='sinkhorn_iter', default=10, type=int,
                        help='param max_iter for pygmtools.sinkhorn in xKx loss')
    
    # gumbel config
    parser.add_argument('--enable_gumbel', dest='enable_gumbel', action='store_true',
                        help='whether enable gumbel sinkhorn')
    parser.add_argument('--gumbel_factor', dest='gumbel_factor', default=1, type=float,
                        help='param factor (lambda) for gumbel noise')
    parser.add_argument('--gumbel_temp', dest='gumbel_temp', default=1, type=float,
                        help='param temperature (tau) for gumbel noise')
    
    # satbench config
    parser.add_argument('--satbench_graph', type=str, choices=['lcg', 'vcg'], default='lcg', help='Graph construction')
    parser.add_argument('--satbench_init_emb', type=str, choices=['learned', 'random'], default='learned', help='Embedding initialization')
    parser.add_argument('--satbench_model', type=str, choices=['neurosat', 'ggnn', 'gcn', 'gin'], default='neurosat', help='GNN model')
    parser.add_argument('--satbench_dim', type=int, default=128, help='Dimension of embeddings and hidden states')
    parser.add_argument('--satbench_n_iterations', type=int, default=32, help='Number of iterations for message passing')
    parser.add_argument('--satbench_n_mlp_layers', type=int, default=2, help='Number of layers in all MLPs')
    parser.add_argument('--satbench_activation', type=str, default='relu', help='Activation function in all MLPs')
    
    # mixing config
    parser.add_argument('--entropy', dest='entropy', action='store_true',
                        help='whether to calculate entropy (and skip mixing phase)')
    parser.add_argument('--match_thresh', dest='match_threshold', default=0, type=float,
                        help='matching threshold of matching similarity')
    parser.add_argument('--mixing_thresh', dest='mixing_threshold', default=0.05, type=float,
                        help='mixing threshold of clauses')
    
    # generating config
    parser.add_argument('--init', dest='init', action='store_true',
                        help='whether use non-finetuned model')
    parser.add_argument('--repeat', dest='repeat', default=1, type=int,
                        help='repeat generated graphs')
    
    args = parser.parse_args()
    return args