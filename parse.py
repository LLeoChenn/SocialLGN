import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # for hyper parameters
    parser.add_argument('-m', '--model', type=str, default='LightGCN')
    parser.add_argument('-d', '--dataset', type=str, default='lastfm')
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalization")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int, default=100)
    # for deep model
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of graphs")
    parser.add_argument('--layer_attention', type=int, default=0,
                        help="use layer attention (1: enable, 0: disable)")
    # normally unchanged
    parser.add_argument('--topks', nargs='?', default="[200]",
                        help="@k test list")
    parser.add_argument('--testbatch', type=str, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--lambda_mmr', type=float, default=0,
                        help="trade-off parameter for MMR (0: focus on diversity, 1: focus on relevance)")
    parser.add_argument('--mmr_T', type=int, default=4,
                        help="MMR re-ranking: select K//T items from K candidates (T>=1)")
    return parser.parse_args()
