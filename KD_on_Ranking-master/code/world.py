'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
import torch
from enum import Enum
from parse import parse_args
import multiprocessing
import sys



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ROOT_PATH = '..'
# CODE_PATH = os.path.join(ROOT_PATH, 'code')
# FILE_PATH = os.path.join(CODE_PATH, 'checkpoints')
# BOARD_PATH = os.path.join(CODE_PATH, 'runs')
# #DATA_PATH = os.path.join(ROOT_PATH, 'data')
# DATA_PATH = ROOT_PATH+'/data'
# LOG_PATH = os.path.join(ROOT_PATH, 'log')
#
# sys.path.append(os.path.join(CODE_PATH, 'sources'))

# ROOT_PATH = '/gdata1/chengang/KD'
# CODE_PATH = os.path.join('/ghome/chengang/KD', 'code')
# FILE_PATH = os.path.join('/gdata1/chengang/KD', 'checkpoints2')
# BOARD_PATH = os.path.join('/gdata1/chengang/KD', 'runs')
# #DATA_PATH = os.path.join(ROOT_PATH, 'data')
# DATA_PATH = ROOT_PATH+'/data'
# LOG_PATH = os.path.join('/gdata1/chengang/KD', 'log2')
#
# sys.path.append(os.path.join(CODE_PATH, 'sources'))

ROOT_PATH = '/data/chengang/KD'
CODE_PATH = os.path.join('/home/chengang/KD', 'code')
FILE_PATH = os.path.join('/data/chengang/KD', 'checkpoints')
BOARD_PATH = os.path.join('/data/chengang/KD', 'runs')
#DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATA_PATH = ROOT_PATH+'/data'
LOG_PATH = os.path.join('/data/chengang/KD', 'log')

sys.path.append(os.path.join(CODE_PATH, 'sources'))

args = parse_args()
ARGS = args
EMBEDDING = args.embedding
DE_EMBEDDING=args.de_loss
SAMPLE_METHOD = args.sampler
# print(SAMPLE_METHOD)
# CD = False

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon']
all_models = ['mf', 'lgn','newModel','ulgn','myModel','ConditionalBPRMF','BPRMF']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['teacher_dim'] = args.teacher_dim
config['teacher_layer'] = args.teacher_layer
config['teacher_model'] = args.teacher_model
config['num_expert'] = args.num_expert
config['de_loss'] = args.de_loss
config['de_weight'] = args.de_weight
kd_weight=args.kd_weight
de_weight=args.de_weight
if SAMPLE_METHOD=='RD':
    loss_weight=1-kd_weight
else:
    loss_weight=1
DNS_K = args.dns_k
margin=args.margin
method = args.method
distill_method=args.distill_method
if method == 'dns' and DNS_K == 1:
    method = 'original'

GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if GPU else "cpu")
DEVICE1 = torch.device('cuda:0' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
SEED = args.seed
lambda_pop= args.lambda_pop
t_lambda_pop= args.t_lambda_pop
dataset = args.dataset
model_name = args.model
dataset_split=args.dataset_split
teacher_model_name=args.teacher_model
if model_name == 'lgn' and args.layer == 0:
    model_method = 'mf'
else:
    model_method = 'lgn'
# else:
#     model_method = 'lgn'
# if dataset not in all_dataset:
# raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(
        f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
ALLDATA = args.alldata
TESTDATA = args.testdata
ONE = args.one
if ONE and TESTDATA:
    raise TypeError("levave-one-out data shouldn't be trained only by test!!!")
T = args.T
beta = args.beta
p0 = args.p0
startepoch = args.startepoch
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
if ONE:
    dataset = dataset + "_one"
    topks = [50]
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str, ends='\n'):
    print(f"\033[0;30;43m{words}\033[0m", end=ends)


logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
