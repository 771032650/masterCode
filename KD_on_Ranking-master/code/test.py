import os
import time
import world
import utils
import torch
import Procedure
import numpy as np
from world import cprint
from pprint import pprint
from tensorboardX import SummaryWriter
from sample import DistillSample
import tracemalloc

tracemalloc.start()
# set seed
utils.set_seed(world.SEED)
print(f"[SEED:{world.SEED}]")
# ----------------------------------------------------------------------------
# init model
world.DISTILL = False
if len(world.comment) == 0:
    comment = f"{world.method}"
    if world.EMBEDDING:
        comment = comment + "-embed"
    world.comment = comment
import register
from register import dataset

def load_popularity():
    r_path = world.DATA_PATH+"/"+world.dataset+"/"
    pop_save_path = r_path+"item_pop_seq_ori2.txt"
    if not os.path.exists(pop_save_path):
        pop_save_path = r_path+"item_pop_seq_ori.txt"
    print("popularity used:",pop_save_path)
    with open(pop_save_path) as f:
        print("pop save path: ", pop_save_path)
        item_list = []
        pop_item_all = []
        for line in f:
            line = line.strip().split()
            item, pop_list = int(line[0]), [float(x) for x in line[1:]]
            item_list.append(item)
            pop_item_all.append(pop_list)
    pop_item_all = np.array(pop_item_all)
    print("pop_item_all shape:", pop_item_all.shape)
    print("load pop information:",pop_item_all.mean(),pop_item_all.max(),pop_item_all.min())
    return pop_item_all

def get_dataset_tot_popularity():
    popularity_matrix=dataset.itemCount()
    popularity_matrix = popularity_matrix.astype(np.float)
    popularity_matrix += 1.0
    popularity_matrix /= popularity_matrix.sum()
    #popularity_matrix = ( popularity_matrix - popularity_matrix.min() ) / ( popularity_matrix.max() - popularity_matrix.min() )
    print("popularity information-- mean:{},max:{},min:{}".format(popularity_matrix.mean(),popularity_matrix.max(),popularity_matrix.min()))
    # popularity_matrix = np.power(popularity_matrix,popularity_exp)
    # print("After power,popularity information-- mean:{},max:{},min:{}".format(popularity_matrix.mean(),popularity_matrix.max(),popularity_matrix.min()))
    return popularity_matrix

def get_popularity_from_load(item_pop_all):
    popularity_matrix = item_pop_all[:,:-1]  # don't contain the popularity in test stages
    print("------ popularity information --------")
    print("   each stage mean:",popularity_matrix.mean(axis=0))
    print("   each stage max:",popularity_matrix.max(axis=0))
    print("   each stage min:",popularity_matrix.min(axis=0))
    # popularity_matrix = np.power(popularity_matrix,popularity_exp)  # don't contain the popullarity for test stages...
    # print("------ popularity information after power  ------")
    # print("   each stage mean:",popularity_matrix.mean(axis=0))
    # print("   each stage max:",popularity_matrix.max(axis=0))
    # print("   each stage min:",popularity_matrix.min(axis=0))
    return popularity_matrix

popularity_exp = world.lambda_pop
# popularity_exp=0
# print("----- popularity_exp : ",popularity_exp)
# pop_item_all = get_dataset_tot_popularity()
# # popularity for test...
# last_stage_popualarity = pop_item_all[:,-2]
# last_stage_popualarity = np.power(last_stage_popualarity,popularity_exp)   # laste stage popularity (method (a) )
# linear_predict_popularity = pop_item_all[:,-2] + 0.5 * (pop_item_all[:,-2] - pop_item_all[:,-3]) # linear predicted popularity (method (b))
# linear_predict_popularity[np.where(linear_predict_popularity<=0)] = 1e-9
# linear_predict_popularity[np.where(linear_predict_popularity>1.0)] = 1.0
# linear_predict_popularity = np.power(linear_predict_popularity,popularity_exp) # pop^(gamma) in paper
# dataset.add_last_popularity(linear_predict_popularity)
popularity_matrix = get_dataset_tot_popularity()
# popularity_matrix[np.where(popularity_matrix<1e-9)] = 1e-9
popularity_matrix = np.power(popularity_matrix, popularity_exp)  # pop^gamma
print("------ popularity information after powed  ------")  # popularity information
print("   each stage mean:", popularity_matrix.mean(axis=0))
print("   each stage max:", popularity_matrix.max(axis=0))
print("   each stage min:", popularity_matrix.min(axis=0))
dataset.add_expo_popularity(popularity_matrix)

# loading
file = utils.getFileName(world.model_name,
                         world.dataset,
                         world.config['latent_dim_rec'],
                         layers=world.config['lightGCN_n_layers'],
                        dns_k=world.DNS_K
                         )
file = world.SAMPLE_METHOD+'-'+str(world.config['teacher_dim'])+'-'+str(world.kd_weight)+'-'+str(world.config['de_weight'])+'-'+str(world.lambda_pop)+ '-'+ file
#file=str(world.lambda_pop)+'-'+str(world.config['de_weight'])+'-'+str(world.config['decay'])+'-'+file
file=world.teacher_model_name+'-'+str(world.t_lambda_pop)+'-'+file
file='new1'+'-'+file
weight_file = os.path.join(world.FILE_PATH, file)
print('-------------------------')
world.cprint("loaded  weights from")
print(weight_file)
print('-------------------------')
if world.SAMPLE_METHOD=='DE_RRD' or world.SAMPLE_METHOD=='SD':
    world.cprint('teacher')
    teacher_file = utils.getFileName(world.model_name,
                             world.dataset,
                             world.config['teacher_dim'],
                             layers=world.config['teacher_layer'])
    #teacher_file = "teacher-" + teacher_file
    teacher_weight_file = os.path.join(world.FILE_PATH, teacher_file)
    print('-------------------------')
    world.cprint("loaded teacher weights from")
    print(teacher_weight_file)
    print('-------------------------')
    teacher_model = register.MODELS[world.model_name](world.config,
                                                  dataset,
                                                  fix=True)
    teacher_model.eval()
    utils.load(teacher_model, teacher_weight_file)
    model = register.MODELS['lep'](world.config,dataset,teacher_model)
else:
    model = register.MODELS[world.model_name](world.config,
                                              dataset)
model.eval()
utils.load(model, weight_file)
model = model.cuda()
if world.model_name=='ConditionalBPRMF':
    model.set_popularity(popularity_matrix)
#all_users, all_items = model.computer()
# popularity=dataset.popularity()
# # index=np.argsort(-popularity)
# # popularity=popularity[index]
# import matplotlib
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# ax.scatter(range(len(popularity)),popularity)
#
# plt.show()
# test teacher
cprint("[TEST Teacher]")
results = Procedure.Test(dataset, model, 0, None, world.config['multicore'], valid=False)
pprint(results)
popularity,user_topk,r= Procedure.Popularity_Bias(dataset, model,valid=False)

cprint("[APT]")
metrics=utils.popularity_ratio(popularity,user_topk,dataset)
print(metrics)

cprint("[PopularityByGrpup]")
metrics=utils.PopularityByGrpup(user_topk,dataset)
print(str(metrics)+',')

cprint("[PrecisionByGrpup]")
testDict = dataset.testDict
metrics=utils.PrecisionByGrpup(testDict,user_topk,dataset,r)
print(str(metrics)+',')

