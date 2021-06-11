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
from dim2pop import plot_bias
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
# loading
file = utils.getFileName(world.model_name,
                         world.dataset,
                         world.config['latent_dim_rec'],
                         layers=world.config['lightGCN_n_layers'],
                        dns_k=world.DNS_K
                         )
file = world.SAMPLE_METHOD+'-'+str(world.config['teacher_dim'])+'-'+str(world.kd_weight)+'-'+str(world.config['de_weight'])+'-'+str(world.lambda_pop)+ '-' + file
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
                                              dataset,
                                              fix=False)
model.eval()
utils.load(model, weight_file)
model = model.to(world.DEVICE)
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
metrics=utils.popularity_ratio(popularity,user_topk,dataset)
print(metrics)

testDict = dataset.testDict
metrics=utils.PrecisionByGrpup(testDict,user_topk,dataset,r)
print(metrics)

