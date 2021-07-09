# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
from pprint import pprint

from world import cprint

import Procedure

print('*** Current working path ***')
print(os.getcwd())
# os.chdir(os.getcwd()+"/NeuRec")

import os

import time
import multiprocessing

import world
import torch
import numpy as np
import utils

from tensorboardX import SummaryWriter



import sys
sys.path.append(os.getcwd())
print(sys.path)
import random as rd
import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def term(sig_num, addtion):
    print('term current pid is %s, group id is %s' % (os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
signal.signal(signal.SIGTERM, term)

cores = multiprocessing.cpu_count() // 2
max_pre = 1
print("half cores:",cores)
# ----------------------------------------------------------------------------
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
    popularity_matrix = np.zeros(dataset.n_items).astype(np.float)
    for a_item,clicked_users in dataset.train_item_list.items():
        popularity_matrix[a_item] = len(clicked_users)
    popularity_matrix = popularity_matrix.astype(np.float)
    popularity_matrix += 1.0
    popularity_matrix /= popularity_matrix.sum()
    popularity_matrix = ( popularity_matrix - popularity_matrix.min() ) / ( popularity_matrix.max() - popularity_matrix.min() )
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




if __name__ == '__main__':
    # random.seed(123)
    # tf.set_random_seed(123)


    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    # config = dict()
    # config['n_users'] = data.n_users
    # config['n_items'] = data.n_items  # in batch_test.py
    # model_type = ''
    # print("data size:",sys.getsizeof(data)/(10**6),"GB")
    # # ----------  important parameters -------------------------
    popularity_exp = world.lambda_pop
    print("----- popularity_exp : ",popularity_exp)
    # test_batch_size = min(1024,args.batch_size)



    # ------------  pre computed popularity ------------------
    pop_item_all = load_popularity()
    # popularity for test...
    last_stage_popualarity = pop_item_all[:,-2]
    t_last_stage_popualarity=np.power(last_stage_popualarity,0)
    last_stage_popualarity = np.power(last_stage_popualarity,popularity_exp)   # laste stage popularity (method (a) )
    linear_predict_popularity = pop_item_all[:,-2] + 0.5 * (pop_item_all[:,-2] - pop_item_all[:,-3]) # linear predicted popularity (method (b))
    linear_predict_popularity[np.where(linear_predict_popularity<=0)] = 1e-9
    linear_predict_popularity[np.where(linear_predict_popularity>1.0)] = 1.0
    linear_predict_popularity = np.power(linear_predict_popularity,popularity_exp) # pop^(gamma) in paper
    dataset.add_last_popularity(linear_predict_popularity)

    popularity_matrix = get_popularity_from_load(pop_item_all)
    # popularity_matrix[np.where(popularity_matrix<1e-9)] = 1e-9
    popularity_matrix = np.power(popularity_matrix, popularity_exp)  # pop^gamma
    print("------ popularity information after powed  ------")  # popularity information
    print("   each stage mean:", popularity_matrix.mean(axis=0))
    print("   each stage max:", popularity_matrix.max(axis=0))
    print("   each stage min:", popularity_matrix.min(axis=0))
    dataset.add_expo_popularity(popularity_matrix)

    # loading teacher
    teacher_file = utils.getFileName(world.teacher_model_name,
                                     world.dataset,
                                     world.config['teacher_dim'],
                                     layers=world.config['teacher_layer'],
                                     dns_k=world.DNS_K)
    teacher_file = str(world.de_weight) + '-' + teacher_file
    teacher_file = str(world.t_lambda_pop) + '-' + teacher_file
    teacher_weight_file = os.path.join(world.FILE_PATH, teacher_file)
    print('-------------------------')
    world.cprint("loaded teacher weights from")
    print(teacher_weight_file)
    print('-------------------------')
    teacher_config = utils.getTeacherConfig(world.config)
    world.cprint('teacher')
    teacher_model = register.MODELS[world.teacher_model_name](teacher_config,
                                                      dataset,
                                                      fix=True)
    teacher_model.eval()
    utils.load(teacher_model, teacher_weight_file)
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # loading student
    world.cprint('student')
    if world.SAMPLE_METHOD == 'DE_RRD':
        student_model = register.MODELS['lep'](world.config, dataset, teacher_model)
    elif world.SAMPLE_METHOD == 'SD':
        student_model = register.MODELS['newModel'](world.config, dataset, teacher_model)
    else:
        student_model = register.MODELS[world.model_name](world.config, dataset)

    # ----------------------------------------------------------------------------
    # to device
    student_model = student_model.to(world.DEVICE)
    teacher_model = teacher_model.to(world.DEVICE)
    if world.model_name=='ConditionalBPRMF':
        student_model.set_popularity(last_stage_popualarity)
    if world.teacher_model_name == 'ConditionalBPRMF':
        teacher_model.set_popularity(t_last_stage_popualarity)
    # ----------------------------------------------------------------------------
    # choosing paradigms
    procedure = register.DISTILL_TRAIN['pop']
    sampler = register.SAMPLER[world.SAMPLE_METHOD](dataset, student_model, teacher_model, world.DNS_K)

    bpr = utils.BPRLoss(student_model, world.config)
    # ------------------
    # ----------------------------------------------------------
    # get names
    file = utils.getFileName(world.model_name,
                             world.dataset,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'],
                             dns_k=world.DNS_K)
    file = world.teacher_model_name + '-' +str(world.t_lambda_pop) + '-'+ world.SAMPLE_METHOD + '-' + str(world.config['teacher_dim']) + '-' + str(
        world.kd_weight) + '-' + str(world.config['de_weight']) + '-' + str(world.lambda_pop) + '-' + file
    weight_file = os.path.join(world.FILE_PATH, file)
    print('-------------------------')
    print(f"load and save student to {weight_file}")
    if world.LOAD:
        utils.load(student_model, weight_file)
    # ----------------------------------------------------------------------------
    # training setting
    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            os.path.join(
                world.BOARD_PATH, time.strftime(
                    "%m-%d-%Hh-%Mm-") + f"{world.method}-{str(world.DNS_K)}-{file.split('.')[0]}-{world.comment}-DISTILL"
            )
        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")
    earlystop = utils.EarlyStop(patience=20,
                                model=student_model,
                                filename=weight_file)
    # ----------------------------------------------------------------------------
    # test teacher
    cprint("[TEST Teacher]")
    results = Procedure.Test(dataset, teacher_model, 0, None, world.config['multicore'], valid=False)
    pprint(results)
    # ----------------------------------------------------------------------------
    # start training
    try:
        for epoch in range(world.TRAIN_epochs):

            start = time.time()
            output_information = procedure(dataset, student_model, sampler, bpr, epoch, w=w)

            print(
                f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
            )
            # snapshot = tracemalloc.take_snapshot()
            # utils.display_top(snapshot)
            print(f"    [TEST TIME] {time.time() - start}")
            if epoch % 5 == 0:
                start = time.time()
                cprint("    [TEST]")
                results = Procedure.Test(dataset, student_model, epoch, w, world.config['multicore'], valid=True)
                pprint(results)
                print(f"    [TEST TIME] {time.time() - start}")
                if earlystop.step(epoch, results):
                    print("trigger earlystop")
                    print(f"best epoch:{earlystop.best_epoch}")
                    print(f"best results:{earlystop.best_result}")
                    break
    finally:
        if world.tensorboard:
            w.close()

    best_result = earlystop.best_result
    torch.save(earlystop.best_model, weight_file)
    student_model.load_state_dict(earlystop.best_model)
    results = Procedure.Test(dataset,
                             student_model,
                             world.TRAIN_epochs,
                             valid=False)
    log_file = os.path.join(world.LOG_PATH, utils.getLogFile())
    with open(log_file, 'a') as f:
        f.write("#######################################\n")
        f.write(
            f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch + 1}/{world.TRAIN_epochs}\n" \
            f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n" \
            f"TopK: {world.topks}\n")
        f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
        f.close()





