# -*- coding: UTF-8 -*-

from __future__ import print_function
import os


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

def multi_generator_with_pop2():
    '''
    multi-processing Dataset Generator with popularity  for PD/PDA Model
    with pair-wise sampling
    '''
    worker = min(cores // max_pre,10)
    worker = max(2,worker)
    pool = []
    Queue_buffer = multiprocessing.Queue(2000)
    Queue_info = multiprocessing.Manager().Queue(worker)
    batch_size = data.batch_size
    tot_num = data.n_train
    tot_batch = tot_num // batch_size + 1
    each_batch = tot_batch // worker + 1
    sub_sampel_batchs = [ each_batch ] * worker
               # PD/PDA, local popularity
    gen_n_batch_method = generator_n_batch_with_pop

        # for i in range(tot_batch):
    #     pool.apply_async(generator_one_batch,args=(Queue_buffer,))
    # pool.map_async(generator_n_batch_with_pop,[(Queue_buffer,n_batch) for n_batch in sub_sampel_batchs])
    for n_batch in sub_sampel_batchs:
        pool.append(multiprocessing.Process(target=gen_n_batch_method,args=(Queue_buffer,n_batch,Queue_info,)))
    for p in pool:
        p.start()
    for i in range(tot_batch):
        # print("... wait data....")
        yield Queue_buffer.get(True)
        #if i%1000 == 0:
        #print("runing..",i,tot_batch)
    # print("finish sampling....")
    for i in range(worker):
        Queue_info.put("out")
    for p in pool:
        p.join()
    print("sampled data finished")





    

def generator_n_batch_with_pop(Queue_buffer,n_batch,q_info):
    '''
    sample n_batch data with popularity (PD/PDA)
    '''
    # print('start')
    s_time = time()
    all_users = list(dataset.train_user_list.keys())
    batch_size = dataset.batch_size
    buffer = []
    for i in range(n_batch):
        batch_pos = []
        batch_neg = []
        batch_pos_pop = []
        batch_neg_pop = []
        if batch_size <= dataset.n_users:
            batch_users = rd.sample(all_users, batch_size)
        else:
            batch_users = [rd.choice(all_users) for _ in range(batch_size)]
        for u in batch_users:
            u_clicked_items = dataset.train_user_list[u]
            u_clicked_times = dataset.train_user_list_time[u]
            if  u_clicked_items == []:
                one_pos_item = 0
                batch_pos.append(one_pos_item)
                u_pos_time = rd.choice(dataset.unique_times)
            else:
                M_num = len(u_clicked_items)
                idx = np.random.randint(M_num)
                one_pos_item = u_clicked_items[idx]
                batch_pos.append(one_pos_item)
                u_pos_time = u_clicked_times[idx]
            while True:
                neg_item = rd.choice(dataset.items)
                if neg_item not in u_clicked_items:
                    batch_neg.append(neg_item)
                    break
            batch_pos_pop.append(dataset.expo_popularity[one_pos_item,u_pos_time])
            batch_neg_pop.append(dataset.expo_popularity[neg_item,u_pos_time])
        one_batch = (batch_users,batch_pos,batch_neg,batch_pos_pop,batch_neg_pop)

        Queue_buffer.put(one_batch)
    #Queue_buffer.close()
    #for one_batch in buffer:
    #    Queue_buffer.put(one_batch)
    # print('end',s_time - time())
    m = q_info.get()
    Queue_buffer.cancel_join_thread()






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

    Recmodel = register.MODELS[world.model_name](world.config, dataset,init=True)
    procedure = register.TRAIN[world.method]
    bpr = utils.BPRLoss(Recmodel, world.config)
    # ----------------------------------------------------------------------------
    file = utils.getFileName(world.model_name,
                             world.dataset,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'],
                             dns_k=world.DNS_K)
    file=str(world.lambda_pop)+'-'+str(world.de_weight)+'-'+file
    weight_file = os.path.join(world.FILE_PATH, file)
    print(f"load and save to {weight_file}")
    if world.LOAD:
        utils.load(Recmodel, weight_file)
    # ----------------------------------------------------------------------------
    earlystop = utils.EarlyStop(patience=20, model=Recmodel, filename=weight_file)
    Recmodel = Recmodel.to(world.DEVICE)
    if world.model_name == 'ConditionalBPRMF':
        Recmodel.set_popularity(linear_predict_popularity)
    # ----------------------------------------------------------------------------
    # init tensorboard
    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(
            os.path.join(
                world.BOARD_PATH,
                time.strftime("%m-%d-%Hh-%Mm-") +
                f"{world.method}-{str(world.DNS_K)}-{file.split('.')[0]}-{world.comment}"
            ))
    else:
        w = None
        world.cprint("not enable tensorflowboard")
    # ----------------------------------------------------------------------------
    # start training
    try:
        for epoch in range(world.TRAIN_epochs):

            start = time.time()
            output_information = procedure(dataset, Recmodel, bpr, epoch, w=w)


            if epoch % 5 == 0:
                print(
                    f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
                )
                cprint("TEST", ends=': ')
                results = Procedure.Test(dataset,
                                         Recmodel,
                                         epoch,
                                         w,
                                         world.config['multicore'],
                                         valid=True)
                print(results)
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
    Recmodel.load_state_dict(earlystop.best_model)
    results = Procedure.Test(dataset, Recmodel, world.TRAIN_epochs, valid=False)
    log_file = os.path.join(world.LOG_PATH, utils.getLogFile())
    with open(log_file, 'a') as f:
        f.write("#######################################\n")
        f.write(f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch+1}/{world.TRAIN_epochs}\n"\
                f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n"\
                f"TopK: {world.topks}\n")
        f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
        f.close()






