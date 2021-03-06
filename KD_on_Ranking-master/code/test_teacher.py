# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
from pprint import pprint

from world import cprint

import Procedure

from utils import timer

from model import TwoLinear, ConditionalBPRMF,OneLinear,ZeroLinear

from sample import userAndMatrix

from Procedure import test_one_batch

import model

from sample import Sample_original

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



def Sample_DNS_python_valid(vaild_users,groundTrue):
        """python implementation for 'UniformSample_DNS'
        """

        allPos = groundTrue
        S = []
        BinForUser = np.zeros(shape=(dataset.m_items,)).astype("int")
        for user in vaild_users:
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            BinForUser[:] = 0
            BinForUser[posForUser] = 1
            NEGforUser = np.where(BinForUser == 0)[0]
            per_user_num = len(posForUser)
            for i in range(per_user_num):
                positem = posForUser[i]
                negindex = np.random.randint(0, len(NEGforUser), size=(1,))
                negitems = NEGforUser[negindex]
                add_pair = [user, positem, negitems]
                S.append(add_pair)
        return S



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

    # test_batch_size = min(1024,args.batch_size)



    # ------------  pre computed popularity ------------------
    # pop_item_all = load_popularity()
    # # popularity for test...
    # last_stage_popualarity = pop_item_all[:,-2]
    # last_stage_popualarity = np.power(last_stage_popualarity,popularity_exp)   # laste stage popularity (method (a) )
    # linear_predict_popularity = pop_item_all[:,-2] + 0.5 * (pop_item_all[:,-2] - pop_item_all[:,-3]) # linear predicted popularity (method (b))
    # linear_predict_popularity[np.where(linear_predict_popularity<=0)] = 1e-9
    # linear_predict_popularity[np.where(linear_predict_popularity>1.0)] = 1.0
    # linear_predict_popularity = np.power(linear_predict_popularity,popularity_exp) # pop^(gamma) in paper
    # dataset.add_last_popularity(linear_predict_popularity)
    # t_linear_predict_popularity = np.power(linear_predict_popularity, t_popularity_exp)

    popularity_matrix = get_dataset_tot_popularity()
    # popularity_matrix[np.where(popularity_matrix<1e-9)] = 1e-9
    #popularity_matrix = np.power(popularity_matrix, popularity_exp)  # pop^gamma
    print("------ popularity information after powed  ------")  # popularity information
    print("   each stage mean:", popularity_matrix.mean(axis=0))
    print("   each stage max:", popularity_matrix.max(axis=0))
    print("   each stage min:", popularity_matrix.min(axis=0))
    popularity_matrix=torch.from_numpy(popularity_matrix).float()
    dataset.add_expo_popularity(popularity_matrix)
    # loading teacher
    teacher_file = utils.getFileName(world.teacher_model_name,
                                     world.dataset,
                                     world.config['teacher_dim'],
                                     layers=world.config['teacher_layer'],
                                     dns_k=world.DNS_K)
    teacher_file = str(world.de_weight) + '-' + str(world.config['decay']) + '-' +teacher_file
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

    # to device

    teacher_model = teacher_model.cuda()
    # if world.model_name=='ConditionalBPRMF':
    #     student_model.set_popularity(popularity_matrix)
    # if world.teacher_model_name == 'ConditionalBPRMF'  :
    #     teacher_model.set_popularity(popularity_matrix)
    if world.mate_model==2:
        weight1_model = TwoLinear(dataset.n_users,dataset.m_items).cuda()

    elif world.mate_model==0:
        weight1_model = ZeroLinear().cuda()

    else:
        weight1_model = OneLinear(dataset.m_items).cuda()

    weight1_optimizer = torch.optim.Adam(weight1_model.parameters(), lr=world.config['mate_lr'],weight_decay=world.config['mate_decay_1'])


    # ----------------------------------------------------------------------------
    # choosing paradigms
    print(world.distill_method)
    procedure = register.DISTILL_TRAIN[world.distill_method]

    #bpr = utils.BPRLoss(student_model, world.config)
    # ------------------
    # ----------------------------------------------------------
    # get names
    file='test-'+teacher_file
    weight_file = os.path.join(world.FILE_PATH, file)
    print('-------------------------')
    print(f"load and save student to {weight_file}")
    # if world.LOAD:
    #     utils.load(student_model, weight_file)
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
                                model=weight1_model,
                                filename=weight_file)
    # ----------------------------------------------------------------------------
    # test teacher
    # cprint("[TEST Teacher]")
    # results = Procedure.Test(dataset, teacher_model, 0, None, world.config['multicore'], valid=False)
    # pprint(results)
    # ----------------------------------------------------------------------------
    # start training


    valid_Dict = dataset.validDict
    vaild_keys = list(valid_Dict.keys())
    groundTrue = [valid_Dict[u] for u in vaild_keys]

    mean_criterion = torch.nn.BCELoss(reduction='mean')




    try:
        for epoch in range(world.TRAIN_epochs):

            start = time.time()
            #output_information = procedure(dataset, student_model, sampler, bpr, epoch, w=w)

            aver_loss_1 = 0
            aver_loss_2 = 0
            aver_loss_3 = 0


            with timer(name='sampling'):
                S = Sample_original(dataset)
            S = torch.Tensor(S).long()
            users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
            #negItems = negItems.reshape((-1, 1))
            users, posItems, negItems= utils.shuffle(users, posItems, negItems)

            vaild_S = Sample_DNS_python_valid(vaild_keys, groundTrue)
            vaild_S = torch.Tensor(vaild_S).long()
            vaild_users, vaild_posItems, vaild_negItems = vaild_S[:, 0], vaild_S[:, 1], vaild_S[:, 2]
            vaild_users = vaild_users.repeat(2)
            vaild_users = vaild_users.cuda()
            vaild_item = torch.cat((vaild_posItems, vaild_negItems),dim=0)
            vaild_item = vaild_item.cuda()
            vaild_lable_1=torch.ones((len(vaild_posItems)))
            vaild_lable_2 = torch.zeros((len(vaild_negItems)))
            vaild_lable = torch.cat((vaild_lable_1, vaild_lable_2), dim=0).cuda()

            batch_users_v, batch_item_v,vaild_lable = utils.shuffle(vaild_users, vaild_item,vaild_lable)

            total_batch = len(users) // world.config['bpr_batch_size'] + 1
            total_batch_1 = len(vaild_users) // world.config['bpr_batch_size'] + 1
            # formal parameter: Using training set to update parameters
            with timer(name="one_step_model"):
                for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                        utils.minibatch(users,
                                        posItems,
                                        negItems,
                                        batch_size=1024)):

                    #weight1 = torch.exp(weight1 / 5)  # for stable training





                        weight1_model.train()
                        batch_item_v_pop = torch.pow(popularity_matrix[batch_item_v].cuda(),
                                                  weight1_model())
                        y_hat_l = teacher_model(batch_users_v, batch_item_v,batch_item_v_pop)
                        y_hat_l=torch.sigmoid(y_hat_l)
                        loss_l_p=mean_criterion(y_hat_l,vaild_lable)
                        weight1_optimizer.zero_grad()
                        loss_l_p.backward()
                        weight1_optimizer.step()


                        if batch_i==0:
                            if world.mate_model==2:
                                print(weight1_model.getUsersRating(torch.tensor([0,1,2,3,4]).long().cuda()))
                            else:
                                print(weight1_model.getUsersRating())
                        cri = loss_l_p.cpu().item()
                        aver_loss_2 += cri



          


            aver_loss_2 = aver_loss_2 / total_batch_1

            info = f"{timer.dict()}[loss_2:{aver_loss_2:.5e}]"
            timer.zero()



            print(
                f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {info}'
            )
            # snapshot = tracemalloc.take_snapshot()
            # utils.display_top(snapshot)
            print(f"    [TEST TIME] {time.time() - start}")
            if epoch % 5 == 0 :
                start = time.time()
                all_pop = torch.pow(popularity_matrix,weight1_model().cpu())
                teacher_model.set_popularity(all_pop)
                cprint("    [valid1]")

                results = Procedure.Test(dataset, teacher_model, epoch, None, world.config['multicore'], valid=True)
                pprint(results)
                cprint("    [valid2]")
                results = Procedure.Test(dataset, teacher_model, epoch, 1, world.config['multicore'], valid=True)
                pprint(results)
                print(f"    [valid TIME] {time.time() - start}")
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
    all_pop = torch.pow(popularity_matrix, weight1_model().cpu())
    teacher_model.set_popularity(all_pop)
    results = Procedure.Test(dataset,
                             teacher_model,
                             world.TRAIN_epochs,
                             valid=False)
    popularity, user_topk, r = Procedure.Popularity_Bias(dataset, teacher_model, valid=False)
    metrics1 = utils.popularity_ratio(popularity, user_topk, dataset)

    testDict = dataset.testDict
    metrics2 = utils.PrecisionByGrpup(testDict, user_topk, dataset, r)

    log_file = os.path.join(world.LOG_PATH, world.SAMPLE_METHOD+'-'+utils.getLogFile())
    with open(log_file, 'a') as f:
        f.write("#######################################\n")
        f.write(f"{file}\n")
        f.write(
            f"SEED: {world.SEED}, DNS_K: {str(world.DNS_K)}, Stop at: {earlystop.best_epoch + 1}/{world.TRAIN_epochs}\n" \
            f"flag: {file.split('.')[0]}. \nLR: {world.config['lr']}, DECAY: {world.config['decay']}\n" \
            f"TopK: {world.topks}\n")
        f.write(f"%%Valid%%\n{best_result}\n%%TEST%%\n{results}\n")
        f.write(f"{metrics1}\n{metrics2}\n")
        f.close()





