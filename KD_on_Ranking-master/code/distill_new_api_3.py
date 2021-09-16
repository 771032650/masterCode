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


def load_vaild_data(dataset, Recmodel):
    """evaluate models

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        Recmodel (PairWiseModel):
        epoch (int):
        w (SummaryWriter, optional): Tensorboard writer
        multicore (int, optional): The num of cpu cores for testing. Defaults to 0.

    Returns:
        dict: summary of metrics
    """
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict
    testDict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel.eval()
    max_K = 1
    with torch.no_grad():
        users =torch.from_numpy(np.arange(dataset.n_users))
        flag=0

        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)

            batch_users_gpu = batch_users.long()
            batch_users_gpu = batch_users_gpu.cuda()

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            if not world.TESTDATA:
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -1e10
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            if flag==0:
                user_array=batch_users
                rating_array=rating_K.cpu()
                flag=1
            else:
                user_array=torch.cat((user_array,batch_users),dim=-1)
                rating_array = torch.cat((rating_array, rating_K.cpu()), dim=0)
        print(user_array.shape)
        print(rating_array.shape)
        return user_array,rating_array


def get_rank_loss(users,p_items,teacher_model,model):
    dim_item = p_items.shape[-1]
    vector_user = users.repeat((dim_item, 1)).t().reshape((-1,))
    vector_item = p_items.reshape((-1,))
    weight_pop = teacher_model(vector_user, vector_item)
    scores = weight_pop.reshape((-1, dim_item))

    samples = scores.sort(dim=-1)[1].cuda()
    new_users=users[samples]
    new_items=p_items[samples]

    dim_item = p_items.shape[-1]
    vector_user = new_users.repeat((dim_item, 1)).t().reshape((-1,))
    vector_item = new_items.reshape((-1,))
    S1=model(vector_user,vector_item)
    S1 = S1.reshape((-1, dim_item))

    #S2 = model(new_users, n_items)
    above = S1.sum(1, keepdim=True)
    below1 = S1.flip(-1).exp().cumsum(1)
    #below2 = S2.exp().sum(1, keepdim=True)
    below = (below1 ).log().sum(1, keepdim=True)
    return -torch.sum(above - below)


if __name__ == '__main__':

    popularity_matrix = get_dataset_tot_popularity()
    # popularity_matrix[np.where(popularity_matrix<1e-9)] = 1e-9
    #popularity_matrix = np.power(popularity_matrix, popularity_exp)  # pop^gamma
    print("------ popularity information after powed  ------")  # popularity information
    print("   each stage mean:", popularity_matrix.mean(axis=0))
    print("   each stage max:", popularity_matrix.max(axis=0))
    print("   each stage min:", popularity_matrix.min(axis=0))
    popularity_matrix=torch.from_numpy(popularity_matrix).float().cuda()
    dataset.add_expo_popularity(popularity_matrix)
    # loading teacher
    teacher_file = utils.getFileName(world.teacher_model_name,
                                     world.dataset,
                                     world.config['teacher_dim'],
                                     layers=world.config['teacher_layer'],
                                     dns_k=world.DNS_K)
    teacher_file = str(world.de_weight) + '-' + str(world.config['decay']) + '-' +teacher_file
    teacher_file = 'new1'+'-'+str(world.t_lambda_pop) + '-' + teacher_file
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

    student_model = register.MODELS[world.model_name](world.config, dataset)

    # ----------------------------------------------------------------------------
    # to device
    student_model = student_model.cuda()
    teacher_model = teacher_model.cuda()
    # if world.model_name=='ConditionalBPRMF':
    #     student_model.set_popularity(popularity_matrix)
    # if world.teacher_model_name == 'ConditionalBPRMF'  :
    #     teacher_model.set_popularity(popularity_matrix)
    weight1_model = TwoLinear(dataset.n_users,dataset.m_items).cuda()
    weight2_model = ZeroLinear().cuda()
    weight1_optimizer = torch.optim.Adam(weight1_model.parameters(), lr=world.config['mate_lr'],weight_decay= world.config['mate_decay_1'])
    weight2_optimizer = torch.optim.Adam(weight2_model.parameters(), lr=world.config['mate_lr'],
                                         weight_decay=world.config['mate_decay_1'])
    #one_step_model_optimizer = torch.optim.Adam(one_step_model.parameters(), lr=0.001)
    student_model_optimizer = torch.optim.SGD(student_model.params(), lr=world.config['lr'])
    # ----------------------------------------------------------------------------
    # choosing paradigms
    print(world.distill_method)
    procedure = register.DISTILL_TRAIN[world.distill_method]

    sampler = register.SAMPLER[world.SAMPLE_METHOD](dataset=dataset, student=student_model,teacher=teacher_model,weight1_model=weight1_model,weight2_model=weight2_model,dns=world.DNS_K)

    #bpr = utils.BPRLoss(student_model, world.config)
    # ------------------
    # ----------------------------------------------------------
    # get names
    file = utils.getFileName(world.model_name,
                             world.dataset,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'],
                             dns_k=world.DNS_K)
    file = world.comment+'-'+world.teacher_model_name + '-' +str(world.t_lambda_pop) + '-'+ world.SAMPLE_METHOD + '-' + str(world.config['teacher_dim']) + '-' + str(
        world.config['mate_decay_1']) + '-' + str(world.mate_model) + '-' + str(world.config['mate_decay_2']) +'-'+file
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
                                model=student_model,
                                filename=weight_file)
    # ----------------------------------------------------------------------------
    # test teacher
    # cprint("[TEST Teacher]")
    # results = Procedure.Test(dataset, teacher_model, 0, None, world.config['multicore'], valid=False)
    # pprint(results)
    # ----------------------------------------------------------------------------
    # start training


    vaild_keys,groundTrue = load_vaild_data(dataset,teacher_model)
    mean_criterion = torch.nn.BCELoss(reduction='sum')
    user_keys = torch.arange(dataset.n_users)




    try:
        for epoch in range(world.TRAIN_epochs):

            start = time.time()
            #output_information = procedure(dataset, student_model, sampler, bpr, epoch, w=w)

            aver_loss_1 = 0
            aver_loss_2 = 0
            aver_loss_3 = 0


            with timer(name='sampling'):
                S = sampler.PerSample()
            S = torch.Tensor(S).long()

            users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
            negItems = negItems.reshape((-1, 1))
            users, posItems, negItems= utils.shuffle(users, posItems, negItems)

            total_batch = len(users) // world.config['bpr_batch_size'] + 1

            vaild_S = Sample_DNS_python_valid(vaild_keys, groundTrue)
            vaild_S = torch.Tensor(vaild_S).long()
            vaild_users, vaild_posItems, vaild_negItems = vaild_S[:, 0], vaild_S[:, 1], vaild_S[:, 2]
            vaild_users = vaild_users.repeat(2)
            vaild_users = vaild_users.cuda()
            vaild_item = torch.cat((vaild_posItems, vaild_negItems), dim=0)
            vaild_item = vaild_item.cuda()
            vaild_lable_1 = torch.ones((len(vaild_posItems)))
            vaild_lable_2 = torch.zeros((len(vaild_negItems)))
            vaild_lable = torch.cat((vaild_lable_1, vaild_lable_2), dim=0).cuda()

            batch_users_v, batch_item_v, vaild_lable = utils.shuffle(vaild_users, vaild_item, vaild_lable)

            # formal parameter: Using training set to update parameters
            with timer(name="one_step_model"):
                for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                        utils.minibatch(users,
                                        posItems,
                                        negItems,
                                        batch_size=1024)):


                        one_step_model = register.MODELS[world.model_name](world.config, dataset).cuda()
                        one_step_model.load_state_dict(student_model.state_dict())
                        one_step_model.train()
                        batch_users=batch_users.cuda()
                        batch_pos = batch_pos.cuda()
                        batch_neg = batch_neg.cuda()
                        batch_neg_1, _, KD_loss,kd_items,kd_users = sampler.Sample(
                                    batch_users, batch_pos, batch_neg, one_step_model,one_step=1)


                        # all pair data in this block
                        loss_f, reg_loss =one_step_model.bpr_loss_pop(batch_users, batch_pos, batch_neg_1, None,
                                                                 None)
                        reg_loss = reg_loss *world.config['decay']
                        assert KD_loss.requires_grad == True
                        loss_f_all_one = loss_f  + KD_loss
                        loss_f_all_one = loss_f_all_one + reg_loss
                        one_step_model.zero_grad()
                        grads = torch.autograd.grad(loss_f_all_one, (one_step_model.params()),create_graph=True)
                        one_step_model.update_params(world.config['lr'], source_params=grads)
                        cri = KD_loss.cpu().item()
                        aver_loss_1 += cri

                        weight1_model.train()

                        y_hat_l_1 = one_step_model(batch_users_v, batch_item_v)
                        y_hat_l_1 = torch.sigmoid(y_hat_l_1)
                        loss_l_p_1 = mean_criterion(y_hat_l_1, vaild_lable)
                        loss_l_p_3=get_rank_loss(user_keys.cuda(),kd_items,teacher_model,one_step_model)
                        loss_l_p = loss_l_p_1+loss_l_p_3

                        weight1_optimizer.zero_grad()
                        loss_l_p.backward()
                        weight1_optimizer.step()





                        cri = loss_l_p.cpu().item()
                        aver_loss_2 += cri

                        if batch_i==0:
                            print(weight1_model.getUsersRating(torch.tensor([0,1,2,3,4]).long().cuda()))



                        student_model.train()
                        batch_neg_2, _, KD_loss_2 ,_,_= sampler.Sample(
                                batch_users, batch_pos, batch_neg, None,one_step=0)

                        loss_f, reg_loss = student_model.bpr_loss_pop(batch_users, batch_pos, batch_neg_2, None,
                                                                       None)
                        reg_loss = reg_loss * world.config['decay']
                        assert KD_loss_2.requires_grad == True

                        loss_f_all_2 = loss_f + KD_loss_2

                        loss_f_all_2 = loss_f_all_2 + reg_loss

                        student_model_optimizer.zero_grad()
                        loss_f_all_2.backward()
                        student_model_optimizer.step()

                        cri=KD_loss_2.cpu().item()
                        aver_loss_3 += cri





            # del k2
          

            aver_loss_1 = aver_loss_1 / total_batch
            aver_loss_2 = aver_loss_2 / total_batch
            aver_loss_3 = aver_loss_3 / total_batch

            info = f"{timer.dict()}[loss_1:{aver_loss_1:.5e}][loss_2:{aver_loss_2:.5e}][BPR loss:{aver_loss_3:.5e}]"
            timer.zero()



            print(
                f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {info}'
            )
            # snapshot = tracemalloc.take_snapshot()
            # utils.display_top(snapshot)
            print(f"    [TEST TIME] {time.time() - start}")
            if epoch % 5 == 0 :
                start = time.time()
                cprint("    [valid1]")
                results = Procedure.Test(dataset, student_model, epoch, None, world.config['multicore'], valid=True)
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
    student_model.load_state_dict(earlystop.best_model)
    results = Procedure.Test(dataset,
                             student_model,
                             world.TRAIN_epochs,
                             valid=False)
    popularity, user_topk, r = Procedure.Popularity_Bias(dataset, student_model, valid=False)
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





