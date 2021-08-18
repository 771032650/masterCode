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
def weight_test(dataset,Recmodel,valid):
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
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel.eval()
    max_K = max(world.topks)

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.DEVICE)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            if not world.TESTDATA:
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -1e10
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        results = {
                'precision': np.zeros(len(world.topks)),
                'recall': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks)),
                # 'dcg': np.zeros(len(world.topks))
        }

        pre_results = []
        for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
                # results['dcg'] += result['dcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        return results


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
    t_popularity_exp = 0.0
    print("----- popularity_exp : ",popularity_exp)
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
    popularity_matrix=torch.from_numpy(popularity_matrix).float().to(world.DEVICE)
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
    teacher_model = teacher_model.to(world.DEVICE1)
    # if world.model_name=='ConditionalBPRMF':
    #     student_model.set_popularity(popularity_matrix)
    # if world.teacher_model_name == 'ConditionalBPRMF'  :
    #     teacher_model.set_popularity(popularity_matrix)

    #weight1_model = TwoLinear(dataset.n_users,dataset.m_items).to(world.DEVICE)
    weight1_model = ZeroLinear().to(world.DEVICE)
    weight1_optimizer = torch.optim.Adam(weight1_model.parameters(), lr=0.01)
    one_step_model = register.MODELS[world.model_name](world.config, dataset).to(world.DEVICE)
    one_step_model_optimizer = torch.optim.Adam(one_step_model.parameters(), lr=0.001)
    student_model_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    # ----------------------------------------------------------------------------
    # choosing paradigms
    print(world.distill_method)
    procedure = register.DISTILL_TRAIN[world.distill_method]

    sampler = register.SAMPLER[world.SAMPLE_METHOD](dataset=dataset, student=student_model,teacher=teacher_model,weight_model=weight1_model,one_step_model=one_step_model,dns=world.DNS_K)

    #bpr = utils.BPRLoss(student_model, world.config)
    # ------------------
    # ----------------------------------------------------------
    # get names
    file = utils.getFileName(world.model_name,
                             world.dataset,
                             world.config['latent_dim_rec'],
                             layers=world.config['lightGCN_n_layers'],
                             dns_k=world.DNS_K)
    file = world.teacher_model_name + '-' +str(world.t_lambda_pop) + '-'+ world.SAMPLE_METHOD + '-' + str(world.config['teacher_dim']) + '-' + str(
        world.kd_weight) + '-' + str(world.config['de_weight']) + '-' + str(world.lambda_pop) +'-'+file
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


    valid_Dict = dataset.validDict
    vaild_users = list(valid_Dict.keys())
    groundTrue = [valid_Dict[u] for u in vaild_users]
    vaild_S=Sample_DNS_python_valid(vaild_users,groundTrue)
    vaild_S = torch.Tensor(vaild_S).long()
    vaild_users, vaild_posItems, vaild_negItems = vaild_S[:, 0], vaild_S[:, 1], vaild_S[:, 2]


    try:
        for epoch in range(world.TRAIN_epochs):

            start = time.time()
            #output_information = procedure(dataset, student_model, sampler, bpr, epoch, w=w)

            aver_loss_1 = 0
            aver_loss_2 = 0
            aver_loss_3 = 0
            aver_kd = 0

            with timer(name='sampling'):
                S = sampler.PerSample()
            S = torch.Tensor(S).long()
            users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2]
            negItems = negItems.reshape((-1, 1))
            users, posItems, negItems= utils.shuffle(users, posItems, negItems)
            total_batch = len(users) // world.config['bpr_batch_size'] + 1
            total_batch_1 = len(vaild_users) // world.config['bpr_batch_size'] + 1
            # formal parameter: Using training set to update parameters

            one_step_model.load_state_dict(student_model.state_dict())
            one_step_model.train()
            with timer(name="one_step_model"):
                for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                        utils.minibatch(users,
                                        posItems,
                                        negItems,
                                        batch_size=1024)):

                    #weight1 = torch.exp(weight1 / 5)  # for stable training

                    batch_users=batch_users.to(world.DEVICE)
                    batch_pos = batch_pos.to(world.DEVICE)
                    batch_neg = batch_neg.to(world.DEVICE)
                    batch_neg, weights, KD_loss = sampler.Sample(
                            batch_users, batch_pos, batch_neg, epoch,one_step=1)

                    # batch_pos_pop = torch.pow(popularity_matrix[batch_pos].to(world.DEVICE),
                    #                           weight1_model(batch_users, batch_pos))
                    # batch_neg_pop = torch.pow(popularity_matrix[batch_neg].to(world.DEVICE),
                    #                           weight1_model(batch_users, batch_neg))

                    # all pair data in this block
                    loss_f, reg_loss =one_step_model.bpr_loss_pop(batch_users, batch_pos, batch_neg, None,
                                                             None, weights=weights)
                    reg_loss = reg_loss *world.config['decay']
                    assert KD_loss.requires_grad == True
                    loss_f_all_one = loss_f  + KD_loss
                    loss_f_all_one = loss_f_all_one + reg_loss
                    one_step_model_optimizer.zero_grad()
                    loss_f_all_one.backward()
                    one_step_model_optimizer.step()
                    cri = loss_f_all_one.cpu().item()
                    aver_loss_1 += cri
                    # update parameters of one_step_model
                    # one_step_model.zero_grad()
                    # grads = torch.autograd.grad(loss_f_all, (one_step_model.params()), create_graph=False)
                    # one_step_model.update_params(world.config['lr'], source_params=grads)
                #
                #     # latter hyper_parameter: Using uniform set to update hyper_parameters
            weight1_model.train()
            if epoch%1==0:
                with timer(name="weight1_model"):
                    for (batch_j, (batch_users_v, batch_pos_v, batch_neg_v)) in enumerate(
                                utils.minibatch(vaild_users,
                                                vaild_posItems,
                                                vaild_negItems,
                                                batch_size=len(vaild_posItems))):

                        batch_users_v = batch_users_v.to(world.DEVICE)
                        batch_pos_v = batch_pos_v.to(world.DEVICE)
                        #batch_neg_v = batch_neg_v.to(world.DEVICE1)
                        #k1=weight1_model(batch_users_v, batch_pos_v)
                        k1 = weight1_model()
                        k2 = torch.pow(popularity_matrix[batch_pos_v].to(world.DEVICE),
                                    k1).to(world.DEVICE1)
                        samples_scores_T = teacher_model(batch_users_v, batch_pos_v,k2).to(world.DEVICE)
                        weights = torch.sigmoid(samples_scores_T/2)
                        y_hat_l = one_step_model(batch_users_v, batch_pos_v).to(world.DEVICE)
                        y_hat_l=torch.sigmoid(y_hat_l)
                        loss_l_p = -(weights* torch.log(y_hat_l + 1e-10) +
                                     (1 - weights) * torch.log(1 - y_hat_l + 1e-10))
                        #loss_l_p = -(weights * torch.log(y_hat_l + 1e-10))
                        loss_l_p = loss_l_p.sum(-1).mean()




                        batch_neg_v = batch_neg_v.to(world.DEVICE)
                        # batch_neg_v = batch_neg_v.to(world.DEVICE1)
                        #k1= weight1_model(batch_users_v, batch_neg_v)
                        k1 = weight1_model()
                        k2 = torch.pow(popularity_matrix[batch_neg_v].to(world.DEVICE),
                                       k1).to(world.DEVICE1)
                        samples_scores_T = teacher_model(batch_users_v, batch_neg_v, k2).to(world.DEVICE)
                        weights = torch.sigmoid(samples_scores_T/2)
                        y_hat_l = one_step_model(batch_users_v, batch_neg_v).to(world.DEVICE)
                        y_hat_l = torch.sigmoid(y_hat_l)
                        loss_l_n = -((1 - weights) * torch.log(y_hat_l + 1e-10) +
                                   weights * torch.log(1 - y_hat_l + 1e-10))
                        #loss_l_n=-((1 - weights) * torch.log(1 - y_hat_l + 1e-10))
                        loss_l_n = loss_l_n.sum(-1).mean()
                        print(batch_j)
                        print(weight1_model.getUsersRating())
                        loss_l_a=loss_l_p+loss_l_n
                        weight1_optimizer.zero_grad()
                        loss_l_a.backward()
                        weight1_optimizer.step()
                        cri = loss_l_a.cpu().item()
                        aver_loss_2 += cri

            student_model.train()
            with timer(name="student_model"):
                for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                            utils.minibatch(users,
                                            posItems,
                                            negItems,
                                            batch_size=1024)):

                    batch_users = batch_users.to(world.DEVICE)
                    batch_pos = batch_pos.to(world.DEVICE)
                    batch_neg = batch_neg.to(world.DEVICE)
                    batch_neg, weights, KD_loss_2 = sampler.Sample(
                            batch_users, batch_pos, batch_neg, 100,one_step=0)

                    # batch_pos_pop = torch.pow(popularity_matrix[batch_pos].to(world.DEVICE),
                    #                           weight1_model(batch_users, batch_pos)).to(world.DEVICE1)
                    # batch_neg_pop = torch.pow(popularity_matrix[batch_neg].to(world.DEVICE),
                    #                           weight1_model(batch_users, batch_neg)).to(world.DEVICE1)
                    # with timer(name="BP"):
                    #     cri = bpr.stageTwo(batch_users,
                    #                        batch_pos,
                    #                        batch_neg,
                    #                        None, None,
                    #                        add_loss=KD_loss_2,
                    #                        weights=weights)
                    loss_f, reg_loss = student_model.bpr_loss_pop(batch_users, batch_pos, batch_neg, None,
                                                                   None, weights=weights)
                    reg_loss = reg_loss * world.config['decay']
                    assert KD_loss_2.requires_grad == True
                    loss_f_all = loss_f + KD_loss_2
                    loss_f_all = loss_f_all + reg_loss
                    student_model_optimizer.zero_grad()
                    loss_f_all.backward()
                    student_model_optimizer.step()
                    cri=loss_f_all.cpu().item()
                    aver_loss_3 += cri
                    aver_kd += KD_loss_2.cpu().item()

            aver_loss_1 = aver_loss_1 / total_batch
            aver_loss_2 = aver_loss_3 / total_batch_1
            aver_loss_3 = aver_loss_3 / total_batch
            aver_kd = aver_kd / total_batch
            info = f"{timer.dict()}[loss_1:{aver_loss_1:.5e}][loss_2:{aver_loss_2:.5e}][BPR loss:{aver_loss_3:.5e}]"
            timer.zero()



            print(
                f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {info}'
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





