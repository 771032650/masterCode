import world
import torch
import multiprocessing
import numpy as np
from torch.nn.functional import softplus
from time import time
from utils import shapes, combinations, timer
from world import cprint
from model import PairWiseModel, LightGCN,BasicModel
from dataloader import BasicDataset
from torch.nn import Softmax, Sigmoid
import torch.nn.functional as F
import utils
import random

# try:
#     from cppimport import imp_from_filepath
#     from os.path import join, dirname
#     path = join(dirname(__file__), "sources/sampling.cpp")
#     sampling = imp_from_filepath(path)
#     sampling.seed(world.SEED)
#     sample_ext = True
# except:
#     world.cprint("Cpp ext not loaded")
#     sample_ext = False


sample_ext = False
ALLPOS = None
# ----------------------------------------------------------------------------
# distill


def userAndMatrix(batch_users, batch_items, model):
    """cal scores between user vector and item matrix

    Args:
        batch_users (tensor): vector (batch_size)
        batch_items (tensor): matrix (batch_size, dim_item)
        model (PairWiseModel):

    Returns:
        tensor: scores, shape like batch_items
    """
    dim_item = batch_items.shape[-1]
    vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1, ))
    vector_item = batch_items.reshape((-1, ))
    return model(vector_user, vector_item).reshape((-1, dim_item))


class DistillSample:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns_k: int,
                 method: int = 3,
                 beta=world.beta):
        """
            method 1 for convex combination
            method 2 for random indicator
            method 3 for simplified method 2
        """
        self.beta = beta
        self.W = torch.Tensor([world.p0])
        self.dataset = dataset
        self.student = student
        self.teacher = teacher
        # self.methods = {
        #     'combine' : self.convex_combine, # not yet
        #     'indicator' : self.random_indicator,
        #     'simple' : self.max_min,
        #     'weight' : self.weight_pair,
        # }
        self.method = 'combine'
        self.Sample = self.convex_combine
        cprint(f"Using {self.method}")
        self.dns_k = dns_k
        self.soft = Softmax(dim=1)
        # self._generateTopK()

    def PerSample(self, batch=None):
        if batch is not None:
            return UniformSample_DNS_yield(self.dataset,
                                           self.dns_k,
                                           batch_size=batch)
        else:
            return UniformSample_DNS(self.dataset, self.dns_k)

    def _generateTopK(self, batch_size=256):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros((self.dataset.n_users, self.topk))
                for user in range(0, self.dataset.n_users, batch_size):
                    end = min(user + batch_size, self.dataset.n_users)
                    scores = self.teacher.getUsersRating(
                        torch.arange(user, end))
                    pos_item = self.dataset.getUserPosItems(
                        np.arange(user, end))

                    # -----
                    exclude_user, exclude_item = [], []
                    for i, items in enumerate(pos_item):
                        exclude_user.extend([i] * len(items))
                        exclude_item.extend(items)
                    scores[exclude_user, exclude_item] = -1e5
                    # -----
                    _, neg_item = torch.topk(scores, self.topk)
                    self.RANK[user:user + batch_size] = neg_item
        self.RANK = self.RANK.cpu().int().numpy()

    # ----------------------------------------------------------------------------
    # method 1
    def convex_combine(self, batch_users, batch_pos, batch_neg, epoch):
        with torch.no_grad():
            student_score = userAndMatrix(batch_users, batch_neg, self.student)
            teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
            start = time()
            batch_list = torch.arange(0, len(batch_neg))
            pos_score = self.teacher(batch_users, batch_pos).unsqueeze(dim=1)
            margin = pos_score - teacher_score
            refine = margin * student_score
            _, student_max = torch.max(refine, dim=1)
            Items = batch_neg[batch_list, student_max]
            return Items, None, None

    # just DNS
    def DNS(self, batch_neg, scores):
        batch_list = torch.arange(0, len(batch_neg))
        _, student_max = torch.max(scores, dim=1)
        student_neg = batch_neg[batch_list, student_max]
        return student_neg

class SimpleSample:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns_k: int,
                 method: int = 3,
                 beta=world.beta):

        self.beta = beta

        self.dataset = dataset
        self.student = student
        self.teacher = teacher

        self.dns_k=dns_k

    def PerSample(self, batch=None):
        # if batch is not None:
        #     return UniformSample_DNS_yield(self.dataset,
        #                                    self.dns_k,
        #                                    batch_size=batch)
        # else:
        #     return UniformSample_DNS(self.dataset, self.dns_k)
        return Sample_DNS_python(self.dataset, self.dns_k)

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg,
               epoch,
               dynamic_samples=None):
        STUDENT = self.student
        TEACHER = self.teacher
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----

        return negitems, None, None



class RD:
    def __init__(
        self,
        dataset: BasicDataset,
        student: PairWiseModel,
        teacher: PairWiseModel,
        dns,
        topK=10,
        mu=0.1,
        lamda=1,
        teach_alpha=1.0,
        dynamic_sample=100,
        dynamic_start_epoch=200,
    ):
        self.rank_aware = False
        self.dataset = dataset
        self.student = student
        self.teacher = teacher.eval()
        self.RANK = None
        self.epoch = 0

        self._weight_renormalize = True
        self.mu, self.topk, self.lamda = mu, topK, lamda
        self.dynamic_sample_num = dynamic_sample
        self.dns_k = dns
        self.teach_alpha = teach_alpha
        self.start_epoch = dynamic_start_epoch
        self._generateTopK()
        self._static_weights = self._generateStaticWeights()

    def PerSample(self, batch=None):
        # if batch is not None:
        #     return UniformSample_DNS_yield(self.dataset,
        #                                    self.dns_k +
        #                                    self.dynamic_sample_num,
        #                                    batch_size=batch)
        # else:
        #     return UniformSample_DNS(self.dataset,
        #                              self.dns_k + self.dynamic_sample_num)
        #
        self.dynamic_samples=Sample_DNS_python_2(self.dataset,self.dynamic_sample_num)
        self.dynamic_samples=torch.Tensor(self.dynamic_samples).long().cuda()
        self.dynamic_samples=self.dynamic_samples[:,2:]
        return Sample_original(self.dataset)
    def _generateStaticWeights(self):
        w = torch.arange(1, self.topk + 1).float()
        w = torch.exp(-w / self.lamda)
        return (w / w.sum()).unsqueeze(0)

    def _generateTopK(self, batch_size=1024):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros(
                    (self.dataset.n_users, self.topk)).cuda()
                for user in range(0, self.dataset.n_users, batch_size):
                    end = min(user + batch_size, self.dataset.n_users)
                    scores = self.teacher.getUsersRating(
                        torch.arange(user, end))
                    pos_item = self.dataset.getUserPosItems(
                        np.arange(user, end))

                    # -----
                    exclude_user, exclude_item = [], []
                    for i, items in enumerate(pos_item):
                        exclude_user.extend([i] * len(items))
                        exclude_item.extend(items)
                    scores[exclude_user, exclude_item] = -1e5
                    # -----
                    _, neg_item = torch.topk(scores, self.topk)
                    self.RANK[user:user + batch_size] = neg_item

    def _rank_aware_weights(self):
        pass

    def _weights(self, S_score_in_T, epoch, dynamic_scores):
        batch = S_score_in_T.shape[0]
        if epoch < self.start_epoch:
            return self._static_weights.repeat((batch, 1)).cuda()
        with torch.no_grad():
            static_weights = self._static_weights.repeat((batch, 1))
            # ---
            topk = S_score_in_T.shape[-1]
            num_dynamic = dynamic_scores.shape[-1]
            m_items = self.dataset.m_items
            dynamic_weights = torch.zeros(batch, topk)
            for col in range(topk):
                col_prediction = S_score_in_T[:, col].unsqueeze(1)
                num_smaller = torch.sum(col_prediction < dynamic_scores,dim=1).float()
                # print(num_smaller.shape)
                relative_rank = num_smaller / num_dynamic
                appro_rank = torch.floor((m_items - 1) * relative_rank)+1

                dynamic = torch.tanh(self.mu * (appro_rank - col))
                dynamic = torch.clamp(dynamic, min=0.)

                dynamic_weights[:, col] = dynamic.squeeze()
            if self._weight_renormalize:
                return F.normalize(static_weights * dynamic_weights,
                                   p=1,
                                   dim=1).cuda()
            else:
                return (static_weights * dynamic_weights).cuda()

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg,
               epoch,
               dynamic_samples=None):
        STUDENT = self.student
        TEACHER = self.teacher
        #assert batch_neg.shape[-1] == (self.dns_k)
        dynamic_samples = self.dynamic_samples[batch_users]
        #batch_neg = batch_neg[:, :self.dns_k]
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        dynamic_scores = userAndMatrix(batch_users, dynamic_samples,STUDENT).detach()
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        topk_teacher = self.RANK[batch_users]
        topk_teacher = topk_teacher.reshape((-1, )).long()
        user_vector = batch_users.repeat((self.topk, 1)).t().reshape((-1, ))

        S_score_in_T = STUDENT(user_vector, topk_teacher).reshape(
            (-1, self.topk))
        weights = self._weights(S_score_in_T.detach(), epoch, dynamic_scores)
        # RD_loss
        RD_loss = -(weights * torch.log(torch.sigmoid(S_score_in_T)))
        # print("RD shape", RD_loss.shape)
        RD_loss = RD_loss.sum(1)
        RD_loss = RD_loss.sum()

        return negitems, None, RD_loss


class CD:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns,
                 lamda=1,
                 n_distill=10,
                 t1=2,
                 t2=0):
        self.student = student
        self.teacher = teacher.eval()
        self.dataset = dataset
        self.dns_k = dns
        self.sample_num = 100
        self.strategy = "student guide"
        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2
        ranking_list = np.asarray([(i+1)/50 for i in range(self.sample_num)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list=torch.exp(-ranking_list*self.lamda)
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False
        # self.crsNeg = self.dataset.getUserPosItems(range(self.dataset.n_users))
        # neglist = np.arange(self.dataset.m_items)
        # self.userneglist=[]
        # for user in range(self.dataset.n_users):
        #     self.userneglist.append(np.delete(neglist, self.crsNeg[user], -1))
        # with timer(name="CD sample"):
        #     self.get_rank_sample(MODEL=self.teacher)
    def PerSample(self, batch=None):
        if self.strategy == "random":
            MODEL = None
        elif self.strategy == "student guide":
            MODEL = self.student
        elif self.strategy == "teacher guide":
            MODEL = self.teacher
        else:
            raise TypeError("CD support [random, student guide, teacher guide], " \
                            f"But got {self.strategy}")
        with timer(name="CD sample"):
            self.get_rank_sample(MODEL=MODEL)
        # if batch is not None:
        #     return UniformSample_DNS_yield(self.dataset,
        #                                    self.dns_k,
        #                                    batch_size=batch)
        # else:
        #     return UniformSample_DNS(self.dataset, self.dns_k)

        return  Sample_original(self.dataset)
    def Sample(self, batch_users, batch_pos, batch_neg, epoch):
        return self.sample_diff(batch_users, batch_pos, batch_neg,
                                self.strategy)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items,
                                   (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()

    def get_rank_sample(self,MODEL):
        if MODEL is None:
            return self.random_sample(self.dataset.n_users)
        MODEL: LightGCN
        all_items = self.dataset.m_items
        self.rank_samples = torch.zeros(self.dataset.n_users, self.n_distill)
        with torch.no_grad():
            #items_score = MODEL.getUsersRating(batch_users)
            batch_size=1024
            rank_scores = torch.zeros(self.dataset.n_users, self.sample_num)
            rank_items = torch.zeros(self.dataset.n_users, self.sample_num)
            for user in range(0, self.dataset.n_users, batch_size):
                end = min(user + batch_size, self.dataset.n_users)
                scores = MODEL.getUsersRating(
                    torch.arange(user, end))
                pos_item = self.dataset.getUserPosItems(
                    np.arange(user, end))
                exclude_user, exclude_item = [], []
                for i, items in enumerate(pos_item):
                    exclude_user.extend([i] * len(items))
                    exclude_item.extend(items)
                scores[exclude_user, exclude_item] = -1e10
                rank_scores[user:end],  rank_items[user:end] = torch.topk(scores, self.sample_num)
                del scores

            for user in range(self.dataset.n_users):
                ranking_list = self.ranking_mat[user]
                rating=rank_scores[user]
                negitems=rank_items[user]
                sampled_items = set()
                while True:
                    with timer(name="compare"):
                        samples = torch.multinomial(ranking_list, 2, replacement=True)
                        if rating[samples[0]] > rating[samples[1]]:
                            sampled_items.add(negitems[samples[0]])
                        else:
                            sampled_items.add(negitems[samples[1]])
                        if len(sampled_items)>=self.n_distill:
                            break
                self.rank_samples[user] = torch.Tensor(list(sampled_items))
        self.rank_samples=self.rank_samples.cuda().long()


    def sample_diff(self, batch_users, batch_pos, batch_neg, strategy):
        STUDENT = self.student
        TEACHER = self.teacher

        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        random_samples=self.rank_samples[batch_users,:]
        # samples_vector = random_samples.reshape((-1, ))
        samples_scores_T = userAndMatrix(batch_users, random_samples, TEACHER)
        samples_scores_S = userAndMatrix(batch_users, random_samples, STUDENT)
        weights = torch.sigmoid((samples_scores_T + self.t2) / self.t1)
        inner = torch.sigmoid(samples_scores_S)
        CD_loss = -(weights * torch.log(inner + 1e-10) +
                    (1 - weights) * torch.log(1 - inner + 1e-10))
        # print(CD_loss.shape)
        CD_loss = CD_loss.sum(1).sum()
        return negitems, None, CD_loss

    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass


class RRD:
    def __init__(self,dataset: BasicDataset,
                student: PairWiseModel,
                teacher: PairWiseModel,
                 dns,T=100, K=50, L=10):
        self.student = student
        self.teacher = teacher.eval()

        self.dataset = dataset
        self.dns_k = dns
        # for KD
        self.T = T
        self.K = K
        self.L = L
        ranking_list = np.asarray([(i+1) / 50 for i in range(self.T)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list)
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False
        # For uninteresting item
        self.mask = torch.ones((self.dataset.n_users,self.dataset.m_items))
        for user in range(self.dataset.n_users):
            user_pos = self.dataset.getOneUserPosItems(user)
            for item in user_pos:
                self.mask[user][item] = 0
        self.mask.requires_grad = False
        with timer(name="RRD sample"):
            self.RRD_sampling()

    def PerSample(self, batch=None):
        # if batch is not None:
        #     return UniformSample_DNS_yield(self.dataset,
        #                                    self.dns_k,
        #                                    batch_size=batch)
        # else:
        #     return UniformSample_DNS(self.dataset, self.dns_k)

        return Sample_original(self.dataset)
    def get_samples(self, batch_user):

        interesting_samples = torch.index_select(self.interesting_items, 0, batch_user)
        uninteresting_samples = torch.index_select(self.uninteresting_items, 0, batch_user)

        return interesting_samples, uninteresting_samples

    # epoch 마다
    def RRD_sampling(self):
            with torch.no_grad():
                # interesting items
                self.interesting_items = torch.zeros((self.dataset.n_users, self.K))
                for user in range(self.dataset.n_users):
                    crsNeg = self.dataset.getOneUserPosItems(user)
                    neglist=np.arange(self.dataset.m_items)
                    neglist= np.delete(neglist,crsNeg,-1)
                    negItems = torch.LongTensor(neglist).cuda()
                    n_pos=500
                    samples = torch.multinomial(self.ranking_mat[user,: self.T], self.K, replacement=False)
                    samples = samples.sort(dim=-1)[0].cuda()
                    rating=userAndMatrix(torch.tensor(user), negItems, self.teacher).reshape((-1))
                    n_rat=rating.sort(dim=-1,descending=True)[1]
                    negItems=negItems[n_rat]
                    self.interesting_items[torch.tensor(user)] = negItems[samples]
                    self.mask[user][negItems[: self.T]]=0
                    del samples
                    del negItems
                    del rating
                self.interesting_items = self.interesting_items.cuda()
                # uninteresting items
                m1 = self.mask[:self.dataset.n_users//2,:]
                tmp1 = torch.multinomial(m1, self.L, replacement=False)
                del m1
                m2 = self.mask[self.dataset.n_users//2:,:]
                tmp2 = torch.multinomial(m2, self.L, replacement=False)
                del m2
                self.uninteresting_items =torch.cat((tmp1,tmp2),dim=0).cuda()



    def relaxed_ranking_loss(self,S1,S2):
        above = S1.sum(1, keepdim=True)

        below1 = S1.flip(-1).exp().cumsum(1)
        below2 = S2.exp().sum(1, keepdim=True)

        below = (below1 + below2).log().sum(1, keepdim=True)

        return above - below

    def Sample(self,
               batch_users,
               batch_pos,
               batch_neg, epoch):
        STUDENT = self.student
        TEACHER = self.teacher
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        interesting_users, uninteresting_users = self.get_samples(batch_users)

        # interesting_users = interesting_users.reshape((-1,)).long()
        # uninteresting_users = uninteresting_users.reshape((-1,)).long()
        # vector_users = batch_users.repeat((self.K, 1)).t().reshape((-1,))
        # interesting_user_prediction =(STUDENT(vector_users,interesting_users).reshape(
        #     (-1, self.K)))
        # vector_users = batch_users.repeat((self.L, 1)).t().reshape((-1,))
        # uninteresting_user_prediction = (STUDENT(vector_users,uninteresting_users).reshape(
        #     (-1, self.L)))
        interesting_user_prediction=torch.sigmoid(userAndMatrix(batch_users, interesting_users.long(), STUDENT))
        uninteresting_user_prediction = torch.sigmoid(userAndMatrix(batch_users, uninteresting_users.long(), STUDENT))
        RRD_loss = self.relaxed_ranking_loss(interesting_user_prediction,uninteresting_user_prediction)
        RRD_loss=-torch.sum(RRD_loss)

        user_de = STUDENT.get_DE_loss(batch_users.long(), is_user=True)
        interesting_users = interesting_users.reshape((-1,)).long()
        interesting_de = STUDENT.get_DE_loss(interesting_users, is_user=False)
        uninteresting_users = uninteresting_users.reshape((-1,)).long()
        uninteresting_de = STUDENT.get_DE_loss(uninteresting_users, is_user=False)
        DE_loss=user_de+interesting_de+uninteresting_de

        return negitems,None,RRD_loss*world.kd_weight+DE_loss*world.de_weight




class UD:
    def __init__(self,
                 dataset: BasicDataset,
                 student: BasicModel,
                 teacher: BasicModel,

                 weight1_model:torch.nn.Module,
                 weight2_model: torch.nn.Module,
                 dns,
                 lamda=1,
                 n_distill=10,
                 t1=1,
                 t2=0):
        self.student = student
        self.teacher = teacher.eval()

        self.weight1_model=weight1_model
        self.weight2_model = weight2_model
        self.dataset = dataset
        self.dns_k = dns


        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2
        self.sample_num=100
        ranking_list = np.asarray([i/50 for i in range(self.sample_num)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list * self.lamda)
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False

        # with timer(name="UD sample"):
        #     self.get_rank_sample(MODEL=self.teacher)

    def PerSample(self, batch=None):

        with timer(name="UD sample"):
            self.get_rank_sample(MODEL=self.student)

        return Sample_original(self.dataset)

    def Sample(self, batch_users, batch_pos, batch_neg, epoch,one_step):
        return self.sample_diff(batch_users, batch_pos, batch_neg,
                                epoch,one_step)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items,
                                   (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()



    def get_rank_sample(self,MODEL):
        if MODEL is None:
            return self.random_sample(self.dataset.n_users)
        MODEL: LightGCN
        all_items = self.dataset.m_items
        self.rank_samples = torch.zeros(self.dataset.n_users, self.n_distill)
        with torch.no_grad():
            #items_score = MODEL.getUsersRating(batch_users)
            batch_size=1024
            rank_scores = torch.zeros(self.dataset.n_users, self.sample_num)
            rank_items = torch.zeros(self.dataset.n_users,self.sample_num)

            #weight_pop = self.weight_model()
            # weight_pop = torch.pow(self.dataset.expo_popularity, 0.1)
            # self.teacher.set_popularity(weight_pop)
            for user in range(0, self.dataset.n_users, batch_size):
                end = min(user + batch_size, self.dataset.n_users)
                scores = torch.clamp(MODEL.getUsersRating(
                    torch.arange(user, end).cuda().long()), min=0)
                pos_item = self.dataset.getUserPosItems(
                    np.arange(user, end))
                exclude_user, exclude_item = [], []
                for i, items in enumerate(pos_item):
                    exclude_user.extend([i] * len(items))
                    exclude_item.extend(items)
                scores[exclude_user, exclude_item] = -1e10
                rank_scores[user:end],  rank_items[user:end] = torch.topk(scores, self.sample_num)
                del scores

            for user in range(self.dataset.n_users):
                ranking_list = self.ranking_mat[user]
                negitems=rank_items[user]
                samples = torch.multinomial(ranking_list, self.n_distill, replacement=False)
                sampled_items=negitems[samples]
                self.rank_samples[user] = torch.Tensor(list(sampled_items))

        self.rank_samples=self.rank_samples.cuda().long()


    def sample_diff(self, batch_users, batch_pos, batch_neg, one_step_model,one_step):
        TEACHER = self.teacher
        if one_step==1:
            STUDENT=one_step_model
        else:
            STUDENT=self.student
        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        random_samples=self.rank_samples[batch_users,:]

        dim_item = random_samples.shape[-1]
        vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1,))
        vector_item = random_samples.reshape((-1,))
        #weight_pop = self.weight_model(vector_user, vector_item,self.dataset.expo_popularity[vector_item])
        #k_mu = self.weight2_model()
        k_mu=1
        if world.mate_model==0:
            weight_pop = self.weight1_model()

        elif world.mate_model==2:
            weight_pop=self.weight1_model(vector_user, vector_item)

        else:
            weight_pop = self.weight1_model(vector_item)

        #weight_pop=world.lambda_pop
        weight_pop=torch.pow(self.dataset.expo_popularity[vector_item],weight_pop)
        samples_scores_T=TEACHER(vector_user, vector_item,weight_pop).reshape((-1, dim_item)).cuda()
        #samples_scores_T=userAndMatrix(batch_users.cuda(), random_samples.cuda(), TEACHER).cuda()
        samples_scores_S = userAndMatrix(batch_users, random_samples, STUDENT)
        #weights = torch.sigmoid((samples_scores_T + self.t2) / self.t1)

        weights=torch.sigmoid(samples_scores_T/1)
        inner = torch.sigmoid(samples_scores_S)
        #inner=samples_scores_S
        UD_loss = -k_mu*(weights * torch.log(inner + 1e-10) +
                        (1 - weights) * torch.log(1 - inner + 1e-10))
        #print(samples_scores_T.shape)
        #UD_loss = UD_loss.sum(-1).mean()
        UD_loss = UD_loss.sum(-1).sum()
        #UD_loss = UD_loss.sum(-1)
        #print(UD_loss)
        return negitems, None, UD_loss,vector_item,vector_user

    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass


class new_UD:
    def __init__(self,
                 dataset: BasicDataset,
                 student: BasicModel,
                 teacher: BasicModel,

                 weight1_model:torch.nn.Module,
                 weight2_model: torch.nn.Module,
                 dns,
                 lamda=1,
                 n_distill=10,
                 t1=1,
                 t2=0):
        self.student = student
        self.teacher = teacher.eval()

        self.weight1_model=weight1_model
        self.weight2_model = weight2_model
        self.dataset = dataset
        self.dns_k = dns


        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2
        self.sample_num=100
        ranking_list = np.asarray([i/50 for i in range(self.sample_num)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list * self.lamda)
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False

        # with timer(name="UD sample"):
        #     self.get_rank_sample(MODEL=self.teacher)

    def PerSample(self, batch=None):

        with timer(name="UD sample"):
            self.get_rank_sample(MODEL=self.student)

        return Sample_original(self.dataset)

    def Sample(self, batch_users, batch_pos, batch_neg, epoch,one_step):
        return self.sample_diff(batch_users, batch_pos, batch_neg,
                                epoch,one_step)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items,
                                   (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()



    def get_rank_sample(self,MODEL):
        if MODEL is None:
            return self.random_sample(self.dataset.n_users)
        MODEL: LightGCN
        all_items = self.dataset.m_items
        self.rank_samples = torch.zeros(self.dataset.n_users, self.n_distill)
        with torch.no_grad():
            #items_score = MODEL.getUsersRating(batch_users)
            batch_size=1024
            rank_scores = torch.zeros(self.dataset.n_users, self.sample_num)
            rank_items = torch.zeros(self.dataset.n_users,self.sample_num)

            #weight_pop = self.weight_model()
            # weight_pop = torch.pow(self.dataset.expo_popularity, 0.1)
            # self.teacher.set_popularity(weight_pop)
            for user in range(0, self.dataset.n_users, batch_size):
                end = min(user + batch_size, self.dataset.n_users)
                scores = torch.clamp(MODEL.getUsersRating(
                    torch.arange(user, end).cuda().long()), min=0)
                pos_item = self.dataset.getUserPosItems(
                    np.arange(user, end))
                exclude_user, exclude_item = [], []
                for i, items in enumerate(pos_item):
                    exclude_user.extend([i] * len(items))
                    exclude_item.extend(items)
                scores[exclude_user, exclude_item] = -1e10
                rank_scores[user:end],  rank_items[user:end] = torch.topk(scores, self.sample_num)
                del scores

            for user in range(self.dataset.n_users):
                ranking_list = self.ranking_mat[user]
                negitems=rank_items[user]
                samples = torch.multinomial(ranking_list, self.n_distill, replacement=False)
                sampled_items=negitems[samples]
                self.rank_samples[user] = torch.Tensor(list(sampled_items))

        self.rank_samples=self.rank_samples.cuda().long()


    def sample_diff(self, batch_users, batch_pos, batch_neg, one_step_model,one_step):
        TEACHER = self.teacher
        if one_step==1:
            STUDENT=one_step_model
        else:
            STUDENT=self.student
        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        random_samples=self.rank_samples[batch_users,:]

        dim_item = random_samples.shape[-1]
        vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1,))
        vector_item = random_samples.reshape((-1,))
        weight_pop=self.weight1_model(vector_user, vector_item)
        samples_scores_S = userAndMatrix(batch_users, random_samples, STUDENT)
        weights=weight_pop.reshape((-1, dim_item))
        inner = torch.sigmoid(samples_scores_S)
        UD_loss = -(weights * torch.log(inner + 1e-10) +
                        (1 - weights) * torch.log(1 - inner + 1e-10))
        UD_loss = UD_loss.sum(-1).sum()
        return negitems, None, UD_loss,self.rank_samples,batch_users
    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass



class PD:
    def __init__(self,
                 dataset: BasicDataset,
                 student: PairWiseModel,
                 teacher: PairWiseModel,
                 dns,
                 lamda=1,
                 n_distill=10,
                 t1=1,
                 t2=0):
        self.student = student
        self.teacher = teacher.eval()
        self.dataset = dataset
        self.dns_k = dns


        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2
        self.sample_num=100
        ranking_list = np.asarray([(i+1)/50 for i in range(self.sample_num)])
        ranking_list = torch.FloatTensor(ranking_list)
        ranking_list = torch.exp(-ranking_list * self.lamda)
        self.ranking_mat = torch.stack([ranking_list] * self.dataset.n_users, 0)
        self.ranking_mat.requires_grad = False

        # with timer(name="UD sample"):
        #     self.get_rank_sample(MODEL=self.teacher)

    def PerSample(self, batch=None):

        with timer(name="UD sample"):
            self.get_rank_sample(MODEL=self.student)

        return Sample_original(self.dataset)

    def Sample(self, batch_users, batch_pos, batch_neg, epoch):
        return self.sample_diff(batch_users, batch_pos, batch_neg,
                                epoch)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items,
                                   (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().cuda()



    def get_rank_sample(self,MODEL):
        if MODEL is None:
            return self.random_sample(self.dataset.n_users)
        MODEL: LightGCN
        all_items = self.dataset.m_items
        self.rank_samples = torch.zeros(self.dataset.n_users, self.n_distill)
        with torch.no_grad():
            #items_score = MODEL.getUsersRating(batch_users)
            batch_size=1024
            rank_scores = torch.zeros(self.dataset.n_users, self.sample_num)
            rank_items = torch.zeros(self.dataset.n_users,self.sample_num)
            for user in range(0, self.dataset.n_users, batch_size):
                end = min(user + batch_size, self.dataset.n_users)
                scores = torch.clamp(MODEL.getUsersRating(
                    torch.arange(user, end).cuda().long()),min=0)
                pos_item = self.dataset.getUserPosItems(
                    np.arange(user, end))
                exclude_user, exclude_item = [], []
                for i, items in enumerate(pos_item):
                    exclude_user.extend([i] * len(items))
                    exclude_item.extend(items)
                scores[exclude_user, exclude_item] = -1e10
                rank_scores[user:end],  rank_items[user:end] = torch.topk(scores, self.sample_num)
                del scores

            for user in range(self.dataset.n_users):
                ranking_list = self.ranking_mat[user]
                negitems=rank_items[user]
                samples = torch.multinomial(ranking_list,  self.n_distill, replacement=False)
                sampled_items=negitems[samples]
                # samples2= random.sample(range(100,len(negitems),1),90)
                # sampled_items2=negitems[samples2]
                # sampled_items=torch.cat((sampled_items,sampled_items2),dim=-1)
                self.rank_samples[user] = torch.Tensor(list(sampled_items))

        self.rank_samples=self.rank_samples.cuda().long()


    def sample_diff(self, batch_users, batch_pos, batch_neg, epoch):
        STUDENT = self.student
        TEACHER = self.teacher

        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        random_samples=self.rank_samples[batch_users,:]

        dim_item = random_samples.shape[-1]
        vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1,))
        vector_item = random_samples.reshape((-1,))

        pop=torch.from_numpy(self.dataset.expo_popularity[vector_item.cpu()]).float().cuda()
        samples_scores_T = TEACHER(vector_user, vector_item,pop ).reshape((-1, dim_item))
        #samples_scores_T=userAndMatrix(batch_users, random_samples, TEACHER)
        samples_scores_S = userAndMatrix(batch_users, random_samples, STUDENT)
        #weights = torch.sigmoid((samples_scores_T + self.t2) / self.t1)
        #print(samples_scores_T)
        weights=torch.sigmoid(samples_scores_T/1)
        #weights=torch.sigmoid(samples_scores_T)
        inner = torch.sigmoid(samples_scores_S)
        #inner=samples_scores_S
        UD_loss = -(weights * torch.log(inner + 1e-10) +
                        (1 - weights) * torch.log(1 - inner + 1e-10))
        #print(samples_scores_T.shape)
        UD_loss = UD_loss.sum(-1).sum()

        #print(UD_loss)
        return negitems, None, UD_loss

    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass


# ==============================================================
# NON-EXPERIMENTAL PART
# ==============================================================


# ----------
# uniform sample
def UniformSample_original(dataset):
    """
    the original implement of BPR Sampling in LightGCN
    NOTE: we sample a whole epoch data at one time
    :return:
        np.array
    """
    return UniformSample_DNS(dataset, 1)

 # sample
def Sample_original(dataset):
    """
    the original implement of BPR Sampling in LightGCN
    NOTE: we sample a whole epoch data at one time
    :return:
        np.array
    """
    return Sample_DNS_python(dataset, 1)

# ----------
# Dns sampling
def UniformSample_DNS_yield(dataset, dns_k, batch_size=None):
    """Generate train samples(already shuffled)

    Args:
        dataset (BasicDataset)
        dns_k ([int]): How many neg samples for one (user,pos) pair
        batch_size ([int], optional) Defaults to world.config['bpr_batch_size'].

    Returns:
        [ndarray]: yield (batch_size, 2+dns_k)
    """
    dataset: BasicDataset
    batch_size = batch_size or world.config['bpr_batch_size']
    allPos = dataset.allPos
    All_users = np.random.randint(dataset.n_users,
                                  size=(dataset.trainDataSize, ))
    for batch_i in range(0, len(All_users), batch_size):
        batch_users = All_users[batch_i:batch_i + batch_size]
        if sample_ext:
            yield sampling.sample_negative_ByUser(batch_users, dataset.m_items,
                                                  allPos, dns_k)
        else:
            yield UniformSample_DNS_python_ByUser(dataset, batch_users, dns_k)


def UniformSample_DNS(dataset, dns_k, add_pos=None):
    """Generate train samples(sorted)

    Args:
        dataset ([BasicDataset])
        dns_k ([int]): How many neg samples for one (user,pos) pair

    Returns:
        [ndarray]: shape (The num of interactions, 2+dns_k)
    """
    dataset: BasicDataset
    allPos = dataset.allPos
    if add_pos is not None:
        assert len(allPos) == len(add_pos)
        for i in range(len(allPos)):
            allPos.append(np.asarray(add_pos, dtype=np.int))
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, dns_k)
        return S
    else:
        return UniformSample_DNS_python(dataset, dns_k)


def UniformSample_DNS_python(dataset, dns_k):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    per_user_num = user_num // dataset.n_users + 1
    allPos = dataset.allPos
    S = []
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        for i in range(per_user_num):
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [user, positem, *negitems]
            S.append(add_pair)
    return S


def UniformSample_DNS_python_ByUser(dataset, users, dns_k):
    """python implementation for 
    cpp ext 'sampling.sample_negative_ByUser' in sources/sampling.cpp
    """
    dataset: BasicDataset
    allPos = dataset.allPos
    S = np.zeros((len(users), 2 + dns_k))
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
        negitems = NEGforUser[negindex]
        S[i] = [user, positem, *negitems]
    return S

def Sample_DNS_python(dataset, dns_k):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    allPos = dataset.allPos
    S = []
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        per_user_num=len(posForUser)
        for i in range(per_user_num):
            positem = posForUser[i]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            pos_pop = dataset.expo_popularity[positem]
            neg_pop = dataset.expo_popularity[negitems]
            add_pair = [user, positem, negitems,pos_pop,neg_pop]
            S.append(add_pair)
    return S

def Sample_DNS_python_2(dataset, dns_k):
    """python implementation for 'UniformSample_DNS'
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    allPos = dataset.allPos
    S = []
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        per_user_num=len(posForUser)
        for i in range(per_user_num):
            positem = posForUser[i]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [user, positem, *negitems]
            S.append(add_pair)
    return S

def DNS_sampling_neg(batch_users, batch_neg, dataset, recmodel):
    """Dynamic Negative Choosing.(return max in Neg)

    Args:
        batch_users ([tensor]): shape (batch_size, )
        batch_neg ([tensor]): shape (batch_size, dns_k)
        dataset ([BasicDataset])
        recmodel ([PairWiseModel])

    Returns:
        [tensor]: Vector of negitems, shape (batch_size, ) 
                  corresponding to batch_users
    """
    dns_k = world.DNS_K
    with torch.no_grad():

        scores = userAndMatrix(batch_users, batch_neg, recmodel)

        _, top1 = scores.max(dim=1)
        idx = torch.arange(len(batch_users)).cuda()
        negitems = batch_neg[idx, top1]
    return negitems

def generator_n_batch_with_pop(dataset):
    '''
    sample n_batch data with popularity (PD/PDA)
    '''
    # print('start')
    s_time = time()
    all_users = list(dataset.train_user_list.keys())

    S = []
    for u in all_users:
        u_clicked_items = dataset.train_user_list[u]
        u_clicked_times = dataset.train_user_list_time[u]
        for idx in range(len(u_clicked_items)):
            one_pos_item = u_clicked_items[idx]
            u_pos_time = u_clicked_times[idx]
            while True:
                neg_item = random.choice(dataset.items)
                if neg_item not in u_clicked_items:
                    break
            pos_pop=dataset.expo_popularity[one_pos_item]
            neg_pop = dataset.expo_popularity[neg_item]
            one_user = [u,one_pos_item,neg_item,pos_pop,neg_pop]
            S.append(one_user)
    return S


if __name__ == "__main__":
    method = UniformSample_DNS
    from register import dataset
    from utils import timer
    for i in range(1):
        with timer():
            # S = method(dataset, 1)
            S = UniformSample_original(dataset)
            print(len(S[S >= dataset.m_items]))
            S = torch.from_numpy(S).long()
            print(len(S[S >= dataset.m_items]))
        print(timer.get())
