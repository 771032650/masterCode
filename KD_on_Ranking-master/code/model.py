"""
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
import numpy as np
from torch import nn
from dataloader import BasicDataset

import utils


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg, weights=None):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def pair_score(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            Sigmoid(Score_pos - Score_neg)
        """
        raise NotImplementedError

class DistillEmbedding(BasicModel):
    '''
        student's embedding is not total free
    '''
    def __init__(self, *args):
        super(DistillEmbedding, self).__init__()

    @property
    def embedding_user(self):
        raise NotImplementedError
    @property
    def embedding_item(self):
        raise NotImplementedError



class LightGCN(BasicModel):
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset,
                 fix:bool = False,
                 init=True):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.fix = fix
        self.init_weight(init)

    def init_weight(self, init):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        if self.fix==True:
            self.latent_dim = self.config['teacher_dim']
            self.n_layers = self.config['teacher_layer']
        else:
            self.latent_dim = self.config['latent_dim_rec']
            self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        if init:
            self.embedding_user = torch.nn.Embedding(
                num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.embedding_item = torch.nn.Embedding(
                num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            if self.config['pretrain'] == 0:
                nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
                nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
                # print('use xavier initilizer')
            else:
                self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
                self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
                # print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        if self.fix:
            try:
                return self.all_users, self.all_items
            except:
                print("teacher only comptue once")
                pass
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.all_users = users
        self.all_items = items
        return users, items

    def getUsersRating(self, users, t1=None, t2=None):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items

        if t1 is not None:
            rating = self.f(
                    (torch.matmul(users_emb, items_emb.t()) + t1)/t2
                )
        else:
            rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg, weights=None):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        if weights is not None:
            # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores) * weights)
            x = (pos_scores - neg_scores)
            loss = torch.mean(
                torch.nn.functional.softplus(-x) + (1-weights)*x
            )
        else:
            #loss = torch.mean(torch.nn.functional.softplus(neg_scores -pos_scores.reshape(pos_scores.shape[0],-1)))
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def pair_score(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        return self.f(pos_scores - neg_scores)

    def forward(self, users, items):
        """
        without sigmoid
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items.long()]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class LightEmb(LightGCN):
    def __init__(self,
                 config        : dict,
                 dataset      : BasicDataset,
                 teacher_model: LightGCN):
        super(LightEmb, self).__init__(config, dataset, init=False)
        self.config = config
        self.dataset = dataset
        self.tea = teacher_model
        self.tea.fix = True
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']

        self._embedding_user = Embedding_wrapper(
             num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self._embedding_item = Embedding_wrapper(
             num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        self._user_tea = self.tea.embedding_user.weight.data.to(world.DEVICE)
        self._item_tea = self.tea.embedding_item.weight.data.to(world.DEVICE)
        # print(self._user_tea.requires_grad, self._item_tea.requires_grad)
        # not grad needed for teacher


        self.latent_dim_tea = self.tea.latent_dim
        self.transfer_user = nn.Sequential(
            nn.Linear(self.latent_dim_tea, self.latent_dim),
        )
        self.transfer_item = nn.Sequential(
            nn.Linear(self.latent_dim_tea, self.latent_dim)
        )
        # self.f = nn.Sigmoid()
        self.f = nn.ReLU()
        # self.f = nn.LeakyReLU()

    @property
    def embedding_user(self):
        weights = self.transfer_user(self._user_tea)
        self._embedding_user.pass_weight(weights)
        return self._embedding_user

    @property
    def embedding_item(self):
        weights = self.transfer_item(self._item_tea)
        self._embedding_item.pass_weight(weights)
        return self._embedding_item


class Embedding_wrapper:
    def __init__(self, num_embeddings, embedding_dim):
        self.num = num_embeddings
        self.dim = embedding_dim
        self.weight = None

    def __call__(self,
                 index : torch.Tensor):
        if not isinstance(index, torch.LongTensor):
            index = index.long()
        if self.weight is not None:
            return self.weight[index]
        else:
            raise TypeError("haven't update embedding")

    def pass_weight(self, weight):
        try:
            assert len(weight.shape)
            assert weight.shape[0] == self.num
            assert weight.shape[1] == self.dim
            self.weight = weight
        except AssertionError:
            raise AssertionError(f"weight your pass is wrong! \n expect {self.num}X{self.dim}, but got {weight.shapet}")

    def __repr__(self):
        return f"Emb({self.num} X {self.dim})"


class LightUni(LightGCN):
    def __init__(self, *args, **kwargs):
        super(LightUni, self).__init__(*args, **kwargs)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        if self.fix:
            try:
                return self.all_users, self.all_items
            except:
                print("teacher only comptue once")
                pass
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        users = users/(users.norm(2, dim=1).unsqueeze(1) + 1e-7)
        items = items/(items.norm(2, dim=1).unsqueeze(1) + 1e-7)
        self.all_users = users
        self.all_items = items
        return users, items

class Expert(nn.Module):
    def __init__(self, dims):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(dims[0], dims[1]),nn.Linear(dims[1], dims[2]),nn.ReLU())

    def forward(self, x):
        return self.mlp(x)

class LightExpert(LightGCN):
    def __init__(self,
                     config: dict,
                     dataset: BasicDataset,
                     teacher_model: LightGCN,
                     ):
            super(LightExpert, self).__init__(config, dataset, init=True)
            self.config = config
            self.dataset = dataset
            self.tea = teacher_model
            self.tea.fix = True
            self.de_weight=config['de_weight']


            # Expert Configuration
            self.num_experts =self.config["num_expert"]
            self.latent_dim_tea=self.tea.latent_dim
            expert_dims = [self.latent_dim, (self.latent_dim_tea +self.latent_dim) // 2, self.latent_dim_tea]

            ## for self-distillation
            if self.tea.latent_dim == self.latent_dim:
                expert_dims = [self.latent_dim, self.latent_dim // 2,self.latent_dim_tea]

            self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
            self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

            self.user_selection_net = nn.Sequential(nn.Linear(self.latent_dim_tea, self.num_experts), nn.Softmax(dim=1))
            self.item_selection_net = nn.Sequential(nn.Linear(self.latent_dim_tea, self.num_experts), nn.Softmax(dim=1))

            # num_experts_params = count_parameters(self.user_experts) + count_parameters(self.item_experts)
            # num_gates_params = count_parameters(self.user_selection_net) + count_parameters(self.item_selection_net)

            self.sm = nn.Softmax(dim=1)

            self.T=10


    def get_DE_loss(self, batch_entity, is_user=True):

            if is_user:
                s = self.embedding_user(batch_entity)
                t = self.tea.embedding_user(batch_entity)

                experts = self.user_experts
                selection_net = self.user_selection_net

            else:
                s = self.embedding_item(batch_entity)
                t = self.tea.embedding_item(batch_entity)

                experts = self.item_experts
                selection_net = self.item_selection_net

            selection_dist = selection_net(t)  # batch_size x num_experts

            if self.num_experts == 1:
                selection_result = 1.
            else:
                # Expert Selection
                g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(world.DEVICE)
                eps = 1e-10  # for numerical stability
                selection_dist = selection_dist + eps
                selection_dist = self.sm((selection_dist.log() + g) / self.T)
                selection_dist = torch.unsqueeze(selection_dist, 1)  # batch_size x 1 x num_experts
                selection_result = selection_dist.repeat(1, self.tea.latent_dim,
                                                         1)  # batch_size x teacher_dims x num_experts
            expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)]  # s -> t
            expert_outputs = torch.cat(expert_outputs, -1)  # batch_size x teacher_dims x num_experts
            expert_outputs = expert_outputs * selection_result  # batch_size x teacher_dims x num_experts
            expert_outputs = expert_outputs.sum(2)  # batch_size x teacher_dims
            DE_loss = torch.mean(((t - expert_outputs) ** 2).sum(-1))
            return DE_loss

    def bpr_loss(self, users, pos, neg, weights=None):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        if weights is not None:
            # loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores) * weights)
            x = (pos_scores - neg_scores)
            loss = torch.mean(
                torch.nn.functional.softplus(-x) + (1-weights)*x
            )
        else:
            #loss = torch.mean(torch.nn.functional.softplus(neg_scores -pos_scores.reshape(pos_scores.shape[0],-1)))
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # de_loss_user = self.get_DE_loss(users.long(), is_user=True)
        # de_loss_pos = self.get_DE_loss(pos.long(), is_user=False)
        # de_loss_neg = self.get_DE_loss(neg.long(), is_user=False)
        # de_loss = de_loss_user + de_loss_pos + de_loss_neg
        return loss, reg_loss

class MyModel(PairWiseModel):
    def __init__(self,
                 config        : dict,
                 dataset      : BasicDataset,
                 teacher_model: LightGCN):
        super(MyModel, self).__init__()
        self.config = config
        self.dataset = dataset
        self.tea = teacher_model
        self.tea.fix = True
        self.__init_weight()

    def __init_weight(self):
        self.student_1=LightGCN(self.config,self.dataset)
        self.student_2 = LightGCN(self.config, self.dataset)
        self._user_tea = self.tea.embedding_user.weight.data.to(world.DEVICE)
        self._item_tea = self.tea.embedding_item.weight.data.to(world.DEVICE)
        # print(self._user_tea.requires_grad, self._item_tea.requires_grad)
        # not grad needed for teacher


        self.latent_dim_tea = self.tea.latent_dim
        self.transfer_user_1 = nn.Sequential(
            nn.Linear(self.latent_dim_tea, self.student_1.latent_dim),
        )
        self.transfer_item_1 = nn.Sequential(
            nn.Linear(self.latent_dim_tea, self.student_1.latent_dim),
        )
        self.transfer_user_2 = nn.Sequential(
            nn.Linear(self.latent_dim_tea, self.student_2.latent_dim),
        )
        self.transfer_item_2 = nn.Sequential(
            nn.Linear(self.latent_dim_tea, self.student_2.latent_dim)
        )
        self.f = nn.Sigmoid()
        #self.f = nn.ReLU()
        # self.f = nn.LeakyReLU()


    def get_distill_loss(self,batch_item,batch_user, n_student=1):
        if n_student==1:
            s1 = self.student_1.embedding_user(batch_user)
            s2 = self.student_1.embedding_item(batch_item)
            t1 = self.tea.embedding_user(batch_user)
            t2 = self.tea.embedding_item(batch_item)
            transfer_1=self.transfer_user_1
            transfer_2 = self.transfer_item_1
        else:
            s1 = self.student_2.embedding_user(batch_user)
            s2 = self.student_2.embedding_item(batch_item)
            t1 = self.tea.embedding_user(batch_user)
            t2 = self.tea.embedding_item(batch_item)
            transfer_1 = self.transfer_user_2
            transfer_2 = self.transfer_item_2
        transfer_out_1=torch.sigmoid(transfer_1(t1))
        transfer_out_2=torch.sigmoid(transfer_2(t2))
        out_1=torch.mean(((s1 - transfer_out_1) ** 2).sum(-1))
        out_2 = torch.mean(((s2 - transfer_out_2) ** 2).sum(-1))

        return out_1+out_2

    def get_DE_loss(self, batch_item, is_user=False):
        if is_user:
            s1 = self.student_1.embedding_user(batch_item)
            s2 = self.student_2.embedding_user(batch_item)
        else:
            s1 = self.student_1.embedding_item(batch_item)
            s2 = self.student_2.embedding_item(batch_item)
        out_1 = torch.mean(((s1 - s2) ** 2).sum(-1))
        return torch.reciprocal(out_1)


    def bpr_loss(self, users, pos, neg, weights=None):
        dim_item = neg.shape[-1]
        vector_user = neg.repeat((dim_item, 1)).t().reshape((-1,))
        vector_item = neg.reshape((-1,))

        student_scores= self.student_1(vector_user, vector_item).reshape((-1, dim_item))
        # ----
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(users))
        negitems = neg[idx, top1]
        # ----
        loss1,reg_loss1=self.student_1.bpr_loss(users,pos,negitems,weights)

        student_scores = self.student_2(vector_user, vector_item).reshape((-1, dim_item))
        # ----
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(users))
        negitems= neg[idx, top1]
        loss2, reg_loss2 = self.student_2.bpr_loss(users, pos, negitems, weights)

        return loss1+loss2, reg_loss1+reg_loss2

    def forward(self, users, items):
        """
        without sigmoid
        """
        gamma     = self.student_2(users,items)
        return gamma

    def getUsersRating(self, users, t1=None, t2=None):
        rating=self.student_2.getUsersRating(users)
        return rating

class newModel(LightGCN):
        def __init__(self,
                     config: dict,
                     dataset: BasicDataset,
                     teacher_model: LightGCN):
            super(newModel, self).__init__(config, dataset, init=True)
            self.config = config
            self.dataset = dataset
            self.tea = teacher_model
            self.tea.fix = True
            self.popularity = self.dataset.popularity() + 1
            sum_popularit = np.sum(self.popularity)
            self.popularity = self.popularity / sum_popularit
            self.popularity = torch.tensor(np.power(self.popularity, world.lambda_pop)).float().to(world.DEVICE)



        def bpr_loss(self, users, pos, neg, weights=None):

            (users_emb, pos_emb, neg_emb,
             userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                                  posEmb0.norm(2).pow(2) +
                                  negEmb0.norm(2).pow(2)) / float(len(users))
            pos_scores = torch.mul(users_emb, pos_emb)
            pos_scores = torch.sum(pos_scores, dim=1)
            neg_scores = torch.mul(users_emb, neg_emb)
            neg_scores = torch.sum(neg_scores, dim=1)
            tea_score=self.tea(users,pos)
            pos_popularity = self.popularity[pos]
            kd_loss1 = torch.mean((pos_scores - torch.mul(tea_score,pos_popularity)) ** 2)
            tea_score = self.tea(users, neg)
            pos_popularity = self.popularity[neg]
            kd_loss2 = torch.mean((pos_scores - torch.mul(tea_score, pos_popularity)) ** 2)
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            return loss+(kd_loss1+kd_loss2)*world.kd_weight, reg_loss

