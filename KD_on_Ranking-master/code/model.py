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

def GrpupBypopularity(dataset):
    popularity = dataset.popularity() + 1
    sum_popularit = np.sum(popularity)
    popularity = popularity / sum_popularit
    popularity = np.power(popularity, world.de_weight)
    mappings=utils.split_item_popularity(popularity,int(world.lambda_pop))
    # mappings = utils.map_item_N(popularity,
    #                        [0.2,0.2,0.2,0.2,0.2])
    popularityGroup=np.zeros([len(popularity)])
    apt = 0
    for mapping in mappings:
        count = list(map(lambda x: x in mapping, range(len(popularity))))
        for k in range(len(count)):
            if count[k]==True:
                popularityGroup[k]=apt
        apt=apt+1

    #metrics['precisionbygroup']=APT(groundTrue,mappings=mapping)
    return popularityGroup


class MyModel(LightGCN):
    def __init__(self,
                 config        : dict,
                 dataset      : BasicDataset,):
        super(MyModel, self).__init__(config, dataset, init=True,fix=False)
        self.config = config
        self.dataset = dataset
        self.popularityGroup=torch.from_numpy(GrpupBypopularity(self.dataset))

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        splite_k=int(world.lambda_pop)
        meanPosEmbed = torch.mean(all_items, dim=0)
        #meanPosEmbed=meanPosEmbed.detach()
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        for i in range(len(pos_items)):
            for k in range(1,splite_k):
                if self.popularityGroup[pos_items[i]]==k:
                    pos_emb[i]=torch.cat((pos_emb[i][:-round(world.config['latent_dim_rec']/splite_k*k)],meanPosEmbed[-round(world.config['latent_dim_rec']/splite_k*k):]),dim=-1)
                    break

        for i in range(len(neg_items)):
            for k in range(1, splite_k):
                if self.popularityGroup[neg_items[i]] == k:
                    neg_emb[i] = torch.cat((neg_emb[i][:-round(world.config['latent_dim_rec'] / splite_k * k)],
                                            meanPosEmbed[-round(world.config['latent_dim_rec'] / splite_k * k):]),
                                           dim=-1)
                    break

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

  



def variantELU(scores):
    n=len(scores)
    for i in range(n):
        if scores[i]>0:
            scores[i]=scores[i]+1
        else:
            scores[i]=torch.exp(scores[i])
    return scores

class UnbaisLGN(LightGCN):
        def __init__(self,
                     config: dict,
                     dataset: BasicDataset):
            super(UnbaisLGN, self).__init__(config, dataset, init=True,fix=False)
            self.config = config
            self.dataset = dataset
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
            neg_popularity = self.popularity[neg]
            pos_popularity = self.popularity[pos]
            loss = torch.mean(torch.nn.functional.softplus(variantELU(neg_scores)*neg_popularity- variantELU(pos_scores)*pos_popularity))
            return loss , reg_loss

        # def getUsersRating(self, users, t1=None, t2=None):
        #     all_users, all_items = self.computer()
        #     users_emb = all_users[users.long()]
        #     items_emb = all_items
        #
        #     if t1 is not None:
        #         rating = self.f(
        #                 (torch.matmul(users_emb, items_emb.t()) + t1)/t2
        #             )
        #     else:
        #         rating = torch.matmul(users_emb, items_emb.t())
        #     return rating*self.popularity


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
            tea_all_user,tea_all_item=self.tea.computer()
            tea_users_emb = tea_all_user[users.long()]
            tea_pos_emb = tea_all_item[pos.long()]
            tea_neg_emb = tea_all_item[neg.long()]
            tea_pos_scores = torch.mul(tea_users_emb[:,:-world.config['latent_dim_rec']], tea_pos_emb[:,:-world.config['latent_dim_rec']])
            tea_pos_scores = torch.sum(tea_pos_scores, dim=1)
            tea_neg_scores = torch.mul(tea_users_emb[:,:-world.config['latent_dim_rec']], tea_neg_emb[:,:-world.config['latent_dim_rec']])
            tea_neg_scores = torch.sum(tea_neg_scores, dim=1)
            loss = torch.mean(torch.nn.functional.softplus(neg_scores+0.5*tea_neg_scores - pos_scores+0.5*tea_pos_scores))
            return loss, reg_loss


class ConditionalBPRMF(BasicModel):
    '''
    PD/PDA
    PDG/PDG-A
    '''

    def __init__( self,config:dict,
                 dataset:BasicDataset,
                 fix:bool = False,
                 init=True):
        super(ConditionalBPRMF, self).__init__()
        self.config=config
        self.dataset=dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.fix=fix
        self.weights = self.init_weights(init)
        self._statistics_params()

    def init_weights(self,init):

            self.latent_dim = self.config['latent_dim_rec']
            self.n_layers = self.config['lightGCN_n_layers']
            self.keep_prob = self.config['keep_prob']
            self.A_split = self.config['A_split']
            weights = dict()
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
                weights['user_embedding'] = self.embedding_user.weight
                weights['item_embedding'] = self.embedding_item.weight
            self.f = nn.Sigmoid()
            self.felu=nn.ELU()
            return weights

    def set_popularity(self,last_popularity):
        self.last_popularity = last_popularity
        self.last_popularity = torch.Tensor(self.last_popularity).to(world.DEVICE)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        return users_emb, items_emb

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items = torch.Tensor(range(self.dataset.m_items)).long().to(world.DEVICE)
        items_emb = all_items[items]
        rating = torch.matmul(users_emb, items_emb.t())
        rating=torch.relu(rating)
        rating=self.felu(rating) + 1
        rating = rating * self.last_popularity[items]
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb,pos_emb,neg_emb,users_emb_ego, pos_emb_ego, neg_emb_ego

    def create_bpr_loss_with_pop_global(self, users, pos_items,
                                        neg_items,pos_pops,neg_pops):  # this global does not refer to global popularity, just a name
        (users_emb, pos_emb, neg_emb,users_emb_ego, pos_emb_ego, neg_emb_ego) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        pos_scores = torch.sum(users_emb*pos_emb,dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(users_emb*neg_emb,dim=1)
        # item stop
        pos_scores = self.felu(pos_scores) + 1
        neg_scores = self.felu(neg_scores) + 1
        pos_scores_with_pop = pos_scores*pos_pops
        neg_scores_with_pop = neg_scores*neg_pops

        maxi = torch.log(self.f(pos_scores_with_pop - neg_scores_with_pop) + 1e-10)
        #self.condition_ratings = (self.felu(self.batch_ratings) + 1) * pos_pops.squeeze()
        #self.mf_loss_ori = -(torch.mean(maxi))
        mf_loss = torch.neg(torch.mean(maxi))
        # fsoft=nn.Softplus()
        # mf_loss = -torch.mean(fsoft(pos_scores_with_pop - neg_scores_with_pop))
        # regular
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)) / float(len(users))
        return mf_loss, reg_loss


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.shape  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)

    def bpr_loss_pop(self, users, pos_items,neg_items,pos_pops,neg_pops,weights=None):
        return  self.create_bpr_loss_with_pop_global(users, pos_items,neg_items,pos_pops,neg_pops)

    def forward(self, users, items):
        """
        without sigmoid
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        rating = torch.sum(users_emb * items_emb, dim=1)

        rating = self.felu(rating) + 1
        items_pop = self.last_popularity[items]
        rating=rating * items_pop
        return rating



class BPRMF(BasicModel):
    '''
    PD/PDA
    PDG/PDG-A
    '''

    def __init__( self,config:dict,
                 dataset:BasicDataset,
                 fix:bool = False,
                 init=True):
        super(BPRMF, self).__init__()
        self.config=config
        self.dataset=dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.fix=fix
        self.weights = self.init_weights(init)
        self._statistics_params()

    def init_weights(self,init):

            self.latent_dim = self.config['latent_dim_rec']
            self.n_layers = self.config['lightGCN_n_layers']
            self.keep_prob = self.config['keep_prob']
            self.A_split = self.config['A_split']
            weights = dict()
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
                weights['user_embedding'] = self.embedding_user.weight
                weights['item_embedding'] = self.embedding_item.weight
            self.f = nn.Sigmoid()
            self.felu=nn.ReLU()
            return weights
    
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        return users_emb, items_emb

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items = torch.Tensor(range(self.dataset.m_items)).long().to(world.DEVICE)
        items_emb = all_items[items]
        rating = torch.matmul(users_emb, items_emb.t())
        #rating = torch.relu(rating)
        #rating=self.felu(rating) + 1
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

    def create_bpr_loss(self, users, pos_items,neg_items):
        (users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        pos_scores = torch.sum(users_emb*pos_emb,dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(users_emb*neg_emb,dim=1)
        # item stop
        # pos_scores = self.felu(pos_scores)
        # neg_scores = self.felu(neg_scores)

        maxi = torch.log(self.f(pos_scores - neg_scores) + 1e-10)
        #self.condition_ratings = (self.felu(self.batch_ratings) + 1) * pos_pops.squeeze()
        #self.mf_loss_ori = -(torch.mean(maxi))
        mf_loss = torch.neg(torch.mean(maxi))
        # fsoft=nn.Softplus()
        # mf_loss = -torch.mean(fsoft(pos_scores_with_pop - neg_scores_with_pop))
        # regular
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2)) / float(len(users))
        return mf_loss, reg_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.shape  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)

    def bpr_loss_pop(self, users, pos_items,neg_items,pos_pops,neg_pops,weights=None):
        return  self.create_bpr_loss(users, pos_items,neg_items)

    def bpr_loss(self, users, pos_items,neg_items,weights=None):
        return  self.create_bpr_loss(users, pos_items,neg_items)

    def forward(self, users, items):
        """
        without sigmoid
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        rating = torch.sum(users_emb*items_emb,dim=1)
        #rating = torch.relu(rating)
        #rating = self.felu(rating) + 1
        return rating