"""
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
import numpy as np
from torch import nn
from dataloader import BasicDataset
from torch.autograd import Variable
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
        self._user_tea = self.tea.embedding_user.weight.data.cuda()
        self._item_tea = self.tea.embedding_item.weight.data.cuda()
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
        self.mlp = nn.Sequential()
        L1=nn.Linear(dims[0], dims[1])
        L2=nn.Linear(dims[1], dims[2])
        self.mlp.add_module("L1",L1)
        self.mlp.add_module("L2", L2)
        self.mlp.add_module("r", nn.ReLU())

    def forward(self, x):
        return self.mlp(x)

class LightExpert(LightGCN):
    def __init__(self,
                     config: dict,
                     dataset: BasicDataset,
                     teacher_model: LightGCN,
                     ):
            super(LightExpert, self).__init__()
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
                g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).cuda()
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




  



def variantELU(scores):
    n=len(scores)
    for i in range(n):
        if scores[i]>0:
            scores[i]=scores[i]+1
        else:
            scores[i]=torch.exp(scores[i])
    return scores


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(BasicModel):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def __init__(self):
        super(MetaModule, self).__init__()

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaEmbed(MetaModule):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        ignore = nn.Embedding(dim_1, dim_2)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))


    def forward(self):
        return self.weight

    def named_leaves(self):
        return [('weight', self.weight)]





class ConditionalBPRMF(MetaModule):
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
        self.dataset = dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim=self.config['latent_dim_rec']
        self.fix=fix
        self.init_weights(init)

    def init_weights(self,init):
        self.embedding_user = MetaEmbed(self.num_users, self.latent_dim)
        self.embedding_item = MetaEmbed(self.num_items, self.latent_dim)
        # self.embedding_user = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out', a=0)

        self.f = nn.Sigmoid()
        self.felu=nn.ELU()

    def set_popularity(self, last_popularity):
        self.last_popularity = last_popularity
        self.last_popularity = torch.Tensor(self.last_popularity).cuda()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # users_emb=torch.nn.functional.normalize(users_emb,p=2,dim=-1)
        # items_emb = torch.nn.functional.normalize(items_emb,p=2, dim=-1)
        return users_emb, items_emb

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items = torch.Tensor(range(self.dataset.m_items)).long().cuda()
        items_emb = all_items[items]
        rating = torch.matmul(users_emb, items_emb.t())
        #rating=self.felu(rating) + 1
        #rating = torch.sigmoid(torch.relu(rating))
        #rating = torch.sigmoid(rating/2)
        #rating = (rating * self.last_popularity[items])
        #rating=torch.sigmoid(rating)
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        # all_users, all_items = self.computer()
        # users_emb = all_users[users]
        # pos_emb = all_items[pos_items]
        # neg_emb = all_items[neg_items]
        # users_emb_ego = self.embedding_user(users)
        # pos_emb_ego = self.embedding_item(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        # return users_emb,pos_emb,neg_emb,users_emb_ego, pos_emb_ego, neg_emb_ego

        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # users_emb_ego = self.embedding_user(users)
        # pos_emb_ego = self.embedding_item(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        # users_emb_ego = users_emb
        # pos_emb_ego = pos_emb
        # neg_emb_ego = neg_emb
        users_emb_ego = self.embedding_user.weight[users]
        pos_emb_ego = self.embedding_item.weight[pos_items]
        neg_emb_ego = self.embedding_item.weight[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def create_bpr_loss_with_pop_global(self, users, pos_items,
                                        neg_items,pos_pops,neg_pops):  # this global does not refer to global popularity, just a name
        (users_emb, pos_emb, neg_emb,users_emb_ego, pos_emb_ego, neg_emb_ego) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())

        pos_scores = torch.sum(users_emb*pos_emb,dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(users_emb*neg_emb,dim=1)
        # item stop
        pos_scores = self.felu(pos_scores) + 1
        neg_scores = self.felu(neg_scores) + 1
        #pos_scores = torch.sigmoid(torch.relu(pos_scores))
        #neg_scores = torch.sigmoid(torch.relu(neg_scores))
        # pos_scores = torch.sigmoid(pos_scores/2)
        # neg_scores = torch.sigmoid(neg_scores/2)
        #pos_scores_with_pop = torch.sigmoid(pos_scores_with_pop)
        #neg_scores_with_pop = torch.sigmoid(neg_scores_with_pop)
        pos_scores_with_pop = pos_scores*pos_pops
        neg_scores_with_pop = neg_scores*neg_pops

        maxi = torch.log(self.f(pos_scores_with_pop - neg_scores_with_pop) + 1e-10)
        #self.condition_ratings = (self.felu(self.batch_ratings) + 1) * pos_pops.squeeze()
        #self.mf_loss_ori = -(torch.mean(maxi))
        #mf_loss = torch.neg(torch.mean(maxi))
        mf_loss = torch.sum(torch.neg(maxi))
        # fsoft=nn.Softplus()
        # mf_loss = -torch.mean(fsoft(pos_scores_with_pop - neg_scores_with_pop))
        # regular

        #mf_loss = torch.mean( torch.nn.functional.softplus(neg_scores_with_pop - pos_scores_with_pop))
        # reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
        #                       pos_emb_ego.norm(2).pow(2) +
        #                       neg_emb_ego.norm(2).pow(2)) / float(len(users))
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2))
        return mf_loss, reg_loss



    def bpr_loss_pop(self, users, pos_items,neg_items,pos_pops,neg_pops,weights=None):
        return  self.create_bpr_loss_with_pop_global(users, pos_items,neg_items,pos_pops,neg_pops)

    def forward(self, users, items,pop):
        """
        without sigmoid
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        rating = torch.sum(users_emb * items_emb, dim=-1)
        rating = self.felu(rating) + 1
        #rating = torch.sigmoid(torch.relu(rating))
        #rating = torch.sigmoid(rating/2)
        rating=rating*pop
        #rating = torch.sigmoid(rating)
        return rating




class BPRMF(MetaModule):
    '''
    PD/PDA
    PDG/PDG-A
    '''

    def __init__( self,config:dict,
                 dataset:BasicDataset,
                 fix:bool = False,):
        super(BPRMF, self).__init__()
        self.config=config
        self.dataset=dataset
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.fix=fix
        self.init_weights()


    def init_weights(self):
        self.embedding_user = MetaEmbed(self.num_users, self.latent_dim)
        self.embedding_item = MetaEmbed(self.num_items, self.latent_dim)
        # self.embedding_user = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        nn.init.kaiming_normal_(self.embedding_user.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.embedding_item.weight, mode='fan_out', a=0)
        self.f = nn.Sigmoid()
        self.felu=nn.ReLU()



    
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        return users_emb, items_emb

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items = torch.Tensor(range(self.dataset.m_items)).long().cuda()
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
        # users_emb_ego = self.embedding_user(users)
        # pos_emb_ego = self.embedding_item(pos_items)
        # neg_emb_ego = self.embedding_item(neg_items)
        users_emb_ego = users_emb
        pos_emb_ego = pos_emb
        neg_emb_ego =neg_emb
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
        #mf_loss = torch.neg(torch.mean(maxi))
        mf_loss = torch.sum(torch.neg(maxi))
        # fsoft=nn.Softplus()
        # mf_loss = -torch.mean(fsoft(pos_scores_with_pop - neg_scores_with_pop))
        # regular
        # reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
        #                       pos_emb_ego.norm(2).pow(2) +
        #                       neg_emb_ego.norm(2).pow(2)) / float(len(users))
        reg_loss = (1 / 2) * (users_emb_ego.norm(2).pow(2) +
                              pos_emb_ego.norm(2).pow(2) +
                              neg_emb_ego.norm(2).pow(2))
        return mf_loss, reg_loss


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

class BPRMFExpert(BPRMF):
        def __init__(self,
                     config: dict,
                     dataset: BasicDataset,
                     teacher_model: BPRMF,
                     ):
            super(BPRMFExpert, self).__init__(config,dataset)
            self.config = config
            self.dataset = dataset
            self.tea = teacher_model
            self.tea.fix = True
            self.de_weight = config['de_weight']

            # Expert Configuration
            self.num_experts = self.config["num_expert"]
            self.latent_dim_tea = self.tea.latent_dim
            expert_dims = [self.latent_dim, (self.latent_dim_tea + self.latent_dim) // 2, self.latent_dim_tea]

            ## for self-distillation
            if self.tea.latent_dim == self.latent_dim:
                expert_dims = [self.latent_dim, self.latent_dim_tea // 2, self.latent_dim_tea]

            self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
            self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

            self.user_selection_net = nn.Sequential(nn.Linear(self.latent_dim_tea, self.num_experts), nn.Softmax(dim=1))
            self.item_selection_net = nn.Sequential(nn.Linear(self.latent_dim_tea, self.num_experts), nn.Softmax(dim=1))

            # num_experts_params = count_parameters(self.user_experts) + count_parameters(self.item_experts)
            # num_gates_params = count_parameters(self.user_selection_net) + count_parameters(self.item_selection_net)

            self.sm = nn.Softmax(dim=1)

            self.T = 10

        def get_DE_loss(self, batch_entity, is_user=True):

            if is_user:
                s = self.embedding_user.weight[batch_entity]
                t = self.tea.embedding_user.weight[batch_entity]

                experts = self.user_experts
                selection_net = self.user_selection_net

            else:
                s = self.embedding_item.weight[batch_entity]
                t = self.tea.embedding_item.weight[batch_entity]

                experts = self.item_experts
                selection_net = self.item_selection_net

            selection_dist = selection_net(t)  # batch_size x num_experts

            if self.num_experts == 1:
                selection_result = 1.
            else:
                # Expert Selection
                g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).cuda()
                eps = 1e-10  # for numerical stability
                selection_dist = selection_dist + eps
                selection_dist = self.sm((selection_dist.log() + g) / self.T)
                selection_dist = torch.unsqueeze(selection_dist, 1)  # batch_size x 1 x num_experts
                selection_result = selection_dist.repeat(1, self.latent_dim_tea,
                                                         1)  # batch_size x teacher_dims x num_experts
            expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)]  # s -> t
            expert_outputs = torch.cat(expert_outputs, -1)  # batch_size x teacher_dims x num_experts
            expert_outputs = expert_outputs * selection_result  # batch_size x teacher_dims x num_experts
            expert_outputs = expert_outputs.sum(-1)  # batch_size x teacher_dims
            DE_loss = torch.sum(((t - expert_outputs) ** 2).sum(-1))
            return DE_loss


class OneLinear(nn.Module):
    """
    linear model: r
    """

    def __init__(self, n):
        super().__init__()
        self.m_items=n
        self.data_bias = nn.Embedding(n, 1)
        self.init_embedding()

    def init_embedding(self):
        nn.init.kaiming_normal_(self.data_bias.weight, mode='fan_out', a=0)


    def forward(self, items):
        d_items = self.data_bias.weight[items]
        return torch.sigmoid(d_items.squeeze())

    def getUsersRating(self):

        items = torch.Tensor(range(self.m_items)).long().cuda()
        items_emb = self.data_bias.weight[items]
        #rating = torch.matmul(users_emb, items_emb.t())
        #rating = torch.sum(users_emb * items_emb, dim=1)
        return torch.sigmoid(items_emb.squeeze())

    def l2_norm(self,items):
        #items = torch.unique(items)
        l2_loss = (torch.sum(self.data_bias.weight[items]** 2)) / 2
        return l2_loss

    def aver_norm(self,items):
        #items = torch.unique(items)
        l2_loss = (torch.sum((self.data_bias.weight[items]-torch.mean(self.data_bias.weight))** 2)) / 2

        return l2_loss

class TwoLinear(nn.Module):
    """
    linear model: u + i + r / o
    """

    def __init__(self, n_user, n_item):
        super().__init__()
        self.m_items=n_item
        self.user_embed= torch.nn.Embedding(
                num_embeddings=n_user, embedding_dim=10)
        self.item_embed= torch.nn.Embedding(
                num_embeddings=n_item, embedding_dim=10)
        self.init_embedding()


    def init_embedding(self):
        nn.init.kaiming_normal_(self.user_embed.weight, mode='fan_out', a=0)
        nn.init.kaiming_normal_(self.item_embed.weight, mode='fan_out', a=0)


    def forward(self, users, items):
        u_embed = self.user_embed(users)
        i_embed = self.item_embed(items)
        preds_1 = torch.sum(u_embed*i_embed,dim=-1)
        #preds_1 = torch.sum(u_embed + i_embed, dim=-1)
        return torch.sigmoid(preds_1.squeeze())

    def getUsersRating(self, users):
        items = torch.Tensor(range(self.m_items)).long().cuda()
        # dim_item = items.shape[-1]
        # users = users.reshape((-1,1)).repeat((1, dim_item))
        u_embed= self.user_embed(users)
        i_embed = self.item_embed(items)
        rating = torch.matmul(u_embed, i_embed.t())

        #rating = torch.sum(u_embed + i_embed, dim=-1)
        return torch.sigmoid(rating)

    def l2_norm(self, users, items):
        # users = torch.unique(users)
        # items = torch.unique(items)

        l2_loss = (torch.mean(torch.sum(self.user_embed.weight[users] ** 2,dim=-1) +torch.sum(self.item_embed.weight[items] ** 2,dim=-1))) / 2
        return l2_loss

    def aver_norm(self,users,items):
        # users = torch.unique(users)
        # items = torch.unique(items)
        l2_loss = (torch.sum(torch.sum((self.user_embed.weight[users]-torch.mean(self.user_embed.weight,dim=0)) ** 2, dim=-1)+
                  torch.sum((self.item_embed.weight[items]-torch.mean(self.item_embed.weight,dim=0)) ** 2, dim=-1))) / 2
        return l2_loss


class ZeroLinear(nn.Module):
    """
    linear model: r
    """

    def __init__(self):
        super().__init__()

        self.data_bias = nn.Embedding(1, 1)
        #self.data_bias.weight.data *= 0.001


    def forward(self):
        d_bias = self.data_bias.weight
        return torch.sigmoid(d_bias.squeeze())

    def l2_norm(self):
        data_bias_weight = self.data_bias.weight
        l2_loss = (torch.sum(data_bias_weight ** 2)) / 2
        return l2_loss

    def getUsersRating(self):
        return torch.sigmoid(self.data_bias.weight)


class MyTwoLinear(nn.Module):
    """
    linear model: u + i + r / o
    """

    def __init__(self, n_user, m_item):
        super().__init__()

        self.user_bias_1 = torch.nn.Embedding(
            num_embeddings=n_user, embedding_dim=10)
        self.item_bias_1 = torch.nn.Embedding(
            num_embeddings=m_item, embedding_dim=10)
        self.L1=nn.Linear(21, 1)
        self.init_embedding(0)


    def init_embedding(self, init):
        # nn.init.kaiming_normal_(self.user_bias.weight, mode='fan_out', a=init)
        # nn.init.kaiming_normal_(self.item_bias.weight, mode='fan_out', a=init)
        nn.init.xavier_uniform_(self.user_bias_1.weight, gain=1)
        nn.init.xavier_uniform_(self.item_bias_1.weight, gain=1)


    def forward(self, users, items,pop):
        u_emb = self.user_bias_1(users)
        i_emb = self.item_bias_1(items)
        print(pop.shape)
        pop=pop.reshape((-1,1))
        emb=torch.cat((u_emb,i_emb,pop),dim=-1)
        # preds=torch.sum(u_bias*i_bias,dim=-1)
        preds=self.L1(emb)
        return torch.sigmoid(preds.squeeze())