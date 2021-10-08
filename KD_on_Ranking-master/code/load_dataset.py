import os
from math import ceil

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
import world
from dataloader import  BasicDataset
from world import cprint


class LoaderUnif(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self, config=world.config, path='./data/gowalla', threshold=4, unif_ratio=0.05, seed=0):
        # train or test
        super().__init__()
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.__n_users = 0
        self.__m_items = 0
        trainUniqueUsers, trainItem, trainUser = [], [], []
        self.__trainsize = 0
        self.validDataSize = 0
        self.testDataSize = 0
        self.valid2DataSize = 0

        user_df = pd.read_csv(path + '/user.txt', sep=',', header=None, names=['uid', 'iid', 'rating'])
        random_df = pd.read_csv(path + '/random.txt', sep=',', header=None, names=['uid', 'iid', 'rating'])
        user_df = user_df.drop(user_df[user_df['rating'] < threshold].index)

        # binirize the rating to -1/1
        user_df['rating'].loc[user_df['rating'] < threshold] = -1
        user_df['rating'].loc[user_df['rating'] >= threshold] = 1

        random_df['rating'].loc[random_df['rating'] < threshold] = -1
        random_df['rating'].loc[random_df['rating'] >= threshold] = 1

        m, n = max(user_df['uid']) + 1, max(user_df['iid']) + 1
        ratio = (unif_ratio, 0.05, 1 - unif_ratio - 0.05)
        unif_train_data, validation_data, test_data = seed_randomly_split(df=random_df, ratio=ratio,
                                                           split_seed=seed, shape=(m, n))
        train_data= sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')

        self.__m_items = n
        self.__n_users = m
        self.trainUniqueUsers = np.array(set(user_df['uid']))
        self.trainUser = np.array(user_df['uid'])
        self.trainItem = np.array( user_df['iid'])
        self.__trainsize=len(train_data.nonzero()[0])
        self.validDataSize=len(validation_data.nonzero()[0])
        self.testDataSize = len(test_data.nonzero()[0])
        self.valid2DataSize = len(unif_train_data.nonzero()[0])

        self.Graph = None
        print(f"({self.n_users} X {self.m_items})")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.validDataSize} interactions for vailding")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.valid2DataSize} interactions for vailding2")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.validDataSize + self.testDataSize+self.valid2DataSize ) / self.n_users / self.m_items}"
        )

        # (users,items), bipartite graph
        self.UserItemNet = train_data

        self.testUser=test_data.nonzero()[0]
        self.testItem = test_data.nonzero()[1]
        self.validUser = validation_data.nonzero()[0]
        self.validItem = validation_data.nonzero()[1]
        self.valid2User = unif_train_data.nonzero()[0]
        self.valid2Item = unif_train_data.nonzero()[1]
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.__n_users)))
        self.__testDict = self.build_dict(self.testUser, self.testItem)
        self.__validDict = self.build_dict(self.validUser, self.validItem)
        self.__valid2Dict = self.build_dict(self.valid2User, self.valid2Item)
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.__n_users

    @property
    def m_items(self):
        return self.__m_items

    @property
    def trainDataSize(self):
        return self.__trainsize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self.__allPos

    def itemCount(self):
        counts = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        return counts


    def add_expo_popularity(self,popularity):
        self.expo_popularity = popularity



    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))



    def build_dict(self, users, items):
        data = {}
        for i in range(len(users)):
            user = users[i]
            item = items[i]
            if data.get(user):
                data[user].append(item)
            else:
                data[user] = [item]
        return data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users,
                                         items]).astype('uint8').reshape(
                                             (-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         k=(self.UserItemNet[user]==0)[0]
    #         negItems.append(k)
    #     return negItems

    def getOneUserPosItems(self, user):
       return self.UserItemNet[user].nonzero()[1]



    @property
    def validDict(self):
        return self.__validDict

    @property
    def valid2Dict(self):
        return self.__valid2Dict




def seed_randomly_split(df, ratio, split_seed, shape):
    """
    Split based on a deterministic seed randomly
    """
    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Randomly shuffle the data
    rows, cols, rating = df['uid'], df['iid'], df['rating']
    num_nonzeros = len(rows)
    permute_indices = np.random.permutation(num_nonzeros)
    rows, cols, rating = rows[permute_indices], cols[permute_indices], rating[permute_indices]

    # Convert to train/valid/test matrix
    idx = [int(ratio[0] * num_nonzeros), int(ratio[0] * num_nonzeros) + int(ratio[1] * num_nonzeros)]

    train = sp.csr_matrix((rating[:idx[0]], (rows[:idx[0]], cols[:idx[0]])),
                              shape=shape, dtype='float32')

    valid = sp.csr_matrix((rating[idx[0]:idx[1]], (rows[idx[0]:idx[1]], cols[idx[0]:idx[1]])),
                              shape=shape, dtype='float32')

    test = sp.csr_matrix((rating[idx[1]:], (rows[idx[1]:], cols[idx[1]:])),
                             shape=shape, dtype='float32')

    return train, valid, test




