import numpy as np
import argparse
import pandas as pd
import shutil
parser = argparse.ArgumentParser(description="Run pop_bias.")
parser.add_argument('--path', nargs='?', default="./data/yelp/",  # change by zyang
                    help='Input data path.')
parser.add_argument('--slot_count', type=int, default=15,  # change by zyang
                    help='Input data path.')
args = parser.parse_args()

root = args.path #'./data/ml_10m/'
slot_count = args.slot_count
item_list = []
user_list = []
idx = slot_count - 2   # the last previous dataset before testign 

path = './t_8.txt'
print("select data:",path)
with open(path) as f:
    for line in f:
        it = int(line.split()[0])
        ues = line.split()[1:]
        for u in ues:
            item_list.append(it)
            user_list.append(int(u))

data = pd.DataFrame(user_list)
data.columns = ['user']
data['item'] = pd.DataFrame(item_list,index=data.index)
print("tot number in last previous data:",data.shape[0])
data = data.sort_values(['user'],ignore_index=True)
data_np = data.values

with open("./train_fine_tune.txt",'w') as f:
    u_pre = data_np[0,0]
    k = 0
    for x in data_np:
        u_ = x[0]
        i_ = x[1]
        if u_ != u_pre or k == 0:
            u_pre = u_ 
            if k>0:
                f.write('\n')
            f.write(str(u_))
            k = 1
        f.write(" " + str(i_))

print('copy data to fine tune dir')
#shutil.copyfile(args.path+"test.txt",args.path+'fine_tune/'+'test.txt')
#shutil.copyfile(args.path+"valid.txt",args.path+'fine_tune/'+'valid.txt')
