import matplotlib.pyplot as plt
import plot_data as pla
import plot_mymodel as pla_m
def array(x):
    return x


#dims = ['c_teacher','u_tracher','student','cd','ud']
#dims = ['my-400','old-400','my-128','old-128']
#dims = ['de_rrd','ud']
#dims=['64','128','192','256','300','350','400']
dims=['0.0','0.02','0.04','0.06','0.08','0.10','0.12', '0.14','0.16','0.18','0.20','0.22','0.24','0.26','0.28']
#dims = ['5','10']
#dims = ['1','2','3','4']
#dims = ['cd','me']
# yelp_lgn = []

def plot_metrics(data, key='precision', topk=1, title='Gowa MF'):
    d = [bag[key][topk] for bag in data]
    x = dims
    plt.plot(x, d, marker="^")
    plt.xlabel("Dims")
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.xticks(x)
    plt.show()

def plot_metrics_double(data1,data2, data3,key='precision', topk=1, title='Gowa MF'):
    d1 = [bag[key][topk] for bag in data1]
    d2 = [bag[key][topk] for bag in data2]
    d3 = [bag[key][topk] for bag in data3]
    x = dims
    plt.plot(x, d1, marker="^")
    plt.plot(x, d3)
    plt.plot(x, d2, marker="_")

    plt.xlabel("pop")
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.xticks(x)
    plt.legend(["0",'0.05',"0.1"])
    plt.show()

def plot_bias(data, key='APT', title="Gowa lgn",b=0):
    if key == "APT":
        #d_SH = [bag[key][0] for bag in data]
        #d_MT = [bag[key][1] for bag in data]
        d_LT = [bag[key][b] for bag in data]
        x = dims
        #plt.plot(x, d_SH, label="Short Head", marker="^")
        #plt.plot(x, d_MT, label="Medium Tail", marker="^")
        plt.plot(x, d_LT, label=title, marker="^")
        plt.legend()
    else:
        d = [bag[key] for bag in data]
        x = dims
        plt.plot(x, d, marker="^")
    plt.xticks(x)
    plt.xlabel("Dims")
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.show()

def plot_new(data, key='APT', title="Gowa lgn",co='blue'):
    labels = dims
    width = 0.35

    if key == 'Short Head':
        d_1 = [round(bag['APT'][0], 5) for bag in data]

    elif key == 'Medium Tail':
        d_1 = [round(bag['APT'][1], 5) for bag in data]

    elif key == 'Short Tail':
        d_1 = [round(bag['APT'][2], 5) for bag in data]

    else:
        d_1 = [round(bag[key][0], 5) for bag in data]
    bar1=plt.bar(labels, d_1, width,color=co)
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.xlabel("model")
    # ax.text(.5,.93, '1-teacher   2-student   3-distillation   4-PCA', transform=ax.transAxes,
    #         ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    plt.xticks(labels)
    for rect in bar1:
        height = rect.get_height()  # 获得bar1的高度
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
    plt.show()

def plot_com(data, key='APT', title="Gowa lgn",co='red'):
    labels = dims
    width = 0.35
    fig, ax = plt.subplots()
    if key=='Short Head':
        d_1 = [round(bag['APT'][0],5) for bag in data[:3]]
        d_2 = [round(bag['APT'][0],5) for bag in data[3:]]
    elif key=='Medium Tail':
        d_1 = [round(bag['APT'][1],5) for bag in data[:3]]
        d_2 = [round(bag['APT'][1],5) for bag in data[3:]]
    elif key=='Short Tail':
        d_1 = [round(bag['APT'][2],5) for bag in data[:3]]
        d_2 = [round(bag['APT'][2],5) for bag in data[3:]]
    else:
        d_1 = [round(bag[key][0],5) for bag in data[:3]]
        d_2 = [round(bag[key][0],5) for bag in data[3:]]

    bar1 = plt.bar([i - 0.2 for i in range(3)], height=d_1, width=0.35, color='r', label='cd')  # 第一个图
    bar2 = plt.bar([i + 0.2 for i in range(3)], height=d_2, width=0.35, color='g', label='new')
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.xlabel("data")
    for rect in bar1:
        height = rect.get_height()  # 获得bar1的高度
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")

    # ax.text(.5,.93, '1-teacher   2-student   3-distillation   4-PCA', transform=ax.transAxes,
    #         ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    plt.xticks(range(3),['gowa','amaz','yelp'])
    plt.legend(loc="lower left")
    #loc="lower left"
    plt.show()


def plot_new_com(data, key='APT', title="Gowa lgn",co='red'):
    labels = dims
    width = 0.35
    fig, ax = plt.subplots()
    if key=='precision':
        d_1 = [round(bag,5) for bag in data[0]]
        d_2 = [round(bag,5) for bag in data[1]]
    else:
        d_1 = [round(bag,5) for bag in data[2]]
        d_2 = [round(bag,5) for bag in data[3]]

    bar1 = plt.bar([i - 0.2 for i in range(5)], height=d_1[:-1], width=0.35, color='r', label='old')  # 第一个图
    bar2 = plt.bar([i + 0.2 for i in range(5)], height=d_2[:-1], width=0.35, color='g', label='new')
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.xlabel("data")
    for rect in bar1:
        height = rect.get_height()  # 获得bar1的高度
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
    for rect in bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")

    # ax.text(.5,.93, '1-teacher   2-student   3-distillation   4-PCA', transform=ax.transAxes,
    #         ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    plt.xticks(range(5))
    plt.legend()
    #loc="lower left"
    plt.show()
if __name__ == "__main__":
    # data = comparsion_yelp_pre
    # bias = comparsion_yelp_bais
    title = 'lgn'
    # data1 =pla.comparsion_20210627_gowa_100_precision
    # data2 = pla.comparsion_20210627_yelp_100_precision
    # data3 = pla.comparsion_20210627_amaz_64_precision
    # bias1=pla.comparsion_20210627_gowa_100_APT
    # bias2 = pla.comparsion_20210627_yelp_100_APT
    # bias3 =pla.comparsion_20210627_amaz_64_APT
    # #plot_metrics(data, key='precision',title='gowa pca')
    # plot_new(data3, key='precision', title='amaz_precision', co=['green','red','black','orange','yellow','blue'])
    # plot_new(bias3, key='Short Head',title='amaz Short Head',co=['green','red','black','orange','yellow','blue'])
    # plot_new(bias3, key='Medium Tail', title='amaz Medium Tail', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_new(bias3, key='Short Tail', title='amaz Short Tail', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])

    # plot_com(data, key='precision', title='precision', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_com(bias, key='Short Head',title='Short Head',co=['green','red','black','orange','yellow','blue'])
    # plot_com(bias, key='Medium Tail', title='Medium Tail', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_com(bias, key='Short Tail', title='Short Tail', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_com(data, key='precision', title='precision', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_new_com(data1, key='account', title='gowa account ratio', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_new_com(data2, key='account', title='amaz account ratio', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    # plot_new_com(data3, key='account', title='yelp account ratio', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    #plot_bias(bias,key='APT', title=title)
    #plot_bias(bias, key='APT5', title=title)
    # plot_bias(bias, key='I_KL', title=title)
    # plot_bias(bias, key='I_bin', title=title)
    data1=pla.CD_comparsion_gowa_0_t_0923_bais
    data2 = pla.PD_comparsion_gowa_1_t_0923_bais
    data3 = pla.PD_comparsion_gowa_2_t_0923_bais
    datalist=[val for val in data1 for i in range(len(data2))]
    #plot_new(data1, key='precision', title='gowa vaild precision',co="red")
    # plot_new(data1, key='ndcg', title='yelp test ndcg' ,co="green")
    # plot_new(data2, key='Short Head', title='yelp test APT', co="red")
    # plot_metrics(data1, key='ndcg', topk=0,title='gowa ndcg')
    # plot_metrics(data2, key='APT', topk=0, title='gowa ndcg')

    #plot_new_com(data3, key='precision', title='yelp precision', co=['green', 'red', 'black', 'orange', 'yellow', 'blue'])
    #plot_new_com(data2, key='precision',title='yelp precision',co=['green','red','black','orange','yellow','blue'])
    #plot_new(data2, key='ndcg', title='yelp valid ndcg', co="blue")
    plot_metrics_double(data1=datalist,data2=data2, data3=data3,topk=0,key='APT', title='gowa_MF')
    #bais=pla.BPRMF_kwai_bais
    #plot_metrics(data=data1,topk=1,title='douban_MF')
    #plot_metrics(data=data2, topk=1, title='douban_300')
    # plot_bias(data=bais,title='kwai long head',b=0)
    # plot_bias(data=bais, title='kwai medium tail', b=1)
    # plot_bias(data=bais, title='kwai short tail', b=2)
