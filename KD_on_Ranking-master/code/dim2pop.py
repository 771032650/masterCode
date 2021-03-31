import matplotlib.pyplot as plt


def array(x):
    return x


dims = [10, 50, 100, 150, 200,250,300,350]

gowa_lgn = [{
    'precision': array([0.02589591, 0.01612298, 0.01110389]),
    'recall': array([0.0406396, 0.06171299, 0.08417723]),
    'ndcg': array([0.04024791, 0.0459588, 0.05295329])
}, {
    'precision': array([0.05382142, 0.03495613, 0.0249722]),
    'recall': array([0.09271196, 0.14699524, 0.20653575]),
    'ndcg': array([0.09651524, 0.11381758, 0.1326939])
}, {
    'precision': array([0.06439815, 0.0413879, 0.02931677]),
    'recall': array([0.10884228, 0.1707099, 0.23840204]),
    'ndcg': array([0.1167588, 0.13564101, 0.15690912])
}, {
    'precision': array([0.06862147, 0.04443566, 0.03145958]),
    'recall': array([0.11523808, 0.1825543, 0.25453744]),
    'ndcg': array([0.1240917, 0.14437278, 0.16688372])
}, {
    'precision': array([0.06999129, 0.045423, 0.03221314]),
    'recall': array([0.11726635, 0.18574399, 0.26045721]),
    'ndcg': array([0.12635811, 0.14689864, 0.17017149])
}, {
    'precision': array([0.07410744, 0.04816933, 0.03416103]),
    'recall': array([0.12302505, 0.19660586, 0.27540294]),
    'ndcg': array([0.13263768, 0.1545908, 0.17912048])
}, {
    'precision': array([0.07537678, 0.04911916, 0.03462791]),
    'recall': array([0.12497033, 0.19994074, 0.2778565]),
    'ndcg': array([0.1345285, 0.15689779, 0.18119717])
}, {
    'precision': array([0.0755342, 0.04939782, 0.0349568]),
    'recall': array([0.12495784, 0.20048313, 0.28018942]),
    'ndcg': array([0.13469465, 0.15736259, 0.18219979])
}]

amaz_lgn = [{
    'precision': array([0.00309253, 0.00311001, 0.00276998]),
    'recall': array([0.00306793, 0.00783127, 0.01400411]),
    'ndcg': array([0.0037216, 0.0056176, 0.00785543])
}, {
    'precision': array([0.01115628, 0.00963319, 0.00851585]),
    'recall': array([0.0135039, 0.02850929, 0.04930234]),
    'ndcg': array([0.01419366, 0.02013497, 0.02778407])
}, {
    'precision': array([0.01436848, 0.01228046, 0.01064453]),
    'recall': array([0.01737184, 0.03661479, 0.0620517]),
    'ndcg': array([0.01817638, 0.02576731, 0.03510711])
}, {
    'precision': array([0.01599073, 0.0139065, 0.01194879]),
    'recall': array([0.01962208, 0.04174201, 0.07011201]),
    'ndcg': array([0.0204256, 0.02926686, 0.03966389])
}, {
    'precision': array([0.01701271, 0.01466254, 0.01245294]),
    'recall': array([0.02113765, 0.04432597, 0.07345692]),
    'ndcg': array([0.02180759, 0.03113499, 0.0418059])
}, {
    'precision': array([0.01953916, 0.01668066, 0.01406455]),
    'recall': array([0.02465987, 0.05098069, 0.08356442]),
    'ndcg': array([0.02525207, 0.0358769, 0.04781296])
}, {
    'precision': array([0.02086507, 0.01759398, 0.01468115]),
    'recall': array([0.02638765, 0.05429681, 0.08776453]),
    'ndcg': array([0.02683676, 0.03802356, 0.0502931])
}, {
    'precision': array([0.02129818, 0.01793363, 0.01501168]),
    'recall': array([0.02707271, 0.05546723, 0.08990857]),
    'ndcg': array([0.02740364, 0.03879779, 0.05143602])
}]

# yelp_lgn = []
amaz_mf = [{
    'precision': array([0.00318941, 0.00269134, 0.00239044]),
    'recall': array([0.00339184, 0.00710964, 0.0126624]),
    'ndcg': array([0.00469683, 0.00596571, 0.0079575])
}, {
    'precision': array([0.00959862, 0.00860513, 0.00763406]),
    'recall': array([0.01101034, 0.02429993, 0.04255774]),
    'ndcg': array([0.01180628, 0.01704522, 0.0237134])
}, {
    'precision': array([0.01171286, 0.01023726, 0.00893148]),
    'recall': array([0.01401258, 0.03009367, 0.05144881]),
    'ndcg': array([0.0146193, 0.02104392, 0.02886385])
}, {
    'precision': array([0.01305017, 0.01140816, 0.00991547]),
    'recall': array([0.01595434, 0.03414778, 0.05813693]),
    'ndcg': array([0.01640149, 0.02373864, 0.03251604])
}, {
    'precision': array([0.01378151, 0.01203655, 0.01033604]),
    'recall': array([0.0171846, 0.03647103, 0.06106943]),
    'ndcg': array([0.01773145, 0.02551998, 0.03452269])
}, {
    'precision': array([0.01694812, 0.01450677, 0.01237505]),
    'recall': array([0.0214776, 0.0448984, 0.07434629]),
    'ndcg': array([0.02186425, 0.03128712, 0.04208539])
}, {
    'precision': array([0.01870904, 0.01587144, 0.01332333]),
    'recall': array([0.02381021, 0.04920175, 0.08024186]),
    'ndcg': array([0.0242411, 0.03448097, 0.04583813])
}, {
    'precision': array([0.01968163, 0.01646487, 0.01385255]),
    'recall': array([0.02525495, 0.05144778, 0.08364991]),
    'ndcg': array([0.0256557, 0.03615927, 0.04798495])
}]

gowa_mf = [{
    'precision': array([0.01424744, 0.01075893, 0.00801661]),
    'recall': array([0.02097629, 0.03852846, 0.05575266]),
    'ndcg': array([0.02308869, 0.02844985, 0.03400099])
}, {
    'precision': array([0.04647666, 0.03055797, 0.02199143]),
    'recall': array([0.07414823, 0.12191316, 0.17504794]),
    'ndcg': array([0.07953955, 0.09324935, 0.10969389])
}, {
    'precision': array([0.04900864, 0.03360305, 0.02467948]),
    'recall': array([0.08049799, 0.1364823, 0.19944917]),
    'ndcg': array([0.08367846, 0.10115474, 0.12070129])
}, {
    'precision': array([0.05140331, 0.03519057, 0.02597495]),
    'recall': array([0.08608, 0.14538852, 0.21165158]),
    'ndcg': array([0.08839272, 0.10717944, 0.12792143])
}, {
    'precision': array([0.05031817, 0.03487173, 0.02560319]),
    'recall': array([0.08193279, 0.14103901, 0.20569138]),
    'ndcg': array([0.07259724, 0.09185756, 0.11215137])
}, {
    'precision': array([0.05815862, 0.03986201, 0.02915132]),
    'recall': array([0.09886457, 0.1675994, 0.2408185]),
    'ndcg': array([0.09995954, 0.12212355, 0.14517874])
}, {
    'precision': array([0.06010784, 0.0413812, 0.03025923]),
    'recall': array([0.10255976, 0.17363906, 0.24919608]),
    'ndcg': array([0.10343133, 0.126663, 0.15055643])
}, {
    'precision': array([0.06160493, 0.04224529, 0.03078036]),
    'recall': array([0.10428687, 0.17561653, 0.25256792]),
    'ndcg': array([0.10480972, 0.12799754, 0.15224579])
}]

gowa_mf_bias = [{
    'APT': [0.8311856677160638, 0.15589903320159607, 0.01291529908232329],
    'I_KL':
    2.375745400841662,
    'I_bin':
    176.32677165354332
}, {
    'APT': [0.7400961774622021, 0.24481155245942163, 0.014982026034340873],
    'I_KL':
    1.8819884503508268,
    'I_bin':
    45.06867924528302
}, {
    'APT': [0.7458721280728748, 0.2337801705852172, 0.020252807734387288],
    'I_KL':
    1.4932388066666107,
    'I_bin':
    302.1045531197302
}, {
    'APT': [0.7358287561122552, 0.24119024493715033, 0.02288052336615493],
    'I_KL':
    1.5141943563065705,
    'I_bin':
    279.04672897196264
}, {
    'APT': [0.7291122423917602, 0.24607726572443375, 0.024712807287828226],
    'I_KL':
    1.533827086443457,
    'I_bin':
    281.0164705882353
}]

amaz_mf_bias = [{
    'APT': [0.7944859715441149, 0.19540204775561998, 0.010077313223030033],
    'I_KL':
    2.097238213855153,
    'I_bin':
    248.1112289383763
}, {
    'APT': [0.7795656592519877, 0.20634367342287424, 0.014087817943500365],
    'I_KL':
    2.2724564111065644,
    'I_bin':
    75.71536442414872
}, {
    'APT': [0.8352834184981036, 0.16029861520049052, 0.004369051915734428],
    'I_KL':
    1.6874541107393488,
    'I_bin':
    722.3739279588336
}, {
    'APT': [0.8285617271051446, 0.16595036377104794, 0.005452291852668146],
    'I_KL':
    1.71128377695144,
    'I_bin':
    729.1274238227146
}, {
    'APT': [0.8245512223846693, 0.16930646049805834, 0.0060805805140285735],
    'I_KL':
    1.7349956512818632,
    'I_bin':
    668.6948237535726
}]


def plot_metrics(data, key='precision', topk=1, title='Gowa MF'):
    d = [bag[key][topk] for bag in data]
    x = dims
    plt.plot(x, d, marker="^")
    plt.xlabel("Dims")
    plt.ylabel(f"{key}")
    plt.title(title)
    plt.xticks(x)
    plt.show()


def plot_bias(data, key='APT', title="Gowa MF"):
    if key == "APT":
        d_SH = [bag[key][0] for bag in data]
        d_MT = [bag[key][1] for bag in data]
        d_LT = [bag[key][2] for bag in data]
        x = dims
        plt.plot(x, d_SH, label="Short Head", marker="^")
        plt.plot(x, d_MT, label="Medium Tail", marker="^")
        plt.plot(x, d_LT, label="Long Tail", marker="^")
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


if __name__ == "__main__":
    data = amaz_lgn
    bias = None
    title = 'Amaz LGN'

    plot_metrics(data, title=title)
    plot_bias(bias, title=title)
    plot_bias(bias, key='I_KL', title=title)
    plot_bias(bias, key='I_bin', title=title)
