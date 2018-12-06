# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/5

# 手动实现Apriori算法


def load_data_set():
    """
    加载样本数据
    Returns:
        A data set: A list of transactions. 嵌套列表的数据格式
    """
    data_set = [['l1', 'l2', 'l5'], ['l2', 'l4'], ['l2', 'l3'],
            ['l1', 'l2', 'l4'], ['l1', 'l3'], ['l2', 'l3'],
            ['l1', 'l3'], ['l1', 'l2', 'l3', 'l5'], ['l1', 'l2', 'l3']]
    return data_set


def create_C1(data_set):
    """
    扫描数据集合，创建频繁1项候选项集
    去重之后的1项集，C1的数据类型为集合，自动去重
    Create frequent candidate 1-itemset C1 by scaning data set.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
    Returns:
        C1: A set which contains all frequent candidate 1-itemsets
    """
    C1 = set()
    for t in data_set:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    # print('C1:',C1)
    # 输出频繁1项候选项集 C1: {frozenset({'l4'}), frozenset({'l5'}), frozenset({'l2'}), frozenset({'l1'}), frozenset({'l3'})}
    return C1

def is_apriori(Ck_item, Lksub1):
    """
    Judge whether a frequent candidate k-itemset satisfy Apriori property.
    判断它是不是符合apriori的规则，即某个项集是频繁的，那么它的所有子集也是频繁的。
    例如，如果 {0, 1} 是频繁的，那么 {0}, {1} 也是频繁的。
    反过来，也就是说如果一个项集是非频繁项集，那么它的所有超集也是非频繁项集
    Args:
        Ck_item: a frequent candidate k-itemset in Ck which contains all frequent
                 candidate k-itemsets.
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
    Returns:
        True: satisfying Apriori property.
        False: Not satisfying Apriori property.
    """
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    """
    创建 k项候选集，C1,1项集，C2,2项集...
    Lksub1，包含所有的频繁候选（k-1）项集
    Create Ck, a set which contains all all frequent candidate k-itemsets
    by Lk-1's own connection operation.
    Args:
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-itemsets.
        k: the item number of a frequent itemset.
    Return:
        Ck: a set which contains all all frequent candidate k-itemsets.
    """
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)# 原始Lksub1 类型为字典
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            # print('l1:',l1)
            # print('l2:',l2)
            l1.sort()
            l2.sort()
            # print('l1:', l1)
            # print('l2:', l2)
            if l1[0:k-2] == l2[0:k-2]:# 加入要得到频繁3项集，3项集由2项集连接而成，2项集连接的条件是第一项相等。
                # 在此，还要判断它是不是符合apriori的规则，即某个项集是频繁的，那么它的所有子集也是频繁的。
                # 例如，如果 {0, 1} 是频繁的，那么 {0}, {1} 也是频繁的。
                # 反过来，也就是说如果一个项集是非频繁项集，那么它的所有超集也是非频繁项集
                Ck_item = list_Lksub1[i] | list_Lksub1[j]# 集合求并集'|'等同于'U',排序后，若前k-2个元素相同，则进行集合合并
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    """
    Lk 为满足支持度的频繁k项集，由频繁候选k项集中大于支持度的数据组成
    Generate Lk by executing a delete policy from Ck.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        Ck: A set which contains all all frequent candidate k-itemsets.
        min_support: The minimum support.
        support_data: A dictionary. The key is frequent itemset and the value is support.
    Returns:
        Lk: A set which contains all all frequent k-itemsets.
    """
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

    # print('item_count:',item_count)
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk

def generate_L(data_set, k, min_support):
    """
    生成所有的频繁项集
    Generate all frequent itemsets.
    Args:
        data_set: A list of transactions. Each transaction contains several items.
        k: Maximum number of items for all frequent itemsets.频繁项集的元素最多有多少个，定义频繁1项集，2项集...
        min_support: The minimum support.最小支持度
    Returns:
        L: The list of Lk.频繁项集 的元素组合
        support_data: A dictionary. The key is frequent itemset and the value is support.
        字典类型，键为频繁项集，值为支持度
    """
    support_data = {}
    C1 = create_C1(data_set)
    L1 = generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    # print('Lksub1',Lksub1)
    L = []
    L.append(Lksub1)# 初始最佳频繁1项集，下面重2项集开始追加
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        # print('L'+str(i),Li)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data

def generate_big_rules(L, support_data, min_conf):
    """
    从频繁项集中生成强关联规则，大于某个置信度的规则
    Generate big rules from frequent itemsets.
    Args:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and the value is support.
        min_conf: Minimal confidence.
    Returns:
        big_rule_list: A list which contains all big rules. Each big rule is represented
                       as a 3-tuple.
    """
    big_rule_list = []
    sub_set_list = []
    # print('sub_set_list',sub_set_list)
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                # print('sub_set',sub_set)
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list

if __name__ == "__main__":
    """
    Test
    """

    # data_set = load_data_set()

    import pymysql
    import pandas as pd

    # load样本数据
    conn = pymysql.Connect(

        host='10.0.7.37',
        port=3306,
        user='maguichang',
        passwd='mgc',
        db='interfaceoptimizer',
        charset='utf8'
    )
    cur = conn.cursor()
    print(conn)
    print(cur)

    sql = "select windfarm,turbine,faultdate,timestamputc,date(timestamputc) AS FD,trigkeynum from pmc \
    where trigkeynum regexp '^SC'"
    cur.execute(sql)
    result = cur.fetchall()
    result = pd.DataFrame(list(result))
    result.columns = ['windfarm', 'turbine', 'faultdate', 'timestamputc', 'FD', 'trigkeynum']

    ll = []
    # 分割片段,合并嵌套字典
    for i in range(result.shape[0]):
        ll.append({
            result['windfarm'][i]:
                {
                    result['turbine'][i]:
                        {str(result['FD'][i]): result['trigkeynum'][i]}
                }
        })
    # 合并相同键的字典，获取故障片段
    from collections import defaultdict

    dic = {}
    for _ in ll:
        for k, v in _.items():
            for k1, v1 in v.items():
                for k2, v2 in v1.items():
                    dic.setdefault(k + '~' + k1 + '~' + k2, []).append(v2)

    # print(dic)
    ll2 = [v for k, v in dic.items()]
    ll3 = [list(set(i)) for i in ll2]
    data_set = ll3

    # data_set = load_data_set()

    L, support_data = generate_L(data_set, k=10, min_support=0.3)
    # print('L',L)

    # print('support_data',support_data)
    # print('support_data_len',len(support_data.keys()))

    big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)

    # print('big_rules_list',big_rules_list)

    for Lk in L:
        print("="*50)
        print("frequent " + str(len(list(Lk)[0])) + "-itemsets\t\tsupport")
        print("="*50)
        for freq_set in Lk:
            print(freq_set, support_data[freq_set])
    print
    print("Big Rules")
    for item in big_rules_list:
        print(item[0], "=>", item[1], "conf: ", item[2])