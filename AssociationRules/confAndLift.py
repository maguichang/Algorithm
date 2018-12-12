def fp_confidence_lift(fp_result,data_set_len):
    """
    基于fp的结果计算置信度与支持度
    :param fp_result:频繁项集结果
    :param data_set_len:原始数据嵌套list长度
    :return:频繁项集,支持度,提升度集A,被提升项集B,提升度,置信度,嵌套列表
    """
    fp_dic = {}
    confidenceAndLift = []
    for i in fp_result:
        fp_dic.setdefault(frozenset(i[0]), i[1])

    for k, v in fp_dic.items():
        if len(list(k)) > 1:
            for k1 in list(k):
                kk = k - frozenset([k1])
                confidenceAndLift.append((k,round(v/data_set_len,3),k1, kk,round(fp_dic[k]/ fp_dic[frozenset([k1])],3),round(fp_dic[k]*data_set_len / (fp_dic[kk] * fp_dic[frozenset([k1])]),3)))
                # print('置信度：', k1, '->', kk, fp_dic[k] / fp_dic[frozenset([k1])])
                # print('提升度：', k1, '->', kk, fp_dic[k] / (fp_dic[kk] * fp_dic[frozenset([k1])]))

    return confidenceAndLift
