# Algorithm
## 常用算法note
## KNN k近邻算法
- 解决分类问题
- 天然可以解决多分类问题
- 思想简单，效果强大

缺点：

1.最大缺点：效率低下

如果训练集有m个样本，n个特征，则预测每一个新的数据，需要O(m*n)
解决方案：使用树结构，KD-tree

2.数据高度相关

3.预测结果不具有可解释性

4.维数灾难

随着维数的增加，看似相近的两个点之间的距离越来越大

维度 | 点 | 距离
---|---|---
1维 | 0到1的距离| 1
2维 | （0，0）到（1，1）的距离|1.414
3维 | （0，0，0）到（1，1，1）的距离| 1.73
64维  | （0，0，...0）到（1，1,...1）的距离|8
10000维| （0，0，...0）到（1，1,...1）的距离|100
