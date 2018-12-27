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

## 降维技术 PCA（Principal Component Aanalysis,PCA）
通俗理解：就是找出一个最主要的特征，然后进行分析
- 使得数据集更容易使用
- 降低很多算法的计算开销
- 去除噪音
- 使得结果易懂
- 便于可视化

## 多项式回归PolynomialRegression

多项式回归本质上就是将一个特征的高次幂也作为特征来处理，
是对线性回归的改进

## 正则化的线性回归 —— 岭回归与Lasso回归

使用多项式回归，如果多项式最高次项比较大，模型就容易出现过拟合。正则化是一种常见的防止过拟合的方法，一般原理是在代价函数后面加上一个对参数的约束项，这个约束项被叫做正则化项（regularizer）。在线性回归模型中，通常有两种不同的正则化项：

- 加上所有参数（不包括θ0）的绝对值之和，即l1范数，此时叫做Lasso回归；
- 加上所有参数（不包括θ0）的平方和，即l2范数，此时叫做岭回归.

## 逻辑回归
- 逻辑回归算法是用回归的方式解决分类的问题，而且只可以解决二分类问题；
- 方案：可以通过改造，使得逻辑回归算法可以解决多分类问题；
改造方法：
- OvR（One vs Rest），一对剩余的意思，有时候也称它为  OvA（One vs All）；一般使用 OvR，更标准；
- OvO（One vs One），一对一的意思；
改造方法不是指针对逻辑回归算法，而是在机器学习领域有通用性，所有二分类的机器学习算法都可使用此方法进行改造，解决多分类问题


## 分类问题评价指标
对于有偏数据来说，准确率并不能很好的反映分类问题的算法正确性。
以下引入新的评价指标
基于混淆矩阵

   - 准确率(Accuracy)
   - 精准率(Precision) 在被识别为正类别的样本中，确实为正类别的比例是多少,你认为的正样本，有多少猜对了（猜的准确性如何）。
   - 召回率(Recall) 在所有正类别样本中，被正确识别为正类别的比例是多少,正样本有多少被找出来了（召回了多少）
   - F1-score
#### ROC曲线的绘制与AUC 值的求取

- ROC曲线：受试者工作特征曲线 （receiver operating characteristic curve），又称为感受性曲线（sensitivity curve）。ROC以假阳性率（False positive rate，1-特异度）为横轴，真阳性率（True positive rate，灵敏度）为纵轴。

- ROC曲线有个很好的特性：当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变
- AUC(Area Under Curve,曲线下面积)：即ROC下面的面积，其可以用于衡量这个分类器的优劣。面积等于0.5随机猜，AUC越大，分类器越好。
## 支持向量机SVM(support vector machine)
- 直观理解，SVM就是间隔最大化分类

svm文档链接 

https://blog.csdn.net/weixin_39605679/article/details/81170300

https://www.jianshu.com/p/341c5edd85f5

- 由于svm算法中应用到距离最大化的计算，故数据预处理需要进行标准化
- svm的推导原理设计，涉及最优化带有约束条件的最值函数问题，拉格朗日函数与对偶问题，KKT条件，序列最小最优化算法SMO

### 高斯核函数也称RBF核(Radial Basis Function Kernel)
- 核心思想，将每一个样本点映射到一个无穷维的特征空间，依靠升维使得原本线性不可分的数据线性可分
- RBF 中的gamma参数越大越容易过拟合，gamma越小越容易欠拟合

## 决策树
### 局限性
- 决策边界，横平竖直，与x，y轴平行。数据偏斜，则决策树效果并不好
- 极度依赖个别样本数据，较为敏感。非参数学习算法，需要依靠调参，实现较好的决策效果。一般不单独使用，详见集成学习与随机森林。

## 随机森林

随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法。
- 为什么要随机抽样训练集？
如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的，这样的话完全没有bagging的必要；
- 为什么要有放回地抽样？（bagging 与 pasting的区别，有放回无放回）
如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是"有偏的"，都是绝对"片面的"（当然这样说可能不对），也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决，这种表决应该是"求同"，因此使用完全不同的训练集来训练每棵树这样对最终分类结果是没有帮助的，这样无异于是"盲人摸象"。

- 袋外错误率（oob error）（out-of-bag error）
随机森林有一个重要的优点就是，没有必要对它进行交叉验证或者用一个独立的测试集来获得误差的一个无偏估计。它可以在内部进行评估，也就是说在生成的过程中就可以对误差建立一个无偏估计。

## 关联规则 Apriori
### 核心概念
- 频繁项集
- 支持度
- 置信度
- 提升度（大于1，规则生效）

*缺点：算法执行速度慢*

*改进方式： fp-growth快速发现频繁项集，基于此频繁项集，计算支持度、置信度、提升度。*

*实践中FP也有一个问题，由于集合和字典的无序性，导致每次生成的FP树都不一样，即获取的频繁项集不一样。*
*解决方案： 将无序的字符组在数据库中与整数id一一对应，应用id 替代集合中的字符。*
