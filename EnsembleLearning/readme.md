## 集成学习说明文档


**参数解析：**
- base_estimator：个体预测器，需要是一个scikit-learn的分类器或回归器。如果没有给出，那么默认使用决策树（不推荐，不如RandomForest）。
- n_estimators：个体预测器的数量。通常，预测器越多，整体模型的variance越低。max_samples：每个数据子集（用于训练个体预测器）的样本数量。可以是浮点数（0.0至1.0，表示取样本占所有样本的比例），也可以是整数（表示样本的实际数量）。注意：如果输入了1而不是1.0，那么每个数据子集仅包含1个样本，会导致严重失误。max_features：每个数据子集的特征数量。数值原理同上。
- bootstrap：在随机选取样本时是否进行重置（sample with replacement）。
- bootstrap_features：在随机选取特征时是否进行重置。
- oob_score：是否计算out-of-bag分数。每个个体预测器，都只在原始数据集的一部分上训练。所以，可以用它在剩下样本上的误差（out-of-bag error），来估计它的泛化误差（generalization error）。
- warm_start：如果是True，那么在下一次使用fit方法时，向原有的模型再增加n_estimators个新的个体预测器，不丢弃原有的个体预测器。
- n_jobs：并行计算的进程数量，如果是-1则使用全部CPU核。random_state：随机状态，用于得出可复现的结果。verbose：控制屏幕上进程记录的冗长程度。

- criterion：每棵树分裂节点时的标准，针对分类器的有gini（基尼指数）和entropy（信息熵），针对回归器的有mse（均方差，mean squared error）和mae（mean absolute error，中文未知，欢迎补充）
- max_depth：每棵树最大的深度，（和boosting相比）一般都比较深。增大此参数可以增加模型variance。
- min_samples_split：节点分裂时的最小样本数量。增大此参数可对抗过拟合。
- min_samples_leaf：每个叶上最小的样本数量。增大此参数可对抗过拟合。
- min_weight_fraction_leaf：每个叶上最小的样本权重比例（在fit时不使用sample_weight时，每个样本权重相同，这时权重比例的计算方法是：叶子上样本数量/总样本数）。增大此参数可对抗过拟合。
- max_features：和普通的bagging相比，增加了auto、sqrt和log2的选项。
- max_leaf_nodes：每棵树叶子数量的限制。
- min_impurity_split：节点继续分裂时的impurity阈值。如果impurity低于这个参数，那么该决策树在这个节点上将停止生长，这个节点将成为一个叶子。impurity的计算方式取决于上文的criterion参数。
