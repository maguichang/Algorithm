# 多项式回归demo

多项式回归本质上就是将一个特征的高次幂也作为特征来处理，
是对线性回归的改进，以解决非线性问题

如何解决模型泛化的问题？
- train_test_split
- cross validtion交叉验证
- L1与L2正则化，正则化的本质是在loss function中加入参数的平方和项（岭回归Ridge）或者绝对值和项（lasso）。
