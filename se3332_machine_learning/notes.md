## Machine Learning 

#### 1. Overview

**监督学习：每个输入样本都有目标输出标签**

我们的目的是学习函数映射

* 分类问题
* 回归问题
* 语义分割
* 目标检测

**无监督学习：Training examples as input patterns, with no associated output**

目标是学习隐含或者潜在的内部结构

* 聚类
* 降维和特征选择
* 数据的概率密度估计

* 异常检测

**强化学习：通过环境交互，学习状态到动作的映射，最大化长期奖励**

* 蒙特卡洛强化学习: $v(s_t) \leftarrow v(s_t) + \alpha [R_t - v(s_t)]$

* AlphaGo

泛化性问题：普遍规律还是特定细节？

欠拟合过拟合：模型是否足够复杂，还是过于复杂导致记住随机噪声

避免过拟合可以考虑增加训练数据、正则化、数据划分交叉验证、早停、引入先验知识

### 2. 线性回归

机器学习的核心要素：

1. 数据 experience
2. 模型 hypothesis
3. 损失函数 objective
4. 优化算法 improve

**Regression is a function that describes the relationship between one dependent variable and a series of other variables.**

回归的核心思路是通过实验观测值来估计统计变量的真实值。

可以用于预测、归因分析、控制等领域。

#### 2.1 模型

$$
y = f(x) = w^Tx + w_0
$$

Linear model for regression is a (d+1)-dimensional hyperplane.

![image-20250411044713197](C:\Users\leo\github\ai-related-learning-notes\se3332_machine_learning\mdref\linear_regression_visialization.png)

训练过程：根据数据误差估算 $w$ 和 $w_0$

预测过程：对于输入的新数据 $x$ 计算 $f(x)$ 得到回归值 $y$

#### 2.2 损失函数

对于输入样本 $x$, 预测值为 $y$, 目标值为 $r$, 则相应的平方误差为： $l(w,w_0 | x,r ) = (r-y)^2$

对于完整数据集，损失函数定义为均方误差：$L(w,w_0) = \frac{\sum_{l=1}^{N}(r^l-y^l)^2}{2N}$

> 为什么是2N？ 在后续对损失函数进行梯度下降求解时，如果没有2会导致计算出来的梯度多了一个常数系数，加上常数因子避免系数干扰，简化后续运算，而N是为了规范化规模

#### 2.3 训练优化

一般用梯度下降法（Gradient Descend）优化损失函数。

优化目标： $min_wL(w)$

损失函数 $ L(w, w_0 | D) $ 定义为：
$$
L(w, w_0 | D) = -\frac{1}{2N} \sum_{\ell=1}^{N} (r^{(\ell)} - y^{(\ell)})^2
$$
其中，$$r^{(\ell)} $$ 是目标值，$$y^{(\ell)} $$ 是模型的预测值，$$ N $$ 是数据的总数。

迭代步骤：$w_{t+1} = w_t - \eta_t\frac{\partial L}{\partial w} $

对于每个权重 $$w_j ( j = 1, \dots, d) $$，我们需要计算损失函数关于 $$ w_j $$ 的偏导数：

$$
\frac{\partial L}{\partial w_j} = -\frac{1}{N} \sum_{\ell=1}^{N} (r^{(\ell)} - y^{(\ell)}) \frac{\partial y^{(\ell)}}{\partial w_j}
$$
偏导数变为：

$$
\frac{\partial L}{\partial w_j} = -\frac{1}{N} \sum_{\ell=1}^{N} (r^{(\ell)} - y^{(\ell)}) x_j^{(\ell)}
$$

$$
w_j^{\text{new}} = w_j^{\text{old}} + \eta \cdot \frac{1}{N} \sum_{\ell=1}^{N} (r^{(\ell)} - y^{(\ell)}) x_j^{(\ell)}
$$

其中，$$ \eta $$ 是学习率，控制更新步长。

以下是线性回归矩阵形式的规范推导，使用 Markdown 格式并用 `$$` 包围数学公式：

**矩阵表示**

令输入矩阵 $$ X $$、权重向量 $$w$$、目标值向量 $$ r $$ 分别表示为：

$$
X = 
\begin{bmatrix}
x^{(1)} \\
x^{(2)} \\
\vdots \\
x^{(N)}
\end{bmatrix}
=
\begin{bmatrix}
x_0^{(1)} & x_1^{(1)} & x_2^{(1)} & \cdots & x_d^{(1)} \\
x_0^{(2)} & x_1^{(2)} & x_2^{(2)} & \cdots & x_d^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_0^{(N)} & x_1^{(N)} & x_2^{(N)} & \cdots & x_d^{(N)}
\end{bmatrix},
\quad
w =
\begin{bmatrix}
w_1 \\
w_2 \\
\vdots \\
w_d
\end{bmatrix},
\quad
r =
\begin{bmatrix}
r^{(1)} \\
r^{(2)} \\
\vdots \\
r^{(N)}
\end{bmatrix}.
$$

$$
y = Xw =
\begin{bmatrix}
x^{(1)}w \\
x^{(2)}w \\
\vdots \\
x^{(N)}w
\end{bmatrix}.
$$

线性回归的损失函数为均方误差，矩阵形式表示为：

$$
L(w) = \frac{1}{2} (r - y)^T (r - y) = \frac{1}{2} (r - Xw)^T (r - Xw).
$$

$$
\frac{\partial L(w)}{\partial w} = -X^T (r - Xw).
$$

$$
\frac{\partial L(w)}{\partial w} = 0 \quad \Rightarrow \quad X^T (r - Xw) = 0.
$$

$$
X^T r = X^T Xw.
$$

$$
w^* = (X^T X)^{-1} X^T r.
$$

> $XX^T$ 可能不可逆，需要正则化引入一个惩罚项

#### 2.3 Coding

[colab-tutorial.ipynb - Colab](https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb)

[Linear Regression Tutorial](https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial/data?select=insurance.csv)