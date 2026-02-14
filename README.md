# DDPM后验均值公式推导
## 前置约定
首先统一扩散模型的基础符号和性质：
1. 前向扩散系数定义：
   - 单步噪声系数 $\beta_t \in (0,1)$，定义 $\alpha_t = 1-\beta_t$
   - 累计乘积 $\boldsymbol{\bar{\alpha}_t = \prod_{i=1}^t \alpha_i}$，自然满足恒等式：$\bar{\alpha}_t = \alpha_t \cdot \bar{\alpha}_{t-1}$
2. 所有分布为**各向同性高斯分布**，对数概率省略归一化常数后形式为：
   $$\log \mathcal{N}(z; \mu, \sigma^2 I) \propto -\frac{\|z - \mu\|^2}{2\sigma^2}$$
3. 前向扩散过程的三个分布：
   - 单步加噪：$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, \beta_t I)$
   - 直接加噪到t步：$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t) I)$
   - t-1步加噪结果：$q(x_{t-1}|x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1}) I)$

---

## 推导过程
### 步骤1：贝叶斯公式拆解后验分布
我们要推导的后验分布$q(x_{t-1}|x_t,x_0)$满足贝叶斯定理：
$$q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1}) q(x_{t-1}|x_0)}{q(x_t|x_0)}$$
分母$q(x_t|x_0)$和$x_{t-1}$无关，属于归一化常数，对数形式下可以直接忽略，因此后验的对数核心项为：
$$
\log q(x_{t-1}|x_t,x_0) \propto -\frac{1}{2}\left[
\frac{\|x_t - \sqrt{\alpha_t}x_{t-1}\|^2}{\beta_t} + \frac{\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\|^2}{1-\bar{\alpha}_{t-1}}
\right]
$$
我们的目标是把括号内的部分整理为标准高斯形式$\frac{(x_{t-1}-\tilde{\mu}_t)^2}{\tilde{\sigma}_t^2} + C$（$C$为和$x_{t-1}$无关的常数），即可得到均值$\tilde{\mu}_t$。

---

### 步骤2：拆分幂次合并系数
把括号内的部分记为$S$，展开平方项：
$$
\begin{align*}
S &= \frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1-\bar{\alpha}_{t-1}} \\
&= \frac{x_t^2 - 2\sqrt{\alpha_t}x_t x_{t-1} + \alpha_t x_{t-1}^2}{\beta_t} + \frac{x_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0 x_{t-1} + \bar{\alpha}_{t-1}x_0^2}{1-\bar{\alpha}_{t-1}}
\end{align*}
$$
我们按$x_{t-1}$的二次项、一次项分别合并系数：

#### （1）合并二次项，求后验方差$\tilde{\sigma}_t^2$
$x_{t-1}^2$的系数记为$A$：
$$
A = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}
$$
利用$\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$、$\beta_t=1-\alpha_t$通分化简：
$$
\begin{align*}
A &= \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})} \\
&= \frac{\alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t}{\beta_t(1-\bar{\alpha}_{t-1})} \\
&= \frac{1 - \bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}
\end{align*}
$$
标准高斯平方展开后$x_{t-1}^2$的系数为$\frac{1}{\tilde{\sigma}_t^2}$，因此：
$$
\tilde{\sigma}_t^2 = \frac{1}{A} = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
$$

#### （2）合并一次项，求均值$\tilde{\mu}_t$
$x_{t-1}$的一次项整体为$-2B x_{t-1}$，其中$B$为：
$$
B = \frac{\sqrt{\alpha_t}x_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}}
$$
对比标准高斯展开的一次项系数$-2\cdot \frac{\tilde{\mu}_t}{\tilde{\sigma}_t^2}$，可得：
$$
\frac{\tilde{\mu}_t}{\tilde{\sigma}_t^2} = B \implies \tilde{\mu}_t = B \cdot \tilde{\sigma}_t^2
$$

---

### 步骤3：化简得到最终结果
代入$B$和$\tilde{\sigma}_t^2$，拆分两项分别化简：
$$
\begin{align*}
\tilde{\mu}_t &= \left( \frac{\sqrt{\alpha_t}x_t}{\beta_t} + \frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}} \right) \cdot \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
\end{align*}
$$
和题目中的表达式完全一致。