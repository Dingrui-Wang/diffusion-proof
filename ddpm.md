
# DDPM 后验均值公式推导
这里对 **DDPM 后验均值公式** 进行更加详细、完整的数学推导，作为 B 站 UP 主 **Nik_Li** 视频 **《一个视频看懂扩散模型 DDPM 原理推导 | AI 绘画底层模型》** 的补充。

🎥 原视频链接: [一个视频看懂扩散模型 DDPM 原理推导 | AI 绘画底层模型](https://www.bilibili.com/video/BV1p24y1K7Pf/?share_source=copy_web&vd_source=8fc6a637b118130ca21b2780805ce690)

## 前置约定

首先统一扩散模型的基础符号和性质：

1. **前向扩散系数定义**

* 单步噪声系数：

```math
\beta_t \in (0,1), \quad \alpha_t = 1 - \beta_t
```

* 累计乘积：

```math
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i
```

满足递推关系：

```math
\bar{\alpha}_t = \alpha_t \cdot \bar{\alpha}_{t-1}
```

2. **高斯分布形式**

所有分布均为**各向同性高斯分布**，其对数概率（省略归一化常数）：

```math
\log \mathcal{N}(z; \mu, \sigma^2 I)
\propto -\frac{\|z - \mu\|^2}{2\sigma^2}
```

3. **前向扩散过程的三个分布**

* 单步加噪：

```math
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, \beta_t I)
```

* 直接加噪到第 $t$ 步：

```math
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t) I)
```

* 第 $t-1$ 步结果：

```math
q(x_{t-1}|x_0) = 
\mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1}) I)
```

---

## 推导过程

### 步骤 1：贝叶斯公式展开后验分布

目标后验分布为：

```math
q(x_{t-1}|x_t,x_0)
```

根据贝叶斯公式：

```math
q(x_{t-1}|x_t,x_0)
= \frac{q(x_t|x_{t-1}) q(x_{t-1}|x_0)}{q(x_t|x_0)}
```

分母与 $x_{t-1}$ 无关，属于归一化常数，在对数空间中可忽略，因此：

```math
\log q(x_{t-1}|x_t,x_0)
\propto
-\frac{1}{2}
\left[
\frac{\|x_t - \sqrt{\alpha_t}x_{t-1}\|^2}{\beta_t}
+
\frac{\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0\|^2}{1-\bar{\alpha}_{t-1}}
\right]
```

目标是将括号内部分整理为标准高斯形式：

```math
\frac{(x_{t-1}-\tilde{\mu}_t)^2}{\tilde{\sigma}_t^2} + C
```

从而得到均值 $\tilde{\mu}_t$ 和方差 $\tilde{\sigma}_t^2$。

---

### 步骤 2：展开平方并合并同类项

记括号内为：

```math
S
```

展开平方项：

```math
S =
\frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{\beta_t}
+
\frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1-\bar{\alpha}_{t-1}}
```

展开得：

```math
S =
\frac{x_t^2 - 2\sqrt{\alpha_t}x_t x_{t-1} + \alpha_t x_{t-1}^2}{\beta_t}
+
\frac{x_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}}x_0 x_{t-1} + \bar{\alpha}_{t-1}x_0^2}{1-\bar{\alpha}_{t-1}}
```

---

#### (1) 合并二次项：求后验方差 $\tilde{\sigma}_t^2$

$x_{t-1}^2$ 的系数为：

```math
A = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}}
```

通分并利用：

```math
\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}, \quad \beta_t = 1-\alpha_t
```

得到：

```math
A =
\frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}
```

因此：

```math
\tilde{\sigma}_t^2 = \frac{1}{A}
= \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
```

---

#### (2) 合并一次项：求后验均值 $\tilde{\mu}_t$

$x_{t-1}$ 的一次项系数记为 $-2B$，其中：

```math
B =
\frac{\sqrt{\alpha_t}x_t}{\beta_t}
+
\frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}}
```

对比标准高斯展开：

```math
\frac{(x_{t-1}-\tilde{\mu}_t)^2}{\tilde{\sigma}_t^2}
```

一次项系数为：

```math
-2\frac{\tilde{\mu}_t}{\tilde{\sigma}_t^2}
```

因此：

```math
\tilde{\mu}_t
= B \cdot \tilde{\sigma}_t^2
```

---

### 步骤 3：代入并化简

代入 $B$ 和 $\tilde{\sigma}_t^2$：

```math
\tilde{\mu}_t
=
\left(
\frac{\sqrt{\alpha_t}x_t}{\beta_t}
+
\frac{\sqrt{\bar{\alpha}_{t-1}}x_0}{1-\bar{\alpha}_{t-1}}
\right)
\cdot
\frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
```

分别化简两项：

```math
\tilde{\mu}_t
=
\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
```