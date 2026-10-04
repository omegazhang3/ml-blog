---
title: "深入理解 Softmax 函数：原理、公式与实战"
date: 2026-10-04T12:21:03Z
tags: ["机器学习", "深度学习", "激活函数", "数学基础"]
categories: ["机器学习基础"]
math: true
---

> Softmax 是深度学习中最经典的函数之一，几乎所有多分类模型的输出层都少不了它。本文将从原理、数学公式、代码实现到实际应用，带你彻底搞懂 Softmax。

<!--more-->

## 一、什么是 Softmax？

**Softmax 函数**（归一化指数函数）是一种将任意实数向量转换为**概率分布**的函数。它是深度学习中多分类任务的"标配"输出函数。

### 直观理解

```
输入（logits）：  [2.0,  1.0,  0.1]        ← 任意实数
                     ↓ Softmax
输出（概率）：   [0.659, 0.242, 0.099]     ← 总和 = 1.0
```

它的作用可以用一句话概括：

> **把一组"分数"转换成一组"概率"**，分数越高的类别获得越大的概率。

### 名字的含义

- **Soft（软）**：相对于 `argmax`（硬性地选最大值），Softmax 给出"软"的概率分布，保留了所有类别的信息
- **max**：最大的输入值仍会获得最大的概率（放大效应）

---

## 二、数学原理

### 2.1 核心公式

对于一个 $n$ 维实数向量 $\mathbf{x} = [x_1, x_2, ..., x_n]$，Softmax 函数定义为：

$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} $$

**公式分解**：

| 部分 | 作用 |
|------|------|
| 分子 $e^{x_i}$ | 对每个元素取指数，保证输出为正，且放大数值差异 |
| 分母 $\sum_j e^{x_j}$ | 所有指数值之和，作为归一化因子 |

### 2.2 为什么用指数函数？

1. **保证非负**：$e^x > 0$ 恒成立，概率必须非负
2. **放大差异**：指数函数会让大的更大、小的更小，使"赢家"更突出
3. **可微分**：处处可导，便于反向传播
4. **信息论意义**：对应最大熵原理下的解

### 2.3 手动计算示例

以输入 `[2.0, 1.0, 0.1]` 为例：

**Step 1：计算每个 $e^{x_i}$**

$$ e^{2.0} \approx 7.389, \quad e^{1.0} \approx 2.718, \quad e^{0.1} \approx 1.105 $$

**Step 2：计算总和**

$$ \text{sum} = 7.389 + 2.718 + 1.105 = 11.212 $$

**Step 3：归一化**

$$ p_1 = \frac{7.389}{11.212} \approx 0.659 $$

$$ p_2 = \frac{2.718}{11.212} \approx 0.242 $$

$$ p_3 = \frac{1.105}{11.212} \approx 0.099 $$

**验证**：$0.659 + 0.242 + 0.099 = 1.000$ ✅

---

## 三、Softmax 的三大特性

### 1️⃣ 输出是概率分布

- 每个输出值 $p_i \in (0, 1)$
- 所有输出值之和 $\sum_i p_i = 1$

### 2️⃣ 保持单调性

输入大的，输出也大，排序不变：

$$ x_i > x_j \iff p_i > p_j $$

### 3️⃣ 指数放大效应

数值差距会被指数函数显著放大：

```
输入：[1, 2, 3]       →  Softmax：[0.09, 0.24, 0.67]
输入：[1, 2, 10]      →  Softmax：[0.0001, 0.0003, 0.9996]
```

最大值会"吃掉"大部分概率。

---

## 四、代码实现

### 4.1 NumPy 基础实现

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
print(probs)         # [0.659, 0.242, 0.099]
print(probs.sum())   # 1.0
```

### 4.2 数值稳定版本（推荐）⭐

直接实现在面对大数值时会**溢出**：

```python
# ❌ 直接计算会溢出
x = np.array([1000, 1001, 1002])
np.exp(x)  # inf，数值溢出！
```

**解决方案**：减去最大值

```python
def softmax_stable(x):
    x_shifted = x - np.max(x)   # 防止 exp 溢出
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)
```

**数学证明**（减最大值不改变结果）：

$$ \frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{e^{-c} \cdot \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}} $$

### 4.3 PyTorch 实现

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])

# 方法 A：函数式
probs = F.softmax(logits, dim=0)

# 方法 B：模块化
softmax = torch.nn.Softmax(dim=0)
probs = softmax(logits)

print(probs)  # tensor([0.6590, 0.2424, 0.0986])
```

### 4.4 批量处理

```python
# batch_size=2, num_classes=3
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 0.5]
])

# dim=1：对每一行做 softmax（最常见用法）
probs = F.softmax(logits, dim=1)
print(probs)
# tensor([[0.6590, 0.2424, 0.0986],
#         [0.1131, 0.8360, 0.0508]])

print(probs.sum(dim=1))  # tensor([1.0, 1.0])
```

> 💡 **dim 参数记忆法**：对哪个维度做 softmax，哪个维度的和就是 1。

---

## 五、梯度与反向传播

### 5.1 Softmax 的梯度

设 $p_i = \text{Softmax}(x_i)$，其雅可比矩阵为：

$$ \frac{\partial p_i}{\partial x_j} = \begin{cases} p_i(1 - p_i), & i = j \\ -p_i \cdot p_j, & i \neq j \end{cases} $$

### 5.2 与交叉熵的完美配合 🎯

当 Softmax 与交叉熵损失（CrossEntropy）组合使用时，梯度变得异常简洁：

$$ \frac{\partial L}{\partial x_i} = p_i - y_i $$

- $p_i$：Softmax 的预测概率
- $y_i$：真实标签（one-hot 编码）

**例子**：

```
预测：[0.659, 0.242, 0.099]
真实：[1.0,   0.0,   0.0]
梯度：[-0.341, 0.242, 0.099]
```

这就是为什么分类任务中 **Softmax + CrossEntropy** 是黄金搭档——梯度形式简洁且不会梯度消失。

---

## 六、Softmax vs Sigmoid 🔥

这是面试和实际应用中的高频考点，必须搞清楚！

### 6.1 对比表格

| 特性 | Softmax | Sigmoid |
|------|---------|---------|
| **输出维度** | 多维向量（n 维） | 单个标量 |
| **输出范围** | 每个值 $\in (0, 1)$ | $\in (0, 1)$ |
| **概率总和** | $\sum p_i = 1$ | 独立，不受约束 |
| **公式** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\frac{1}{1+e^{-x}}$ |
| **适用场景** | 多分类（**互斥**） | 二分类 / 多标签（**独立**） |
| **类别关系** | 竞争关系 | 独立关系 |
| **常搭配损失** | CrossEntropyLoss | BCEWithLogitsLoss |

### 6.2 场景对比

#### 场景 A：多分类（互斥）→ 用 Softmax

> 一张图片识别是**猫、狗、鸟**中的哪一种，三者只能选一个。

```python
logits = torch.tensor([2.0, 1.0, 0.1])  # [猫, 狗, 鸟]
probs = F.softmax(logits, dim=0)
# [0.659, 0.242, 0.099]  → 和为 1，是"猫"的概率最高
```

#### 场景 B：多标签（独立）→ 用 Sigmoid

> 一张图片可能同时包含**猫**和**室内**两个独立标签。

```python
logits = torch.tensor([2.0, 1.0, 0.1])  # [有猫, 室内, 夜景]
probs = torch.sigmoid(logits)
# [0.881, 0.731, 0.525]  → 各自独立，和不为 1
```

### 6.3 有趣的数学关系

当 Softmax 用于**二分类**时，它等价于 Sigmoid：

$$ \text{Softmax}([x_1, x_2])_1 = \frac{e^{x_1}}{e^{x_1}+e^{x_2}} = \frac{1}{1+e^{-(x_1-x_2)}} = \text{Sigmoid}(x_1-x_2) $$

所以 **Sigmoid 可以看作 Softmax 的二维特例**。

### 6.4 选择决策树

```
分类任务？
├── 二分类 → Sigmoid（输出1个值）或 Softmax（输出2个值）
├── 多分类，类别互斥 → Softmax ✅
└── 多标签，类别独立 → Sigmoid（对每个类独立输出）✅
```

---

## 七、Softmax 的进阶变体

### 7.1 带温度的 Softmax（Temperature Scaling）

通过引入"温度"参数 $T$ 调整分布的平滑程度：

$$ \text{Softmax}(x_i / T) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}} $$

```python
def softmax_with_temperature(x, T=1.0):
    return np.exp(x/T) / np.sum(np.exp(x/T))

x = np.array([2.0, 1.0, 0.1])

print(softmax_with_temperature(x, T=0.5))  # [0.866, 0.117, 0.017]  更尖锐
print(softmax_with_temperature(x, T=1.0))  # [0.659, 0.242, 0.099]  原始
print(softmax_with_temperature(x, T=5.0))  # [0.418, 0.342, 0.240]  更平滑
```

**温度的影响**：

| 温度 | 效果 | 应用场景 |
|------|------|---------|
| $T < 1$ | 分布更尖锐，更自信 | 模型蒸馏（学生模型） |
| $T = 1$ | 标准 Softmax | 常规训练 |
| $T > 1$ | 分布更平滑 | 探索性采样 |
| $T \to 0$ | 趋近于 argmax（one-hot） | 确定性输出 |
| $T \to \infty$ | 趋近于均匀分布 | 完全随机 |

> 💡 **温度的直觉**：温度越高，模型越"犹豫"（概率越平均）；温度越低，模型越"自信"（概率越集中）。这与物理中的热力学分布同源。

### 7.2 LogSoftmax

在实际工程中，常用 `LogSoftmax` 代替 `Softmax`，即对 Softmax 的结果取对数：

$$ \text{LogSoftmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right) = x_i - \log\sum_j e^{x_j} $$

```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
log_probs = F.log_softmax(logits, dim=0)
print(log_probs)  # tensor([-0.4170, -1.4170, -2.3170])
```

**为什么用 LogSoftmax？**

1. **数值稳定**：避免概率值过小导致的下溢
2. **计算高效**：交叉熵损失中需要 $\log p$，直接用 LogSoftmax 省去一次对数运算
3. **配合 `NLLLoss`**：`LogSoftmax + NLLLoss` 等价于 `CrossEntropyLoss`

```python
# 以下两种写法完全等价
loss_a = F.cross_entropy(logits, target)

log_probs = F.log_softmax(logits, dim=0)
loss_b = F.nll_loss(log_probs.unsqueeze(0), target)
```

---

## 八、实际应用场景

### 8.1 图像分类

多分类模型（ResNet、ViT 等）的输出层几乎都用 Softmax：

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(512, 10),   # 10 个类别的 logits
    # 注意：训练时通常不显式加 Softmax，
    # 因为 CrossEntropyLoss 内部已包含
)

# 推理时才显式转为概率
logits = model(features)
probs = F.softmax(logits, dim=1)
pred = probs.argmax(dim=1)   # 预测类别
```

> ⚠️ **常见错误**：训练时在模型里加了 Softmax，又用 `CrossEntropyLoss`，会导致"双重 Softmax"，训练效果变差！

### 8.2 注意力机制（Attention）

Transformer 的核心——注意力权重就是用 Softmax 归一化的：

$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V $$

```python
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
weights = F.softmax(scores, dim=-1)   # 每一行的注意力权重和为 1
output = weights @ V
```

### 8.3 强化学习中的策略

在策略梯度方法中，Softmax 把动作价值转为动作选择的概率分布：

```python
action_logits = policy_net(state)
action_probs = F.softmax(action_logits, dim=-1)
action = torch.multinomial(action_probs, num_samples=1)  # 按概率采样
```

### 8.4 知识蒸馏

用带温度的 Softmax 让学生模型学习教师模型的"软标签"：

```python
T = 3.0
teacher_probs = F.softmax(teacher_logits / T, dim=1)   # 软化的教师输出
student_log = F.log_softmax(student_logits / T, dim=1)
distill_loss = F.kl_div(student_log, teacher_probs) * (T * T)
```

---

## 九、常见陷阱与注意事项 ⚠️

### 9.1 数值溢出

```python
# ❌ 错误：大数值直接 exp 会溢出为 inf
x = np.array([1000, 1001, 1002])
np.exp(x) / np.sum(np.exp(x))   # nan！

# ✅ 正确：减去最大值
softmax_stable(x)   # 正常输出
```

### 9.2 双重 Softmax

```python
# ❌ 错误：模型已输出概率，又传给 CrossEntropyLoss
probs = F.softmax(model(x), dim=1)
loss = F.cross_entropy(probs, target)   # 错！cross_entropy 期望 logits

# ✅ 正确：直接传 logits
logits = model(x)
loss = F.cross_entropy(logits, target)
```

### 9.3 dim 参数搞错

```python
# batch_size=2, num_classes=3
logits = torch.randn(2, 3)

F.softmax(logits, dim=0)   # ❌ 对 batch 维归一化（错误！）
F.softmax(logits, dim=1)   # ✅ 对类别维归一化（正确）
```

> 💡 **口诀**：分类任务中，`dim` 要指向**类别所在的维度**。

---

## 十、常见问题 FAQ

**Q1：Softmax 和 argmax 有什么区别？**

- `argmax` 直接返回最大值的**索引**（硬决策，不可导）
- `Softmax` 返回一个**概率分布**（软决策，可导），训练中必须用 Softmax 才能反向传播

**Q2：为什么训练时推荐用 `CrossEntropyLoss` 而不手动写 Softmax？**

因为 `CrossEntropyLoss` 内部融合了 `LogSoftmax + NLLLoss`，数值更稳定、计算更高效，还能避免双重 Softmax 的坑。

**Q3：Softmax 一定要配交叉熵吗？**

绝大多数多分类场景是的，因为两者组合后梯度形式最简洁（$p_i - y_i$），训练稳定。但在某些场景（如蒸馏、强化学习）也会配 KL 散度等其他损失。

**Q4：输入全是负数，Softmax 还能用吗？**

可以。$e^x$ 对任意实数都为正，负数输入只是让对应概率变小，不影响归一化。

---

## 十一、总结

| 要点 | 内容 |
|------|------|
| **本质** | 把实数向量转为概率分布 |
| **公式** | $\text{Softmax}(x_i) = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$ |
| **三大特性** | 输出为概率、保持单调、指数放大 |
| **数值稳定** | 计算前减去最大值 |
| **黄金搭档** | Softmax + CrossEntropy（梯度为 $p_i - y_i$） |
| **vs Sigmoid** | 多分类互斥用 Softmax，多标签独立用 Sigmoid |
| **工程实践** | 用 `CrossEntropyLoss`，别手动加 Softmax |

Softmax 看似简单，却贯穿了分类、注意力、强化学习、知识蒸馏等几乎所有深度学习的核心场景。理解它的原理、数值技巧和使用陷阱，是打好深度学习基础的关键一步。

> 📌 **下一篇预告**：交叉熵损失（Cross-Entropy Loss）—— 揭开 Softmax 最佳拍档的神秘面纱。
