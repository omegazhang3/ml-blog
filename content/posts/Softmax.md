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
| $T \to
