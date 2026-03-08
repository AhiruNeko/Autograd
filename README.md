# 🧠 Autograd: 自制深度学习框架与应用

这是一个基于 **NumPy** 从零实现的轻量级深度学习框架，涵盖了自动求导引擎、循环神经网络（RNN）及手写数字识别。

## 🚀 核心功能

* **Autograd 引擎**：支持 `matmul`, `add`, `relu`, `exp` 等算子的自动微分（Autograd）。
* **模块化设计**：继承 `Module` 类，自动管理 `Linear` 层参数与梯度。
* **序列建模**：字符级 RNN 文本生成，包含梯度裁剪（Gradient Clipping）技术。
* **视觉分类**：基于 MLP 的 $8 \times 8$ 手写数字识别，支持置信度分析。

---

## 🏗️ 核心架构

### 1. 自动求导与模型 (`Module.py`)

```python
# 典型的模型构建
class DigitModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = Linear(input_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.l2(self.l1(x).relu())

```

### 2. 预测逻辑（置信度获取）

```python
# 从原始 Logits 提取预测概率
probs = np.exp(logits.data - np.max(logits.data)) / np.sum(np.exp(logits.data - np.max(logits.data)))
max_prob = round(np.argmax(probs), 2) # 预测数字

```

---

## 🔢 实战：数字识别 (MNIST 8x8)

项目实现了从 $32 \times 32$ 到 $8 \times 8$ 的数据降维，显著提升了在手写框架上的训练效率。

| 特性 | 描述 |
| --- | --- |
| **数据集** | Scikit-learn Digits (1797 样本) |
| **输入维度** | 64 (8x8 像素展平) |
| **优化器** | SGD |
| **损失函数** | CrossEntropyLoss (交叉熵) |

---

## 🛠️ 关键技术实现

* **梯度裁剪**：`np.clip(p.grad, -5, 5, out=p.grad)` 解决 RNN 梯度爆炸。
* **数值稳定性**：在 Softmax 计算中减去 `max(logits)`，防止 `exp` 溢出。
* **数据反转**：`(255 - img) / 255` 预处理，确保背景为 0、字迹为 1，加速收敛。

---

## 🔢 实战：序列生成 (Char-RNN)

项目实现了基于字符级的循环神经网络，能够学习文本序列的统计规律并进行自主创作（如模仿特定风格写诗或生成代码）。

| 特性 | 描述 |
| --- | --- |
| **任务类型** | 字符级语言建模 (Many-to-Many) |
| **核心单元** | RNN (包含 $W_{xh}, W_{hh}, W_{hy}$) |
| **隐藏状态** | 循环传递的 $h$，负责捕捉序列的长短期记忆 |
| **采样策略** | 基于 Softmax 概率分布的随机采样 (Random Choice) |

---

## 🛠️ 关键技术实现

* **随时间反向传播 (BPTT)**：梯度不仅在层间传递，还沿时间轴向后追溯，使模型具备时序关联能力。
* **梯度裁剪 (Gradient Clipping)**：通过 `np.clip(p.grad, -5, 5, out=p.grad)` 强行截断过大的梯度，有效避免 RNN 训练中常见的**梯度爆炸**问题。
* **独热编码 (One-hot Encoding)**：将字符转换为稀疏向量输入，确保模型能区分词表（Vocab）中的每一个独立单位。
* **状态重置策略**：在处理长文本或不同 batch 切换时，适时清空隐藏状态 $h$，平衡记忆长度与计算负担。
