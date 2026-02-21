# CME 295 Lecture 1

## 一、专业术语

### 1. 模型

- **Transformer 基模型**
  - BERT：Bidirectional Encoder Representations from Transformers
  - GPT：Generative Pre-trained Transformer
  - T5：Text-to-Text Transfer Transformer
  - LLaMA：Large Language Model Meta AI

### 2. 训练策略

- SFT：Supervised Fine-Tuning 监督微调
- PEFT：Parameter-Efficient Fine-Tuning 参数高效微调
- RL：Reinforcement Learning 强化学习
- RM：Reward Model 奖励模型
- RLHF：Reinforcement Learning from Human Feedback 基于人类反馈的强化学习
- PPO：Proximal Policy Optimization 近端策略优化
- DPO：Direct Preference Optimization 直接偏好优化
- FLAN：Fine-tuned Language Net 指令微调范式

### 3. 任务/技术

- NLP：Natural Language Processing 自然语言处理
- NER：Named Entity Recognition 命名实体识别
- PoS：Part-of-Speech 词性标注
- MLM：Masked Language Model 掩码语言模型
- NSP：Next Sentence Prediction 下一句预测
- MT：Machine Translation 机器翻译
- QA：Question Answering 问答
- NLG：Natural Language Generation 自然语言生成
- CoT：Chain-of-Thought 思维链
- ToT：Tree-of-Thought 思维树
- SC：Self-Consistency 自一致性
- RAG：Retrieval-Augmented Generation 检索增强生成

### 4. 经典结构/方法

- RNN：Recurrent Neural Network 循环神经网络
- LSTM：Long Short-Term Memory 长短期记忆网络
- GRU：Gated Recurrent Unit 门控循环单元
- GloVe：Global Vectors for Word Representation 全局词向量
- BPE：Byte Pair Encoding 字节对编码（子词分词）
- OOV：Out-of-Vocabulary 未登录词

### 5. 评估指标

- F1：F1-Score 精确率与召回率调和平均
- PPL：Perplexity 困惑度（越低越好）
- BLEU：Bilingual Evaluation Understudy 机器翻译评价指标
- ROUGE：Recall-Oriented Understudy for Gisting Evaluation 生成文本评价指标
- WER：Word Error Rate 词错误率

---

## 二、核心概念

1. **Tokenization（分词）**
   把文本拆成模型能处理的最小单位；主流：**子词分词（BPE/WordPiece）**。

2. **Word Embedding（词嵌入）**
   将词映射为低维稠密向量，可计算语义相似度；代表：**Word2vec**。

3. **Self-Attention（自注意力）**
   让每个 token 关注句子中所有其他 token，捕捉长距离依赖；
   公式：
   \[
   \text{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]

4. **Multi-Head Attention（多头注意力）**
   多组自注意力并行，从不同角度提取语义特征。

5. **Transformer**
   仅用注意力机制的架构，**全并行计算**，是所有大模型基础。
   - Encoder：编码上下文
   - Decoder：自回归生成
   - 位置编码（Positional Encoding）：补充序列位置信息

6. **Encoder-Decoder**
   编码器理解输入，解码器逐 token 生成输出。

7. **Mask（掩码）**
   Decoder 中屏蔽未来 token，保证生成只能看前面。

8. **Add & Norm**
   残差连接 + 层归一化，防止深度网络梯度消失。

---

# CME 295 Lecture 2 

## 一、专业术语

### 1. 位置编码/归一化相关
- **PE**：Positional Encoding 位置编码
- **RoPE**：Rotary Position Embeddings 旋转位置编码
- **LN**：Layer Normalization 层归一化
- **RMSNorm**：Root Mean Square Layer Normalization 均方根层归一化
- **Post-Norm**：Post-Layer Normalization 后归一化
- **Pre-Norm**：Pre-Layer Normalization 前归一化

### 2. 注意力近似方法
- **SWA**：Sliding Window Attention 滑动窗口注意力
- **MHA**：Multi-Head Attention 多头注意力（原始版）
- **MQA**：Multi-Query Attention 多查询注意力
- **GQA**：Group-Query Attention 分组查询注意力
- **CLS**：Classification Token 分类标记（BERT专用）
- **SEP**：Separator Token 分隔标记（BERT专用）
- **PAD**：Padding Token 补全标记（BERT专用）
- **MASK**：Mask Token 掩码标记（BERT专用）

### 3. BERT相关
- **MLM**：Masked Language Modeling 掩码语言模型
- **NSP**：Next Sentence Prediction 下一句预测
- **DistilBERT**：Distilled BERT 蒸馏版BERT
- **RoBERTa**：Robustly Optimized BERT Pretraining Approach 优化版BERT

### 4. 模型架构分类
- **Encoder-Decoder**：编解码器架构（T5/BART）
- **Encoder-only**：仅编码器架构（BERT/RoBERTa）
- **Decoder-only**：仅解码器架构（GPT/LLaMA）
- **Cross-Attention**：交叉注意力（Encoder-Decoder专属）

## 二、核心概念

### 1. 注意力机制（Transformer核心）

#### （1）缩放点积注意力
- 公式：$\text{Attention}(Q,K,V) = softmax\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$
- 核心：通过Q-K相似度算权重，加权V实现上下文关注；$\sqrt{d_k}$ 解决内积爆炸导致的梯度消失。
- 注意力得分矩阵：元素$(i,j)$ 代表第$i$个token对第$j$个token的关注程度。

#### （2）多头注意力（MHA）
- 步骤：Q/K/V拆分为$h$组→每组算缩放点积→拼接+线性映射
- 作用：多组注意力头捕捉不同维度的语义关联（局部/长距离/词法）。

#### （3）掩码多头注意力
- 核心：将注意力得分矩阵上三角置为-∞，防止模型看到未来token（Decoder专属）。

### 2. 位置编码（解决Transformer位置丢失）

#### （1）绝对位置编码
- 可学习PE：为每个位置初始化向量，随模型训练，**无法扩展到更长序列**；
- 硬编码PE（三角函数）：$PE_{m,2i}=\sin(\omega_i·m)$、$PE_{m,2i+1}=\cos(\omega_i·m)$，可扩展，能捕捉相对位置（内积仅与$m-n$有关）。

#### （2）相对位置编码（主流）
- **线性偏置**：注意力得分加仅与$m-n$相关的偏置（T5 Bias/ALiBi）；
- **RoPE（旋转位置编码）**：通过旋转矩阵将位置融入Q/K，内积仅依赖相对位置，无额外参数，支持长序列外推（LLaMA/GPT-4标配）。

### 3. 层归一化（LN）

#### （1）基础LN
- 计算：均值→方差→标准化→缩放平移（$\gamma/\beta$可学习）；
- 特点：按样本特征维度归一化，不受批次大小影响，解决内部协变量偏移。

#### （2）核心类型
- **Post-Norm**：LN在残差后（原始Transformer），深层训练不稳定；
- **Pre-Norm**：LN在残差前（主流），训练稳定，适配深层模型；
- **RMSNorm**：移除均值计算，仅保留方差归一化，计算量减半（GPT-3/LLaMA标配）。

### 4. 注意力近似方法（长序列适配）

#### （1）稀疏注意力
- 滑动窗口注意力（SWA）：每个token仅关注局部窗口内的token，复杂度$O(n d·w)$（$w$为窗口大小）；
- Longformer：普通token用SWA，[CLS]用全局注意力，兼顾局部/全局。

#### （2）注意力头共享
| 方案 | 核心设计 | 效率/效果 | 应用 |
|------|----------|-----------|------|
| MHA | $h$个Q对应$h$个K/V | 效果最好，效率最低 | 小序列模型 |
| GQA | $h$个Q分$G$组，每组共享1个K/V | 平衡效率&效果（主流） | GPT-3/LLaMA |
| MQA | $h$个Q共享1个K/V | 效率最高，效果略降 | 低延迟推理 |

### 5. Transformer三大架构
| 架构类型 | 核心组件 | 注意力 | 核心任务 | 代表模型 |
|----------|----------|--------|----------|----------|
| Encoder-Decoder | 编码器+解码器 | 编码器双向自注意力；解码器掩码+交叉注意力 | 翻译/摘要 | T5/BART |
| Encoder-only | 仅编码器 | 双向自注意力 | 分类/NER | BERT/RoBERTa |
| Decoder-only | 仅解码器 | 掩码自注意力 | 文本生成 | GPT/LLaMA |
- 交叉注意力：Decoder的Q匹配Encoder的K/V，关注源文本（翻译/摘要核心）。

### 6. BERT（Encoder-only代表）
#### （1）核心范式：预训练+微调
- 预训练：无监督学习通用语言知识（大规模无标注语料）；
- 微调：少量标注数据适配下游任务（冻结底层，微调顶层）。

#### （2）预训练核心任务
- **MLM（掩码语言模型）**：随机遮蔽15%token（80%[MASK]/10%随机/10%不变），预测遮蔽token，学习双向上下文；
- **NSP（下一句预测）**：预测句子B是否为A的真实下一句，学习句间关联（RoBERTa已移除）。

#### （3）BERT输入处理
- 分词：WordPiece（子词分词）；
- 特殊标记：[CLS]（分类）、[SEP]（分隔）、[MASK]（掩码）、[PAD]（补全）；
- 输入嵌入：词嵌入+位置编码+段嵌入（三嵌入相加，维度768）。

#### （4）BERT变体
- **DistilBERT**：知识蒸馏，12层→6层，参数量减40%，保留97%性能，推理快1.6倍；
- **RoBERTa**：移除NSP+动态掩码+扩大语料，性能比原始BERT提升4%+（架构不变，仅优化训练）。

### 7. 关键公式

- **RMS计算**：$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}$（RMSNorm核心）；
- **知识蒸馏损失**：$\mathcal{L} = \alpha \mathcal{L}_{CE} + (1-\alpha) \mathcal{L}_{KL}$（硬损失+软损失）；
- **KL散度**：$KL\left(\hat{y}_T \| \hat{y}_S\right) = \sum_{i} \hat{y}_T^{(i)} \log\left(\frac{\hat{y}_T^{(i)}}{\hat{y}_S^{(i)}}\right)$（学生模仿教师输出）。

---

# CME 295 Lecture 3

## 一、专业术语

### 1. 模型架构/生成相关

- **LLM**：Large Language Model 大语言模型
- **MoE**：Mixture of Experts 混合专家模型
- **ICL**：In-Context Learning 上下文学习
- **CoT**：Chain of Thought 思维链
- **MTP**：Multi-Token Prediction 多Token预测

### 2. 解码/推理优化相关

- **Top-p**：Nucleus Sampling 核采样（Top-p采样）
- **KV Caching**：Key-Value Caching 键值缓存
- **MQA**：Multi-Query Attention 多查询注意力
- **GQA**：Group-Query Attention 分组查询注意力
- **PagedAttention**：Paged Attention 分页注意力
- **Latent Attention**：Latent Attention 潜在注意力
- **Speculative Decoding**：Speculative Decoding 推测解码

### 3. 基础术语
- **BOS**：Beginning of Sequence 序列起始符
- **EOS**：End of Sequence 序列终止符
- **T**：Temperature 温度系数

## 二、核心概念

### 1. 大语言模型（LLM）核心
- **定义**：为token序列分配概率的模型，核心是基于上下文预测下一个token的条件概率 $P(w_{t+1}=w|C)$；
- **特征**：数十亿级参数、千亿级token训练数据、依赖GPU集群；
- **主流架构**：Decoder-only的Transformer（掩码多头注意力+Add&Norm+FFN堆叠）。

### 2. 混合专家模型（MoE）
- **核心思想**：将超大模型拆分为多个小型子模型（专家），仅激活部分专家计算，降低计算量/显存占用；
- **组成**：门控网络（分配专家权重）+ 专家网络（独立计算单元）+ 输出层（加权求和）；
- **核心类型**：
  - 稠密MoE：激活所有专家，加权平均输出（计算量大）；
  - 稀疏MoE：Top-k选择专家，仅计算选中专家（工业界主流）；
- **核心问题**：路由坍缩（少数专家被高频选中），需通过**辅助损失**均衡专家使用频率；
- **集成方式**：替换Transformer解码器的FFN层，对每个token单独路由。

### 3. LLM解码策略（下一个Token预测）
| 策略 | 核心逻辑 | 特点 | 适用场景 |
|------|----------|------|----------|
| 贪心解码 | 选概率最高的token | 简单快，局部最优，缺乏多样性 | 快速生成、低多样性需求场景 |
| 束搜索 | 保留k个概率最高的生成路径，选全局最优 | 流畅度高，计算量随k线性增加 | 机器翻译、摘要（精准优先） |
| 基础采样 | 从概率分布随机采样 | 多样性高，易生成无意义token | 文本创作、对话（创意优先） |
| Top-k采样 | 仅在概率最高的k个token中采样 | 约束采样范围，k固定灵活性低 | 平衡多样性与合理性 |
| Top-p采样 | 选累积概率≥p的最小token集合采样 | 自适应调整候选集，主流方案 | 绝大多数生成类任务 |
- **温度系数（T）**：调节概率分布平滑度：
  - $T→0^+$：退化为贪心解码（完全确定）；
  - $0<T<1$：分布更陡峭，生成保守/重复；
  - $T>1$：分布更平缓，生成随机/有创造性；
  - $T=1$：原始概率分布。
- **引导解码**：仅允许选择符合格式要求的有效token，强制结构化输出（JSON/表格/代码）。

### 4. 提示工程核心
- **上下文长度**：LLM能处理的最大输入token数，存在**上下文衰减**（远距离上下文理解能力下降）；
- **高效提示结构**：上下文（背景）+ 指令（任务）+ 输入（具体信息）+ 约束（格式/内容限制）；
- **上下文学习（ICL）**：
  - 零样本：无示例，直接提问（简单任务）；
  - 少样本：加1~5个示例，性能远优于零样本（复杂任务）；
- **思维链（CoT）**：提示模型分步推理，提升复杂任务（逻辑/数学）性能，增强可解释性；
- **自一致性**：生成多条推理路径，投票/聚合得答案，提升CoT精度（成本大幅增加）。

### 5. LLM推理优化（两大维度）
#### （1）精确效率优化（无性能损失）
- **KV缓存**：缓存历史token的K/V向量，新token仅算Q，注意力复杂度从$O(n^2d)$降至$O(nd)$（基础优化）；
- **分页注意力（PagedAttention）**：将KV缓存拆为固定大小显存块，动态分配，降低显存碎片，提升利用率3~5倍；
- **推测解码**：小模型快速生成k个候选token，大模型一次性验证，解码速度提升2~3倍（无显著性能损失）。

#### （2）近似优化（轻微性能损失）
- **注意力头共享**：
  - MQA：所有Q头共享1个K/V头（效率最高，性能略降）；
  - GQA：Q头分组，每组共享1个K/V头（效率+性能平衡，GPT-3.5/4主流）；
- **潜在注意力**：压缩K/V向量为低维潜在表示后缓存，显存占用降低50%以上；
- **多Token预测（MTP）**：模型设k个预测头，一次生成k个token，提升解码速度（k过大会降质量）。

### 6. 关键公式
- **MoE稀疏输出**：$\hat{y}=\sum_{i \in \mathcal{I}_{k}} G(x)_{i} E_{i}(x)$（$\mathcal{I}_k$为Top-k专家索引）；
- **MoE路由坍缩辅助损失**：$loss=\alpha \cdot N \cdot \sum_{i=1}^{N} f_{i} \cdot P_{i}$（均衡专家使用频率）；
- **温度系数调整概率**：$P_{adj }\left(w_{t+1}=w_{i} | C\right)=\frac{exp \left(\frac{x_{i}}{T}\right)}{\sum_{j=1}^{n} exp \left(\frac{x_{j}}{T}\right)}$；
- **概率比（温度系数影响）**：$R_{ij} = exp\left(\frac{x_i - x_j}{T}\right)$（T越小，高logits token的概率占比越高）。
   
---

# CME 295 Lecture 4 

## 一、专业术语

### 1. 训练/微调相关

- **SFT**：Supervised FineTuning 有监督微调
- **LoRA**：Low-Rank Adaptation 低秩适配
- **QLoRA**：Quantized LoRA 量化低秩适配
- **FP16/FP32/FP64**：Floating Point 16/32/64 16/32/64位浮点精度
- **BFLOAT16**：Brain Floating Point 16 16位脑浮点精度
- **NF4**：NormalFloat 4 4位正态浮点（QLoRA专用量化格式）

### 2. 并行/内存优化相关

- **ZeRO**：Zero Redundancy Optimization 零冗余优化
- **TP**：Tensor Parallelism 张量并行
- **PP**：Pipeline Parallelism 流水线并行
- **SP/CP**：Sequence/Context Parallelism 序列/上下文并行
- **EP**：Expert Parallelism 专家并行
- **HBM**：High Bandwidth Memory 高带宽显存（GPU低速大显存）
- **SRAM**：Static Random-Access Memory 静态随机存取存储器（GPU高速小显存）

### 3. 评估/性能相关

- **FLOPs**：Floating-point Operations 浮点运算总次数（衡量计算量）
- **FLOPS/FLOP/s**：Floating-point Operations per Second 每秒浮点运算次数（衡量硬件性能）
- **MMLU**：Massive Multitask Language Understanding 多任务语言理解（通用知识评估基准）
- **ARC-Challenge**：AI2 Reasoning Challenge 人工智能推理挑战（基础推理评估基准）
- **GSM8K**：Grade School Math 8K 小学数学应用题（数学推理评估基准）
- **HumanEval**：Human Evaluation 代码生成评估基准

## 二、核心概念

### 1. LLM训练范式
| 范式 | 核心逻辑 | 核心问题/优势 |
|------|----------|---------------|
| 传统机器学习 | 具体任务→从头训练专属模型 | 资源浪费、泛化差、无迁移性 |
| 迁移学习 | 预训练通用模型→微调适配新任务 | 部分复用资源，跨任务能力有限 |
| LLM核心范式 | 预训练（学通用规律）+ 微调（适配端任务） | 通用语言能力，单模型适配多任务，资源复用最大化 |

### 2. 预训练核心
- **核心目标**：学习自然语言/代码的底层模式（语法、语义、逻辑、常识），具备通用理解与生成能力；
- **目标函数**：下一个令牌预测，最小化预测令牌与真实令牌的交叉熵损失（自回归训练）；
- **数据规模**：万亿级令牌（GPT-3：3000亿，LLaMA 3：15万亿）；
- **核心规律**：
  - 缩放规律：测试损失随计算量、数据集大小、模型参数数增加而单调下降（幂律关系）；
  - 样本效率：模型参数越多，达到相同损失所需训练令牌越少；
  - Chinchilla定律：模型参数数N与训练令牌数T需按比例缩放（670亿参数最优训练令牌数：1.5万亿）。

### 3. LLM训练核心流程与公式
#### （1）训练流程
初始化（随机初始化参数$\theta_0$）→ 前向传播（计算损失$L$）→ 反向传播（计算梯度$\nabla \mathscr{L}(\theta_t)$）→ 参数更新（Adam优化器）

#### （2）Adam优化器核心公式
$$\theta_{t+1} \leftarrow \theta_{t}-\alpha \frac{m_{t}}{\sqrt{v_{t}}+\epsilon}$$
- $m_t$（一阶矩估计）：$m_{t+1} \leftarrow \beta_1 m_t + (1-\beta_1)\nabla \mathscr{L}(\theta_t)$（动量，平滑更新）；
- $v_t$（二阶矩估计）：$v_{t+1} \leftarrow \beta_2 v_t + (1-\beta_2)(\nabla \mathscr{L}(\theta_t))^2$（自适应学习率）；
- 超参数：$\alpha$（学习率）、$\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$（防分母为0）。

### 4. LLM训练核心瓶颈：显存优化
| 优化手段 | 核心思路 | 核心优势 | 性能损失 |
|----------|----------|----------|----------|
| 数据并行 | 拆分训练数据，多GPU复制完整模型 | 简单易实现 | 显存冗余（利用率极低） |
| ZeRO优化 | 拆分参数/梯度/优化器状态（ZeRO-1/2/3） | 消除冗余，提升显存利用率 | 无 |
| 模型并行 | 拆分模型参数/层到多GPU（TP/PP/SP/EP） | 适配万亿参数模型 | 无 |
| Flash Attention | 分块加载Q/K/V到SRAM，重计算中间结果 | 显存节省~50%，速度提升5.7倍 | 无（精确注意力） |
| 混合精度训练 | 低精度（FP16/BF16）计算，高精度（FP32）保存参数 | 速度提升+显存降低 | 无 |

#### Flash Attention关键
- 核心：用SRAM高速读写替代HBM低速读写，分块计算softmax（跟踪局部最大值$m_t$、局部归一化和$s_t$）；
- 效果：计算量增加13%，HBM读写减少90%，运行时间降低85%（从41.7ms→7.3ms）。

### 5. 有监督微调（SFT）
- **定义**：基于人工标注的输入-输出配对数据，以“下一词预测”为目标微调预训练模型；
- **核心场景**：指令调优（让模型成为遵循自然语言指令的助手）；
- **数据规模**：GPT-3（1.3万样本）、LLaMA 3（1000万样本）；
- **核心效果**：从“陈述事实”→“响应指令”（贴合人类需求）；
- **核心挑战**：标注成本高、对提示词分布敏感、泛化能力有限、评估标准缺失；
- **评估方式**：
  - 量化：MMLU（通用知识）、ARC-Challenge（基础推理）、GSM8K（数学推理）、HumanEval（代码）；
  - 主观：Chatbot Arena（A/B盲测，量化“使用体验”）。

### 6. 参数高效微调（LoRA/QLoRA）
#### （1）LoRA核心
- 公式：$W = W_0 + B·A$
  - $W_0$：预训练权重（冻结不更新）；
  - $A$（$d×r$）：低秩投影矩阵（随机初始化，需微调）；
  - $B$（$r×k$）：低秩还原矩阵（初始0，需微调）；
  - $r$：低秩维度（4/8/16，远小于$d/k$）；
- 优势：参数量极少、任务切换灵活、训练快、性能无损；
- 经验规律：学习率需比全量微调高10~100倍，适合小批次训练；
- 最优应用位置：前馈网络（FFN）。

#### （2）QLoRA核心
- 基础：LoRA + 权重量化；
- 关键：
  - $W_0$：量化为4位NF4存储（适配LLM权重正态分布，精度损失小）；
  - $B·A$：FP16/FP32存储并微调；
  - 双重量化：对量化常数（缩放因子/偏移量）再量化为FP8，额外省6%显存；
- 优势：显存节省16倍（LLaMA 65B）、消费级GPU可微调千亿参数模型、性能无损。

### 7. LLM全生命周期
模型初始化 → 预训练（学通用规律） → SFT/指令调优（适配任务） → 偏好调优（贴合人类偏好/安全）
- 核心目标：Model Alignment（模型对齐）——让模型能力、行为、价值观与人类需求一致。

### 8. LLM训练核心挑战
- 成本：经济（数百万美元）、时间（数周/数月）、环境（巨量电力）；
- 知识：知识截止（无实时信息）、知识编辑困难（分布式存储）、抄袭风险（版权问题）。

---

# CME 295 Lecture 5

## 一、专业术语

### 1. 偏好调优相关
- **RLHF**：Reinforcement Learning from Human Feedback 从人类反馈的强化学习
- **DPO**：Direct Preference Optimization 直接偏好优化
- **BoN**：Best of N 最优N选1（RL简易替代方案）
- **RM**：Reward Model 奖励模型（RLHF第一阶段产物）
- **PPO**：Proximal Policy Optimization 近端策略优化（RLHF核心RL算法）
- **GAE**：Generalized Advantage Estimation 广义优势估计（PPO优势值计算方法）

### 2. 算法/指标相关
- **KL散度**：Kullback-Leibler Divergence 库尔贝克-莱布勒散度（衡量分布差异）
- **SFT**：Supervised FineTuning 有监督微调（偏好调优的基础模型）
- **β/λ/ε**：超参数（DPO/PPO的核心调优参数）
- **σ(·)**：Sigmoid函数（将分数差映射为0~1概率）

## 二、核心概念

### 1. LLM完整训练四阶段
| 阶段 | 核心目标 | 关键输出 |
|------|----------|----------|
| 初始化模型 | 定义架构+随机参数 | 无知识的空模型 |
| 预训练 | 学习语言/代码/知识的通用规律 | 基础通用模型 |
| 微调（SFT） | 适配具体任务/指令 | 任务适配模型 |
| 偏好调优 | 注入人类偏好，修正失范行为 | 对齐人类需求的实用模型 |

### 2. 偏好调优核心基础
- **产生原因**：SFT模型易生成“无意义、违背用户需求”的回答，需注入人类偏好信号修正；
- **核心优势**：
  1. 人类标注“比较（A比B好）”比重构“完美回答A”更简单，标注成本低；
  2. 对数据分布容错性强，可扩展性高；
  3. 反向检验SFT数据集质量；
- **偏好数据类型**（标注粒度从低到高）：
  | 类型 | 标注形式 | 核心特点 |
  |------|----------|----------|
  | 点态型（Pointwise） | 单回答标量评分（0~1/连续值） | 无相对关系，独立评分 |
  | 成对型（Pairwise） | 两回答优劣比较（A>B/A<B） | 标注成本低、易规模化，RLHF/DPO核心数据 |
  | 列表型（Listwise） | 多回答整体排序 | 粒度最高，标注成本最高 |
- **成对型数据获取流程**：
  1. 同一提示词x → SFT模型（T>0增加多样性）生成2个有优劣差异的回答；
  2. 通过人类评分/LLM-as-a-judge标注“优回答y_w”和“劣回答y_l”。

### 3. RLHF（经典偏好调优方法）
#### （1）RLHF核心定义
- 两阶段训练：**奖励建模（RM）+ 强化学习（RL）**；
- LLM的RL形式化对应：
  | RL要素 | LLM对应形式 |
  |--------|-------------|
  | 智能体 | LLM本身 |
  | 状态 | 截至当前的输入序列s_t |
  | 动作 | 下一个生成的token a_t |
  | 策略 | 生成token的概率π_θ(a_t|s_t) |
  | 奖励 | 人类偏好对应的奖励值r_t |
- 核心目标：学习参数θ，让策略π_θ生成高奖励的token序列（贴合人类偏好）。

#### （2）第一阶段：奖励建模（RM）
- **核心目标**：训练RM输出标量分数r(x,ŷ)，区分回答优劣；
- **理论基础**：Bradley-Terry公式
  $$p(y_i>y_j) = \frac{e^{r_i}}{e^{r_i}+e^{r_j}} = \sigma(r_i - r_j)$$
- **损失函数**（负对数似然，最大化优回答分数高于劣回答的概率）：
  $$\mathscr{L}(\theta) = -\mathbb{E}\left[log\left(\sigma\left(r(x,y_w)-r(x,y_l)\right)\right)\right]$$
- **训练细节**：
  1. 数据量：~1万条人类标注的成对数据；
  2. 模型架构：预训练LLM替换预测头为分类头（编码器用[CLS]向量，解码器用最后token向量投影）；
  3. 输入：(x,ŷ)，输出：标量奖励分数r。

#### （3）第二阶段：强化学习（RL）
- **核心目标**：以SFT模型为初始化，用RM的分数作为奖励信号，通过RL优化LLM策略；
- **核心约束**：最大化奖励 + 最小化与基础模型的偏离（防止奖励破解/训练崩塌）；
- **核心算法：PPO**
  - 基础目标函数（最大化奖励 - λ×KL惩罚）：
    $$\mathscr{L}(\theta) = -\left[r(x,\hat{y}) - \lambda KL\left(\pi_{\theta}(\hat{y}|x)||\pi_{ref}(\hat{y}|x)\right)\right]$$
  - 核心变体：
    | 变体 | 核心思想 | 关键优化 |
    |------|----------|----------|
    | PPO-Clip | 裁剪新/旧策略的概率比，限制更新幅度 | 目标函数：$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[min(r_t(\theta)\hat{A}_t, clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\right]$ |
    | PPO-KL Penalty | 直接惩罚新策略与基础模型的KL散度 | 目标函数：$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{ref}(a_t|s_t)}\hat{A}_t - \beta KL(\pi_{ref}||\pi_{\theta})\right]$ |
  - 优势值（Advantage）：$\hat{A}_t \approx r - V(s_t)$（V(s_t)为值函数预测的基线奖励，用GAE计算，平衡偏差和方差）；
- **训练细节**：数据量~10万条，RM分数替代人类标注，初始化用SFT模型参数。

#### （4）RLHF局限性
- 模型数量多（4个）、训练流程复杂；
- 超参数难调（λ/ε/β/GAE参数）；
- 训练不稳定（RL更新易震荡/崩塌）；
- 对生成多样性要求高。

#### （5）RL简易替代方案：BoN
- 核心策略：同一x → SFT模型生成N个回答 → RM打分 → 选最高分回答；
- 缺点：推理成本高、延迟高，仅临时替代。

### 4. DPO（工业界主流偏好调优方法）
#### （1）提出动机
将偏好调优转化为**纯有监督学习问题**，解决RLHF的复杂性问题。

#### （2）核心公式
- 损失函数（直接基于成对偏好数据，无需RM）：
  $$\mathcal{L}_{DPO}(\pi_{\theta};\pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[log\sigma\left(\beta log\frac{\pi_{\theta}(y_w|x)/\pi_{ref}(y_w|x)}{\pi_{\theta}(y_l|x)/\pi_{ref}(y_l|x)}\right)\right]$$
- 隐性奖励函数（无需训练RM，由LLM生成概率直接计算）：
  $$r_{\theta}(x,y) = \beta log\frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)}$$

#### （3）核心优势
1. 无需单独训练奖励模型/值函数，模型数量仅2个（目标模型+参考模型）；
2. 纯有监督训练，无需RL框架，工程化难度极低；
3. 直接使用成对偏好数据，数据利用率高；
4. 兼容Bradley-Terry公式，保证偏好学习的有效性。

#### （4）公式推导核心逻辑
- 从PPO最优目标出发 → 推导最优策略 → 反解隐性奖励 → 代入Bradley-Terry公式 → 负对数似然损失 → 得到DPO损失函数；
- 本质：PPO最优策略的有监督近似，用SL实现RL的最优目标。

### 5. RLHF vs DPO
| 对比维度 | RLHF | DPO |
|----------|------|-----|
| 训练方式 | 多阶段（RM+RL） | 单阶段（纯SL） |
| 所需模型 | 4个（策略/值函数/RM/基础模型） | 2个（目标模型/基础模型） |
| 依赖框架 | 专业RL框架（PPO/GAE） | 普通SL框架（PyTorch/TensorFlow） |
| 超参数 | 多（λ/ε/β/GAE等） | 少（仅β） |
| 训练稳定性 | 低（RL易震荡/崩塌） | 高（SL参数更新平稳） |
| 性能 | 精细化对齐场景略优 | 多数任务不劣于RLHF，数据差时容错性更强 |
| 工业界优先级 | 低（仅精细化对齐场景） | 高（成本低、易落地） |

### 6. 偏好调优核心效果
| 模型阶段 | 回答特点 | 示例（泰迪熊能机洗吗？） |
|----------|----------|--------------------------|
| 预训练+SFT | 客观直接，无情感 | No, it might get damaged. Try hand washing instead. |
| 预训练+SFT+偏好调优 | 友好有情感，贴合人类偏好 | It's better not to. Your teddy could get hurt! A gentle hand wash is safer. |

---

# CME 295 Lecture 6

## 一、专业术语

### 1. 推理模型相关
- **CoT**：Chain of Thought 思维链（提升推理能力的核心策略）
- **GRPO**：Group Relative Policy Optimization 群体相对策略优化（推理模型核心RL算法）
- **DAPO**：Distributed Advantage Policy Optimization 分布式优势策略优化（GRPO改进版）
- **Dr. GRPO**：Doctor GRPO（GRPO改进版，解决长度膨胀）
- **MoE**：Mixture of Experts 混合专家模型（DeepSeek V3-Base架构）
- **MLA**：Multi-Head Latent Attention 多头部潜在注意力（提升长推理链效率）
- **RMSNorm**：Root Mean Square Normalization 均方根归一化（模型层归一化方法）

### 2. 评估指标相关
- **Pass@k**：在k次生成中至少1次成功解决问题的概率（推理模型核心指标）
- **Cons@k**：Consensus at k 多数投票准确率（高风险推理任务评估）
- **Ground Truth**：地面真值（推理答案的标准答案）

## 二、核心概念

### 1. 传统LLM vs 推理模型
| 维度 | 传统LLM | 推理模型 |
|------|---------|----------|
| 核心能力 | 模式匹配+概率生成（表层能力） | 逻辑推导+步骤拆解（深层推理） |
| 输出范式 | $Question \rightarrow Answer$（黑箱输出） | $Question \rightarrow Reasoning Chain \rightarrow Answer$（透明推导） |
| 关键短板 | 有限推理、知识静态化、无法执行动作、难以评估 | 可解释、可验证，能解决复杂多步问题 |
| 形式化表示 | $Output = f(Input, Knowledge)$ | $Output = f(Input, Knowledge, Reasoning Chain)$ |

### 2. 思维链（CoT）—— 推理模型的核心范式
- **核心思想**：让模型先输出“逐步推理的自然语言文本”，再输出答案，拆解复杂问题；
- **关键延伸**：Scaling CoT（通过预训练/微调/RL让模型主动生成高质量CoT，而非依赖人工提示）；
- **识别特征**：
  1. 思考阶段可视化（显示“Thinking”状态及时长）；
  2. 推理链完整性（条件分析→步骤推导→结论）；
  3. 推理链默认隐藏（仅展示总结答案）；
- **计费规则**：推理token（含隐藏部分）+ 答案token均计入成本，占用上下文窗口。

### 3. 推理模型评估体系
#### （1）基准测试（可验证的推理任务）
| 类型 | 任务示例 | 代表数据集 | 评估方式 |
|------|----------|------------|----------|
| 代码推理 | 编程题、bug修复、代码优化 | HumanEval、CodeForces、SWE-bench | 代码通过所有测试用例 |
| 数学推理 | 奥数、竞赛题、代数方程 | AIME、GSM8K | 答案匹配地面真值 |

#### （2）核心评估指标：Pass@k
- **定义**：k次独立生成中至少1次成功的概率；
- **核心公式**：$Pass@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$（n=总尝试数，c=成功数）；
- **特殊情况**：Pass@1 = c/n（单次推理准确率）；
- **衍生指标**：Cons@k（多数投票准确率，适合高风险任务）；
- **影响因素**：
  - k越大，Pass@k越高；
  - T（温度）越高，生成越随机，Pass@k提升越明显；T越低，生成越确定，提升越平缓。

### 4. 强化学习提升推理能力
#### （1）核心思路
- 测试时扩展：推理阶段通过RL激励模型主动生成高质量CoT；
- 无需人工标注推理链，仅用“问题+答案正确性”作为监督信号；
- 奖励信号需客观可验证（代码测试用例/数学真值匹配）。

#### （2）双维度奖励函数
$$Reward = w_1 \cdot R_{CoT} + w_2 \cdot R_{Accuracy}$$（通常w1=0.3，w2=0.7）
| 奖励维度 | 验证逻辑 | 取值 |
|----------|----------|------|
| $R_{CoT}$（推理链存在） | 长度>阈值+含逻辑连接词+完整结构 | 1（满足）/0（不满足） |
| $R_{Accuracy}$（答案正确） | 代码通过测试用例/数学答案匹配真值 | 1（正确）/0（错误） |

#### （3）思维控制策略
| 策略 | 核心逻辑 | 应用场景 |
|------|----------|----------|
| 动态预算 | 按问题复杂度分配推理token预算 | 适配不同难度的推理任务 |
| 上下文感知 | 按上下文自动调整推理深度 | 数学对话（详细）/闲聊（简化） |
| 预算强制 | 强制在指定token预算内完成推理 | 低延迟场景 |
| 连续思维 | 隐空间完成推理，仅最终映射为自然语言 | 提升推理速度 |

### 5. 核心算法：GRPO（推理模型专属PPO改进版）
#### （1）核心创新：群体相对优势值
- **PPO**：单样本绝对优势值 $A_t = Q(S_t,a_t) - V(S_t)$（易聚焦局部最优）；
- **GRPO**：群体相对优势值（Z分数，反映样本在群体中的相对优势）：
  $$\hat{A}_{i,t} = \frac{r_i - mean(\{r_1,...,r_G\})}{std(\{r_1,...,r_G\})}$$
- **优势**：减少个体奖励波动，鼓励探索全局更优推理路径。

#### （2）GRPO vs PPO核心差异
| 维度 | GRPO | PPO |
|------|------|-----|
| 采样方式 | 群体采样（G条输出） | 单次采样（1条输出） |
| 优势值 | 群体相对优势（Z-score） | 单样本绝对优势（GAE） |
| 模型依赖 | 策略+奖励+参考模型（3个） | 策略+奖励+参考+价值模型（4个） |
| 损失函数 | 含群体平均+KL惩罚项 | 无群体平均，仅裁剪限制偏差 |
| 适配场景 | 推理任务（多路径求解） | 通用RL场景 |

#### （3）GRPO关键问题与解决方案
| 问题 | 成因 | 解决方案 | 效果 |
|------|------|----------|------|
| 输出长度持续增加（冗余推理步骤） | 序列平均项 $\frac{1}{|o_i|}$ 对短输出惩罚更重，模型主动生成长输出规避惩罚 | 1. DAPO：群体总token平均 $\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}$<br>2. Dr. GRPO：移除样本内token平均项 $\frac{1}{G}\sum_{i=1}^G\sum_{t=1}^{|o_i|}$ | 平均长度从12000+ tokens降至4000以下，Pass@1从0.6升至0.8 |

#### （4）GRPO其他改进
- **难度偏置调整**：按问题难度分组计算优势值，避免简单问题掩盖复杂问题能力；
- **非对称clip区间**：$[1-\varepsilon_{low},1+\varepsilon_{high}]$（如0.2/0.5），鼓励推理路径多样性。

### 6. 训练实例：DeepSeek R1/R1-Zero
#### （1）底座模型：DeepSeek V3-Base
- 架构：MoE混合专家模型（总参671B，激活参37B）；
- 核心组件：路由层（选Top-K专家）、专家层（专注特定推理任务）、MLA（缓存中间状态）、RMSNorm（轻量归一化）；
- 预训练目标：交叉熵损失 $L = -\sum_{t=1}^T \log \pi_\theta(x_t|x_{<t})$。

#### （2）R1-Zero（无SFT的概念验证版）
- 训练流程：预训练V3-Base → GRPO强化学习；
- 数据：仅“问题+答案+测试用例”，无人工推理链；
- 结果：具备基础推理能力，但推理链可读性差、效率低。

#### （3）R1（全流程优化版）
- 训练流程：预训练 → SFT-1（规范推理链格式） → GRPO-1（Dr. GRPO解决长度膨胀） → SFT-2（提升泛化） → GRPO-2（分推理/通用分支并行训练）；
- 核心逻辑：SFT学“格式”，GRPO学“正确性”，多阶段融合平衡能力。

### 7. 推理模型的蒸馏
#### （1）蒸馏范式变革
| 维度 | 传统LLM蒸馏 | 推理模型蒸馏 |
|------|-------------|--------------|
| 学习目标 | 匹配教师的下一个Token概率 | 匹配教师的“推理链+答案” |
| 核心思想 | 抄答案 | 抄解题过程 |

#### （2）蒸馏流程（R1-Distill）
1. 教师侧：DeepSeek R1生成含完整推理链的高质量样本；
2. 学生侧：小模型（如Qwen-32B）通过SFT学习教师的推理链+答案；
3. 效果：R1-Distill-Qwen-32B在AIME上Pass@1达72.6%，远超同规模模型，部署成本大幅降低。

### 8. 关键公式

### 1. 评估指标
- Pass@k：$Pass@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$
- Pass@1：$Pass@1 = \frac{c}{n}$
- GRPO群体相对优势值：$\hat{A}_{i,t} = \frac{r_i - mean(\{r_1,...,r_G\})}{std(\{r_1,...,r_G\})}$

### 2. 奖励函数
- 双维度奖励：$Reward = w_1 \cdot R_{CoT} + w_2 \cdot R_{Accuracy}$

### 3. GRPO损失函数核心结构
\[
\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)]} \Bigg[ 
& \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Bigg\{ 
\min 
\Bigg( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t}, \quad \text{clip}
\Bigg( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}, 1-\varepsilon, 1+\varepsilon 
\Bigg) 
\hat{A}_{i,t} 
\Bigg) 
\Bigg\} 
& - \beta \, \mathbb{D}_{\text{KL}}\big[ \pi_\theta \| \pi_{\text{ref}} \big]
\Bigg]
\end{aligned}
\]
### 4. 改进方案
- DAPO：$\frac{1}{\sum_{i=1}^G|o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|}$
- Dr. GRPO：$\frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|o_i|}$

---


# CME 295 Lecture 7

## 一、专业术语

### 1. 检索增强生成（RAG）
- **RAG**：Retrieval-Augmented Generation 检索增强生成
- **NDCG@k**：Normalized Discounted Cumulative Gain at K 归一化折损累积增益
- **RR@k**：Reciprocal Rank at K 倒数排名
- **Recall@k**：召回率（前k个结果中找全相关内容的能力）
- **Precision@k**：精确率（前k个结果中找对相关内容的能力）
- **bi-encoder**：双编码器（候选检索核心）
- **cross-encoder**：交叉编码器（重排序核心）

### 2. 工具调用 & 智能体
- **MCP**：Model Context Protocol 模型上下文协议（工具调用行业标准）
- **Agent**：智能体（RAG+工具调用的融合升级）
- **ReAct**：Reason + Act 推理+行动（智能体核心框架）
- **A2A**：Agent2Agent 智能体间通信协议（多智能体协作标准）
- **Agent-SafetyBench**：智能体安全评估基准

## 二、核心概念

### 1. 检索增强生成（RAG）—— 解决LLMs知识静态/上下文有限问题
#### （1）LLMs应用限制（RAG的核心动机）
| 限制类型 | 核心问题 | 案例/影响 |
|----------|----------|-----------|
| 知识截止 | 仅依赖预训练数据，无实时信息 | GPT-5截止2024.9.30，无法回答此后新事件 |
| 上下文窗口有限 | 长文本处理丢失关键信息 | GPT-5仅支持40万输入token，无法处理整本书 |
| 易被无关信息干扰 | 远端关键信息捕捉能力弱 | Needle In A HayStack测试：底部关键信息准确率随长度骤降 |
| Token计费成本 | 长文本token量剧增，成本飙升 | GPT-5输入$1.25/1M token，输出$10/1M token |

#### （2）RAG核心逻辑
- **定义**：生成回答前，从外部知识库检索相关信息补充到prompt，让LLMs“带素材生成”；
- **核心步骤**：Retrieve（检索）→ Augment（增强）→ Generate（生成）；
- **前置工作**：构建外部知识库（Collect收集→Divide切分→Embed向量化），关键超参数：Embedding size、Chunk size、Chunk overlap（解决关键信息被切分问题）。

#### （3）检索阶段（两步法）
| 阶段 | 目标 | 核心方法 | 特点 |
|------|------|----------|------|
| 候选检索 | 最大化召回率（不漏相关内容） | 1. 语义检索（bi-encoder，余弦相似度）<br>2. 关键词检索（BM25）<br>3. 混合检索 | 速度快，覆盖所有潜在相关Chunk |
| 重排序 | 最大化精确率（选最相关内容） | cross-encoder（Query+Chunk整体计算相似度） | 精度高，对候选结果精细化排序 |

#### （4）检索优化技巧
1. 伪文档（Fake document）：让查询嵌入与Chunk嵌入更匹配；
2. Chunk上下文化：补充全局上下文，避免检索偏差；
3. 提示词缓存：重复检索成本降至1/10，提升效率。

#### （5）检索性能评估（四大核心指标）
| 指标 | 核心思想 | 关键公式 | 取值范围/解读 |
|------|----------|----------|---------------|
| NDCG@k | 综合相关性+排序位置 | $DCG@k=\sum_{i=1}^k\frac{rel_i}{log_2(i+1)}$<br>$NDCG@k=\frac{DCG@k}{IDCG@k}$ | [0,1]，越接近1效果越好 |
| RR@k | 最快找到第一个相关Chunk | $RR=\frac{1}{rank}$（rank=首个相关Chunk排名） | 越大表示找到相关内容越快 |
| Recall@k | 找全所有相关Chunk | $Recall@k=\frac{相关在topk的数量}{总相关数量}$ | [0,1]，越接近1召回率越高 |
| Precision@k | 找对相关Chunk | $Precision@k=\frac{相关在topk的数量}{k}$ | [0,1]，越接近1精确率越高 |

### 2. 工具调用（Tool calling）—— 解决LLMs无法执行实际操作的问题
#### （1）核心定位
- 分工：RAG解决非结构化文本知识补充，工具调用解决结构化数据/计算/实际操作；
- 本质：LLM做“大脑”（解析参数），工具做“手脚”（执行操作）。

#### （2）核心步骤
1. 解析参数：LLM结合prompt+工具API文档，确定调用参数；
2. 执行调用：参数传入工具，调用后端/API完成操作，返回结构化结果；
3. 生成回答：LLM基于结构化结果生成自然语言回答。

#### （3）LLM使用工具的两种方法
| 方法 | 核心思路 | 优缺点 |
|------|----------|--------|
| 训练法（微调） | 用“prompt+API+参数+结果+回答”数据集微调 | 优点：效果稳定、准确率高<br>缺点：标注成本高、周期长 |
| 提示词法 | 工具API文档+使用说明写入prompt，零训练 | 优点：灵活高效、迭代快<br>缺点：对prompt要求高、复杂工具易出错 |

#### （4）多工具场景解决方法
- 核心问题：工具越多，匹配准确率越低、上下文占用越多；
- 解决方案：引入Router模块，先解析prompt筛选少量高相关工具，再调用。

#### （5）标准化：MCP协议
- 动机：解决不同LLM与工具对接不统一、重复开发的问题；
- 核心：制定LLM与工具/数据对接的标准协议，实现互联互通；
- 架构：MCP host → MCP client → MCP server → Tools/Prompts/Resources。

### 3. 智能体（Agents）—— RAG+工具调用的融合升级
#### （1）核心定义
- 智能体：能自主追求目标、代表用户完成端到端复杂任务的系统，核心是“自主决策”；
- 能力升级：传统LLM（无推理无操作）→ 推理LLM（有推理无操作）→ 智能体（有推理+自主操作）。

#### （2）核心框架：ReAct（Reason + Act）
- 流程：Input → (Observe → Plan → Act)循环 → Output；
- 各环节核心：
  - Observe：综合先前行动+明确已知内容，推理核心矛盾；
  - Plan：制定任务步骤，确定调用工具；
  - Act：执行工具调用，获取新信息，重新推理方向；
  - 循环：直至完成任务，生成最终Output。

#### （3）多智能体协作：A2A协议
- 动机：解决单一智能体仅能完成单一领域任务的问题；
- 核心组件：
  1. AgentSkill：智能体核心能力（ID、描述、示例）；
  2. AgentCard：智能体身份卡片（名称、地址、技能列表）；
  3. AgentExecutor：智能体执行逻辑（execute()/cancel()）；
- 价值：实现不同智能体的标准化通信与协同工作。

#### （4）智能体安全问题
- 核心风险：数据泄露、AI策划的网络攻击；
- 解决措施：训练对齐、推理防护、Agent-SafetyBench评估、行业监管。

### 4. 工程实践核心原则
1. 由简入繁，迭代升级：先实现简单RAG/单一工具调用，验证后扩展为多工具/智能体；
2. 先选强模型，再做轻量化：先用GPT-5/DeepSeek-R1验证方案，再轻量化平衡性能与成本；
3. 提升透明度/可观测性：让推理/调用/决策过程可追溯，便于调试和提升信任。


## 三、核心对比
### 1. 传统LLM vs 推理LLM vs 智能体
| 维度 | 传统LLM | 推理LLM | 智能体 |
|------|---------|----------|--------|
| 核心流程 | Question→LLM→Answer | Question→LLM→Reasoning→Answer | Question→LLM→Calls→LLM→...→Answer |
| 推理能力 | 无 | 有（CoT） | 有（ReAct循环） |
| 操作能力 | 无（仅文本生成） | 无 | 有（自主调用工具/其他智能体） |
| 核心目标 | 文本生成 | 准确推理 | 自主完成复杂任务 |

### 2. RAG vs 工具调用
| 维度 | RAG | 工具调用 |
|------|-----|----------|
| 解决问题 | 非结构化文本知识补充（文档/网页） | 结构化数据查询/计算/实际操作 |
| 数据类型 | 非结构化文本 | 结构化数据（数据库/API返回） |
| 核心动作 | 检索→增强→生成 | 解析参数→执行调用→生成回答 |
| 核心价值 | 缓解知识截止/幻觉 | 让LLM具备实际操作能力 |

### 3. RAG检索性能指标
- DCG@k：$DCG@k = \sum_{i=1}^k \frac{rel_i}{log_2(i+1)}$
- NDCG@k：$NDCG@k = \frac{DCG@k}{IDCG@k}$
- RR@k：$RR = \frac{1}{rank}$
- Recall@k：$Recall@k = \frac{|relevant\ in\ top\ k|}{|relevant|}$
- Precision@k：$Precision@k = \frac{|relevant\ in\ top\ k|}{k}$


---

# CME 295 Lecture 8

## 一、专业术语

### 1. 评估指标类
- **METEOR**：Metric for Evaluation of Translation with Explicit Ordering（带有显式排序的翻译评估指标）
- **BLEU**：Bilingual Evaluation Understudy（面向精度的机器翻译评估指标）
- **ROUGE**：Recall-Oriented Understudy for Gisting Evaluation（面向召回率的摘要评估指标）
- **LaaJ**：LLM-as-a-Judge（以LLM作为评估器）

### 2. 基准测试类
- **MMLU**：Massive Multitask Language Understanding（大规模多任务语言理解，知识维度）
- **AIME**：American Invitational Mathematics Examination（美国数学邀请考试，数学推理维度）
- **PIQA**：Physical Interaction: Question Answering（物理常识问答，常识推理维度）
- **SWE-bench**：Software Engineering benchmark（软件工程基准，编码维度）
- **HarmBench**：Harmful Behavior Benchmark（有害行为基准，安全维度）
- **τ-bench**：Tool-Agent-User Interaction Benchmark（工具-智能体-用户交互基准，智能体专属）

### 3. 其他核心缩写
- **ASR**：Attack Success Rate（攻击成功率，安全评估指标）
- **Pass^k**：k次尝试全部成功的概率（智能体可靠性核心指标）

## 二、核心概念

### 1. LLM评估的核心维度与方式
#### （1）评估的双重维度
| 维度 | 核心子指标 | 评估目标 |
|------|------------|----------|
| 输出质量 | 指令遵循、连贯性、事实性、有用性、相关性 | 模型回答的内容质量 |
| 系统性能 | 延迟、定价、可靠性 | 模型落地的工程性能 |

#### （2）评估方式的演进（从人工到自动化）
| 评估方式 | 核心逻辑 | 优点 | 缺点 |
|----------|----------|------|------|
| 人工评分（黄金标准） | 人类基于预设标准打分 | 贴近真实人类判断 | 主观性强、速度慢、成本高 |
| 规则化指标（METEOR/BLEU/ROUGE） | 固定算法对比输出与参考标签 | 自动化、速度快 | 无法适配风格变体、与人类评分相关性差、依赖人工标注 |
| LLM-as-a-Judge（主流） | 用LLM作为“法官”直接评估输出 | 无参考标签、可解释、多维度评估 | 存在位置/冗长/自我增强偏置 |

### 2. 人工评分的核心补充：卡帕系数（$\kappa$）
- **核心公式**：$\kappa = \frac{p_o - p_e}{1 - p_e}$
  - $p_o$：实际观察到的一致性率（评估者达成共识的比例）
  - $p_e$：随机期望的一致性率（纯随机下的共识比例）
- **核心作用**：剔除随机一致性，量化评估者的真实共识程度，$\kappa$越接近1，一致性越强
- **常见变体**：
  - Cohen's Kappa：2名评估者，二分类/名义尺度
  - Fleiss' Kappa：多名评估者，名义尺度
  - Krippendorff's alpha：通用型，适配任意评估者数量/数据尺度

### 3. 规则化评估指标（三大核心）
| 指标 | 核心公式/逻辑 | 侧重点 | 适用场景 |
|------|---------------|--------|----------|
| METEOR | $METEOR = F_{mean} \times (1-p)$<br>$F_{mean}$：P/R调和均值；$p$：词序/词形惩罚项 | 召回率+精确率+语言特征 | 机器翻译、文本摘要 |
| BLEU | $BLEU = BP \times exp(\frac{1}{N}\sum_{n=1}^N log(p_n))$<br>$BP$：简短惩罚项；$p_n$：n-gram精确率 | 精确率（n-gram匹配） | 机器翻译（大规模快速评估） |
| ROUGE | ROUGE-N：n元语法召回率<br>ROUGE-L：最长公共子序列（LCS） | 召回率（信息覆盖度） | 文本摘要 |

### 4. LLM-as-a-Judge（LaaJ）核心要点
#### （1）核心逻辑
- 输入：Prompt + Model Response + Criteria（评估标准）
- 输出：Rationale（评价理由） + Score（评分）

#### （2）两大变体
| 类型 | 评估方式 | 输出形式 | 适用场景 |
|------|----------|----------|----------|
| Pointwise（点式） | 单独评估单个回答 | 绝对分数（0/1、0-10分） | 单模型效果评估 |
| Pairwise（配对） | 对比两个回答 | 相对评价（A更好/B更好） | 多模型对比、Chatbot Arena |

#### （3）三大固有偏置及解决方案
| 偏置类型 | 核心问题 | 解决方案 |
|----------|----------|----------|
| 位置偏置 | 偏向排序靠前的回答 | 多次评估取平均、调整位置嵌入 |
| 冗长偏置 | 偏向更长的回答 | 明确评价准则、长度惩罚、少样本引导 |
| 自我增强偏置 | 偏向自身生成的回答 | 使用第三方高性能LLM作为评估器 |

#### （4）实践原则（核心易考点）
1. 评估准则清晰简洁，减少主观模糊性；
2. 优先使用二值评分（0/1/PASS/FAIL），替代细粒度评分；
3. 先输出理由，再输出分数；
4. 针对性缓解各类偏置；
5. 用人类评分校准LaaJ；
6. 设置低温度参数，保证输出稳定可复现。

#### （5）现代评估工作流
$\text{LLM Output} \to \text{LaaJ（初评）} \to \text{Human ratings（校准）} \to \text{Final Result}$

### 5. 事实性量化（解决幻觉问题）
- **核心方法**：事实分解 + 加权求和
- **步骤**：
  1. 事实拆解：将长文本输出拆分为多个独立事实陈述；
  2. 权重分配：为每个事实分配重要性权重$\alpha_i$（$\sum\alpha_i=1$）；
  3. 真实性评分：每个事实打0/1分（错误/正确）；
  4. 加权求和：$score = \sum_{i=1}^n \alpha_i \times score_i$。
- **核心价值**：精准定位幻觉点，为模型优化提供方向。

### 6. 智能体工具调用故障模式（全流程）
#### （1）工具调用核心流程
$\text{LLM确定工具/生成参数} \to \text{工具执行} \to \text{LLM整合结果生成答案}$

#### （2）各环节故障类型（最常见：工具预测错误）
| 环节 | 故障类型 | 核心症状 | 解决方法 |
|------|----------|----------|----------|
| LLM确定工具/参数 | 未使用工具 | 直接回答，不调用工具 | 重训练Router、SFT微调、优化提示词 |
|  | 幻觉出工具 | 调用不存在的工具 | 升级模型、重构API命名、优化指令 |
|  | 使用错误工具 | 调用不匹配的工具 | 同“未使用工具” |
|  | 参数推断错误 | 传入错误/无效参数 | 补充上下文、SFT微调、优化提示词 |
| 工具执行 | 返回错误结果 | 工具返回错误值/抛异常 | 修复工具代码、增加异常处理 |
|  | 无返回结果 | 工具无输出，LLM幻觉生成答案 | 工具返回空JSON、设计兜底输出、增加监控 |
| LLM整合结果 | 生成错误回答 | 答案与工具结果不符 | 升级LLM、裁剪冗余输出、优化工具输出格式 |

### 7. 标准化基准测试（全维度）
| 能力维度 | 基准测试 | 测试内容 | 评价标准 |
|----------|----------|----------|----------|
| 知识 | MMLU | 57个领域4选1选择题 | 标准答案准确率 |
| 数学推理 | AIME | 约30道数学题（几何/代数） | 3位数字正确答案匹配 |
| 常识推理 | PIQA | 2万道物理常识2选1题 | 标准答案准确率 |
| 编码 | SWE-bench | 2294个真实GitHub Python问题 | PR通过所有测试用例 |
| 安全 | HarmBench | 510种有害行为测试 | 攻击成功率（ASR），越低越安全 |
| 智能体工具调用 | τ-bench | 航空/零售真实场景工具调用 | 奖励值、Pass^k（核心） |

#### （1）智能体核心指标：Pass^k
- **定义**：k次尝试全部成功的概率
- **公式**：$\text{Pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}$
  - $n$：总尝试次数；$c$：成功次数；$\binom{x}{y}$：组合数
- **核心意义**：衡量智能体工具调用的一致性/可靠性，值越高越可靠

#### （2）基准测试两大核心问题
| 问题 | 定义 | 解决方法 |
|------|------|----------|
| 数据污染 | 基准测试题目出现在训练集，评估失真 | 哈希标记、工具黑名单、新版本测试集、盲测 |
| 古德哈特定律 | 指标成目标后，模型“刷分”但实际能力脱节 | 不过度依赖、结合有机评估、多维度评估、真实场景测试 |

### 8. 帕累托前沿（Pareto frontier）
- **定义**：优化权衡关系的最优解集合，曲线上的模型无法在提升一个指标的同时不降低另一个；
- **常见权衡维度**：性能vs成本/延迟、性能vs安全、性能vs上下文长度；
- **核心价值**：为LLM落地选型提供最优解参考。

## 四、核心对比与公式
### 1. 三大规则化评估指标
| 指标 | 核心侧重 | 匹配方式 | 适用场景 | 局限性 |
|------|----------|----------|----------|--------|
| METEOR | 召回率+精确率+语言特征 | 词干、同义词、词序 | 翻译、摘要 | 计算稍复杂 |
| BLEU | 精确率（n-gram） | 纯n-gram匹配 | 机器翻译 | 不考虑召回率、对短文本不友好 |
| ROUGE | 召回率（信息覆盖） | n-gram、LCS | 文本摘要 | 忽略语序、易受冗余信息影响 |

### 2. LaaJ的Pointwise vs Pairwise
| 维度 | Pointwise | Pairwise |
|------|-----------|----------|
| 评估对象 | 单个回答 | 两个回答对比 |
| 输出类型 | 绝对分数（0/1、0-10） | 相对评价（A更好/B更好） |
| 核心优势 | 单模型量化评分 | 多模型效果对比 |
| 适用场景 | 单模型迭代、单维度打分 | 模型选型、Chatbot Arena |

### 1. 评估一致性
- 卡帕系数：$\kappa = \frac{p_o - p_e}{1 - p_e}$

### 2. 规则化评估指标
- METEOR：$METEOR = F_{mean} \times (1-p)$，$F_{mean} = \frac{P \times R}{\alpha P + (1-\alpha)R}$
- BLEU：$BLEU = BP \times exp\left(\frac{1}{N}\sum_{n=1}^N log(p_n)\right)$，$BP = min\left(1, exp\left(1-\frac{n_{ref}}{n_{pred}}\right)\right)$

### 3. 事实性量化
- 加权求和：$score = \sum_{i=1}^n \alpha_i \times score_i$（$\sum\alpha_i=1$）

### 4. 智能体可靠性
- Pass ^ k：Pass ^ k = $\frac{\binom{c}{k}}{\binom{n}{k}}$

---

# CME 295 Lecture 9

## 一、专业术语

### 1. 多模态架构类
- **ViT**：Vision Transformer（视觉Transformer）
- **VLM**：Vision Language Model（视觉语言模型）
- **MDM**：Masked Diffusion Model（掩码扩散模型，文本扩散核心）
- **DiT**：Diffusion Transformer（扩散Transformer，图像扩散）
- **MSRoPE**：Multi-modal Scalable Rotary Position Embedding（多模态可扩展旋转位置编码）

### 2. 模型范式/硬件类
- **ARM**：AutoRegressive Modeling（自回归建模，传统LLM生成范式）
- **MoE**：Mixture of Experts（混合专家模型）
- **MHA**：Multi-Head Attention（多头注意力）
- **MQA**：Multi-Query Attention（多查询注意力）
- **GQA**：Grouped Query Attention（分组查询注意力）

## 二、核心概念

### 1. Transformer多模态泛化核心
- **本质**：弱归纳偏置 + QKV自注意力的强关联捕捉能力；
- **关键**：任意模态数据→**离散Token/连续特征向量 + 位置信息**→Transformer统一建模；
- **结论**：Transformer从“文本专属”升级为“多模态统一基础架构”。

### 2. ViT（视觉Transformer）核心逻辑
#### （1）核心改造：图像“文本化”建模
| 图像处理步骤 | 对应文本处理逻辑 | 核心作用 |
|--------------|------------------|----------|
| 图像切割为N个固定大小Patch（如16×16） | 文本分词为Token | 离散化视觉输入，适配Transformer输入格式 |
| Patch线性投影 + 二维位置嵌入 | 词嵌入 + 位置编码 | 将视觉Patch映射为高维特征，赋予空间位置信息 |
| 增加可学习[CLS]嵌入 | 文本分类标记[CLS] | 聚合全局视觉特征，用于分类任务 |
| 送入标准Transformer Encoder | 文本Encoder | 统一的特征编码逻辑 |
| [CLS]输出→全连接层分类 | 文本[CLS]→分类 | 输出任务结果 |

#### （2）案例：泰迪熊图像识别
输入（泰迪熊图像）→ 切块→Patch投影+位置嵌入+[CLS]→6层Transformer Encoder→[CLS]特征→FFN→分类为“teddy bear”。

### 3. VLM（视觉语言模型）两大主流架构
| 架构类型 | 核心流程 | 优势 | 应用案例 |
|----------|----------|------|----------|
| LLM解码器复用型 | ViT编码视觉特征→投影层映射到LLM词嵌入空间→纯文本LLM解码器生成回答 | 复用成熟LLM权重，开发成本低 | LLaMA 3 7B多模态版 |
| 交叉注意力融合型 | ViT生成视觉特征→LLM解码器新增交叉注意力层（文本Query关注视觉KV）→特征交互后生成回答 | 视觉-文本深度融合，效果更优 | GPT-4V、Gemini Pro Vision |

#### （3）核心关键：模态空间对齐
- **问题**：视觉特征与文本Token嵌入空间维度/分布不同；
- **解决方案**：可学习的投影矩阵，将视觉特征映射到LLM一致的高维嵌入空间。

### 4. 各模态Transformer适配方式（核心易考）
| 模态 | 核心处理方式 | 关键改造 |
|------|--------------|----------|
| 视频 | 帧级切块 + 时空位置嵌入 | 增加时间步位置信息（叠加空间位置） |
| 语音 | Conformer块（Transformer+CNN）编码→跨注意力融合LLM | 适配语音的时序特征 |
| 推荐系统 | 用户/物品作为Token，自注意力捕捉关联 | 建模用户-物品、物品-物品关系 |
| 代码生成 | 代码语法结构转Token，自注意力捕捉上下文逻辑 | 适配代码的语法/逻辑关联 |

### 5. Diffusion LLMs（扩散LLMs）核心
#### （1）背景：传统ARM的核心缺陷
- 生成逻辑：$p(x_t | x_1,...,x_{t-1})$，逐Token生成；
- 核心瓶颈：推理无法并行，长文本生成延迟极高；
- 工业痛点：并行策略无法突破“逐Token生成”本质，用户等待时间长。

#### （2）扩散模型适配文本：MDM（掩码扩散模型）
| 阶段 | 图像扩散（高斯噪声） | 文本扩散（掩码替代） | 核心公式 |
|------|----------------------|----------------------|----------|
| 前向过程 | 逐步加高斯噪声→完全噪声图$x_T$ | 逐步掩码Token→全掩码序列$x_T$ | $q(x_t | x_{t-1}) = \prod_{i=1}^N p(x_{t,i} | x_{t-1,i})$ |
| 逆向过程 | 逐步去噪→还原$x_0$ | 逐步解掩码→还原$x_0$ | $p_\theta(x_{0:T}) = p(x_T) \prod_{t=T}^1 p_\theta(x_{t-1} | x_t)$ |

#### （3）Diffusion LLMs的优劣势
| 优势 | 挑战 |
|------|------|
| 推理并行（T步前向传播，T<<N） | 效果略低于最优ARM模型 |
| 全局上下文捕捉（适配长文本/代码） | ARM优化技巧（RoPE/GQA/MoE）未完全适配 |
| 低延迟（Tokens生成速度提升10倍） | 超参数（T/$\gamma_t$）敏感，无统一调优标准 |

### 6. 跨模态技术交叉融合（核心方向）
| 交叉维度 | 迁移方向 | 代表成果 | 核心价值 |
|----------|----------|----------|----------|
| 架构 | 图像→文本 | Diffusion LLMs（MDM） | 解决ARM推理并行问题 |
| 架构 | 文本→图像 | DiT（Diffusion Transformer） | 提升图像生成质量 |
| 输入表示 | 文本→视觉 | DeepSeek-OCR（上下文光学压缩） | 提升VLM推理速度 |
| 工程技巧 | 文本→多模态 | MSRoPE（二维RoPE） | 支持任意分辨率图像输入 |

### 7. LLM基础研究与发展趋势
#### （1）Transformer核心组件研究（无统一最优解）
| 组件 | 主流方案 | 新型方案 | 研究焦点 |
|------|----------|----------|----------|
| 优化器 | AdamW | MuonClip | 收敛速度、显存占用 |
| 归一化层 | LayerNorm（Pre-LN） | RMSNorm | 训练稳定性、推理速度 |
| 注意力 | MHA | MQA、GQA | 效率与表达能力平衡 |
| 激活函数 | RELU | GELU、SwiGLU | 非线性能力、计算成本 |
| 架构 | 稠密模型 | MoE | 参数量效率、任务适配性 |

#### （2）数据层面：递归诅咒
- **定义**：用LLM生成数据训练新模型→模型遗忘真实知识、性能退化、幻觉增多；
- **对策**：优先用人类标注高质量真实数据，控制生成数据比例。

#### （3）优化目标转变
- 2025前：追求“更大、更强、更好”；
- 2025起：追求“性能-成本帕累托最优解”（无法在提升一个指标时不降低另一个）。

#### （4）硬件优化：类存内计算
- **GPU痛点**：为矩阵运算优化，Transformer的KV缓存读写（内存移动）成瓶颈；
- **核心设计**：专用KV缓存单元+模拟信号计算注意力相似度；
- **实测效果**：对比H100，推理延迟降100倍，能耗省70000倍；
- **应用前景**：推动LLM边缘端落地，解决高成本/高能耗问题。

### 8. LLM应用与挑战
#### （1）当下核心应用
- 代码开发：生成、调试、文本转SQL；
- 通用对话：问答、信息检索、多轮对话；
- 创意生成：文案、小说、绘画提示词；
- 教育科研：知识点讲解、论文润色、数据处理。

#### （2）发展趋势
| 时间维度 | 趋势 | 核心场景 |
|----------|------|----------|
| 短期（1-3年） | 智能体民主化 | 办公软件集成AI智能体 |
|  | LLM+浏览器深度融合 | 网页总结、智能搜索、自动填表 |
|  | 操作系统级LLM | 自然语言控制全系统 |
| 长期（3-10年） | 自治智能体规模化 | 工厂调度、医疗诊断 |
|  | 全场景智能客服 | 替代传统人工客服 |

#### （3）核心挑战
| 类型 | 具体问题 | 影响 |
|------|----------|------|
| 技术挑战 | 无持续学习能力、幻觉、可解释性差、个性化不足 | 限制医疗/金融等关键任务落地 |
| 伦理挑战 | 安全对齐、数据隐私、就业冲击 | 需政策+技术双重约束 |

### 9. 前沿科研方法（易考“如何跟进LLM前沿”）
| 渠道类型 | 核心平台/资源 | 核心价值 |
|----------|---------------|----------|
| 论文 | arXiv（cs.CL）、NeurIPS/ICML/ICLR（ML顶会）、ACL/EMNLP（NLP顶会） | 最快获取最新研究（arXiv提前顶会3-6个月） |
| 代码 | 论文作者GitHub、Hugging Face（Transformers/Datasets/Model Hub） | 获取权威复现代码、最新模型/数据集 |
| 资讯 | Twitter（X）（Andrej Karpathy等学者）、YouTube（Two Minute Papers/Yannic Kilcher）、企业技术博客（Google DeepMind/Meta AI） | 实时跟进研究进展与工业落地成果 |

## 三、核心对比与公式

### 1. VLM两大主流架构
| 架构 | 流程 | 优势 | 劣势 | 应用案例 |
|------|------|------|------|----------|
| LLM解码器复用型 | ViT→投影层→纯文本LLM解码器 | 复用LLM权重，开发成本低 | 视觉-文本融合浅 | LLaMA 3 7B多模态 |
| 交叉注意力融合型 | ViT→LLM解码器（新增交叉注意力） | 特征深度融合，效果优 | 开发成本高，需修改LLM架构 | GPT-4V、Gemini Pro Vision |

### 2. ARM vs Diffusion LLMs
| 维度 | ARM（自回归LLM） | Diffusion LLMs（MDM） |
|------|------------------|-----------------------|
| 生成逻辑 | 逐Token生成（$p(x_t|x_1..x_{t-1})$） | 掩码-解掩码（T步并行） |
| 推理并行性 | 完全无法并行 | 支持并行（T步<<序列长度N） |
| 延迟 | 高（长文本等待久） | 低（Tokens生成速度提升10倍） |
| 上下文捕捉 | 局部（依赖前序Token） | 全局（依赖完整掩码序列） |
| 效果 | 常识/推理任务更优 | 稍低于最优ARM模型 |

### 1. 图像扩散前向过程
$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$

### 2. 文本扩散（MDM）前向过程
$q(x_t | x_{t-1}) = \prod_{i=1}^N p(x_{t,i} | x_{t-1,i})$

### 3. 文本扩散（MDM）逆向过程
$p_\theta(x_{0:T}) = p(x_T) \prod_{t=T}^1 p_\theta(x_{t-1} | x_t)$
