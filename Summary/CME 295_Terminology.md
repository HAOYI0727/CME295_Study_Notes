# CME 295 Terminology

## 一、自然语言处理领域基础任务

- $NLP（Natural \ Language \ Processing）$：**自然语言处理**——研究计算机理解和生成人类语言的技术
- $NLG（Natural \ Language \ Generation）$：**自然语言生成**——根据输入，生成连贯的自然语言文本
- $NLU（Natural \ Language \ Understanding）$：**自然语言理解**——从文本或语音中读懂语义、识别意图、理解上下文、处理歧义
- $NER（Named \ Entity \ Recognition）$：**命名实体识别**——识别文本中的人名、地名、机构名等
- $PoS（Part{-}of{-}Speech）$：**词性标注**——标注文本中每个词的词性（名词、动词等）
- $MT（Machine \ Translation）$：**机器翻译**——实现不同语言间的文本转换
- $QA（Question \ Answering）$：**问答**——根据输入问题，生成准确回答
- $RE（Relation \ Extraction）$：**关系抽取**——识别实体之间的语义关系
- $EE（Event \ Extraction）$：**事件抽取**——从文本中抽取出事件结构
- $SRL（Semantic \ Role \ Labeling）$：**语义角色标注**——识别谓词-论元结构（谁做、做什么、对谁做）
- $SA（Sentiment \ Analysis）$：**情感分析**——判断文本情感倾向（正面/负面/中性）
- $TC（Text \ Classification）$：**文本分类**——将文本分到预定义类别
- $TS（Text \ Summarization）$：**文本摘要**——生成原文精简且关键信息完整的摘要
- $ASR（Automatic \ Speech \ Recognition）$：**自动语音识别**——语音转文字
- $TTS（Text{-}to{-}Speech）$：**文本转语音**——文字生成语音
- $LM（Language \ Model）$：**语言模型**——建模文本序列的概率分布
  
## 二、基础模型 & 架构

- $Transformer$：**Transformer**——大模型核心基础架构，仅用注意力机制，支持全并行计算，可泛化至多模态任务
- $BERT（Bidirectional \ Encoder \ Representations \ from \ Transformers）$：**双向编码器表征**——仅编码器架构，擅长双向上下文理解，预训练任务为MLM和NSP
- $GPT（Generative \ Pre{-}trained \ Transformer）$：**生成式预训练Transformer**——仅解码器架构，擅长自回归文本生成
- $T5（Text{-}to{-}Text \ Transfer \ Transformer）$：**文本到文本迁移Transformer**——编码器-解码器架构，将所有NLP任务统一为文本到文本形式
- $LLaMA（Large \ Language \ Model \ Meta \ AI）$：**Meta大语言模型**——Meta推出的开源仅解码器大语言模型，可扩展多模态版本
- $RNN（Recurrent \ Neural \ Network）$：**循环神经网络**——循环神经网络基础款，传统序列建模方案
- $LSTM（Long \ Short{-}Term \ Memory）$：**长短期记忆网络**——解决RNN长距离依赖问题的循环神经网络变体
- $GRU（Gated \ Recurrent \ Unit）$：**门控循环单元**——简化版LSTM，解决长距离依赖问题的循环神经网络变体
- $MoE（Mixture \ of \ Experts）$：**混合专家模型**——拆分模型为多个子专家，仅激活部分计算，降低成本，可用于推理模型底座（如DeepSeek V3-Base）
- $DistilBERT（Distilled \ BERT）$：**蒸馏版BERT**——12层→6层，参数量减40%，保留97%性能，推理更快
- $RoBERTa（Robustly \ Optimized \ BERT \ Pretraining \ Approach）$：**鲁棒优化BERT预训练方法**——优化版BERT，移除NSP、采用动态掩码，性能优于原始BERT
- $Encoder{-}only$：**仅编码器架构**——代表模型BERT/RoBERTa，擅长分类、NER等理解类任务
- $Decoder{-}only$：**仅解码器架构**——代表模型GPT/LLaMA，擅长文本生成类任务
- $LLM（Large \ Language \ Model）$：**大语言模型**——数十亿级参数，核心是基于上下文预测下一个token的条件概率



## 三、分词 & 嵌入

- $Tokenization$：**分词**——将文本拆成模型能处理的最小单位（Token）
- $BPE（Byte \ Pair \ Encoding）$：**字节对编码**——主流子词分词方法，平衡词汇量与未登录词
- $WordPiece$：**词片段分词法**——子词分词方法，BERT专用，基于贪心算法拆分文本
- $SentencePiece$：**句子级子词分词**——不依赖预分词，支持多语言，T5/LLaMA常用
- $OOV（Out{-}of{-}Vocabulary）$：**未登录词**——不在模型词汇表中的词
- $Word \ Embedding$：**词嵌入**——将词映射为低维稠密向量，捕捉语义信息
- $Token \ Embedding$：**Token嵌入**——模型最底层输入向量
- $Segment \ Embedding$：**段落/句子段嵌入**——BERT区分两句话用
- $Contextual \ Embedding$：**上下文嵌入**——根据语境动态生成，如BERT/GPT输出
- $Word2Vec$：**词向量模型**——浅层神经网络词嵌入，CBOW、Skip-gram两种训练方式
- $GloVe（Global \ Vectors \ for \ Word \ Representation）$：**全局词向量**——结合全局词频信息，生成高质量词嵌入
- $Modal \ Space \ Alignment$：**模态空间对齐**——VLM核心技术，通过可学习投影矩阵，将视觉特征映射到LLM一致的嵌入空间

## 四、位置编码 & 归一化

- $PE（Positional \ Encoding）$：**位置编码**——补充Transformer的序列位置信息，解决位置丢失问题，分绝对和相对两种
- $CLS$：**分类标记**——BERT专用特殊标记，用于分类任务
- $SEP$：**分隔标记**——BERT专用特殊标记，用于分隔句子
- $PAD$：**补全标记**——BERT专用特殊标记，用于补齐序列长度
- $MASK$：**掩码标记**——BERT专用特殊标记，用于MLM任务遮蔽token
- $Absolute \ PE$：**绝对位置编码**——分可学习PE（无法扩展长序列）和硬编码PE（三角函数，可扩展）
- $Relative \ PE$：**相对位置编码**——主流方案，含线性偏置（T5 Bias/ALiBi）和RoPE，内积仅依赖token相对位置
- $Sinusoidal \ PE$：**三角函数位置编码**——Transformer原版绝对位置编码，无需训练
- $Learnable \ PE$：**可学习位置编码**——BERT/GPT-2使用，固定长度，长序列外推弱
- $ALiBi（Attention \ with \ Linear \ Biases）$：**线性偏置注意力**——无需显式位置编码，靠注意力偏置注入相对位置，长文本外推友好
- $RoPE（Rotary \ Position \ Embeddings）$：**旋转位置编码**——主流位置编码，支持长序列外推，LLaMA/GPT-4标配
- $MSRoPE（Multi{-}Scale \ Rotary \ Position \ Embeddings）$：**多模态可扩展旋转位置编码**——适配多模态任务，支持任意分辨率输入
- $LN（Layer \ Normalization）$：**层归一化**——按样本特征归一化，解决内部协变量偏移
- $RMSNorm（Root \ Mean \ Square \ Layer \ Normalization）$：**均方根归一化**——简化LN，减少计算量，GPT-3/LLaMA标配
- $Pre{-}Norm$：**前归一化**——LN在残差前的归一化方式，训练更稳定，适配深层模型
- $Post{-}Norm$：**后归一化**——LN在残差后的归一化方式，传统Transformer使用

## 五、注意力 & 结构细节

- $MHA（Multi{-}Head \ Attention）$：**多头注意力**——多组自注意力并行，捕捉不同维度语义关联，原始版注意力方案
- $MQA（Multi{-}Query \ Attention）$：**多查询注意力**——多个Q头共享1个K/V头，提升推理效率，适配低延迟场景
- $GQA（Group{-}Query \ Attention）$：**分组查询注意力**——Q头分组共享K/V头，平衡效率与效果，GPT-3.5/4主流
- $Self{-}Attention$：**自注意力**——让每个token关注句子中所有其他token，捕捉长距离依赖，Transformer核心
- $Cross{-}Attention$：**交叉注意力**——编解码器架构专属，Decoder关注Encoder的特征，VLM融合视觉与文本的核心组件
- $SDPA（Scaled \ Dot{-}Product \ Attention）$：**缩放点积注意力**——注意力核心计算方式，公式为 $Attention(Q,K,V) = softmax((QKᵀ)/√dₖ)V$ ，$√dₖ$解决内积爆炸
- $MMHA（Masked \ Multi{-}Head \ Attention）$：**掩码多头注意力**——Decoder专属，将注意力得分矩阵上三角置为-∞，防止模型看到未来token
- $MLA（Multi{-}Head \ Latent \ Attention）$：**多头部潜在注意力**——压缩KV向量，提升长推理链效率，推理模型核心组件
- $SWA（Sliding \ Window \ Attention）$：**滑动窗口注意力**——每个token仅关注局部窗口，适配长序列，属于稀疏注意力
- $Flash \ Attention$：**加速注意力**——分块加载Q/K/V到SRAM，重计算中间结果，节省显存且提升速度，无性能损失
- $FFN（Feed{-}Forward \ Network）$：**前馈网络**——Transformer每层标配，两层线性+激活函数，负责特征变换
- $AdamW$：**优化器**——LLM训练主流优化器，基于Adam改进，加入权重衰减，提升训练稳定性
- $GeLU（Gaussian \ Error \ Linear \ Unit）$：**高斯误差线性单元**——Transformer常用激活函数，比ReLU更平滑
- $Swish/SiLU$：**平滑激活函数**——升级版平滑激活，部分现代大模型使用
- $Residual \ Connection$：**残差连接**——解决深层网络梯度消失，保证模型可深可训练

## 六、预训练任务

- $ARLM（AutoRegressive \ Language \ Modeling）$：**自回归语言建模**——按照顺序，一个一个token地生成文本，每一步只根据“前面已经生成的内容”来预测下一个 token
- $MLM（Masked \ Language \ Model）$：**掩码语言模型**——随机遮蔽token并预测，学习双向上下文，BERT核心预训练任务
- $CLM（Causal \ Language \ Model）$：**因果语言模型**——自回归预测下一个token，仅能看见前文，GPT/LLaMA核心预训练任务
- $PLM（Permuted \ Language \ Model）$：**排列语言模型**——随机打乱token顺序再预测，XLNet使用，解决MLM不能建模依赖的问题
- $NSP（Next \ Sentence \ Prediction）$：**下一句预测**——预测句子间的逻辑关联，BERT预训练任务，RoBERTa已移除
- $NTP（Next \ Token \ Prediction）$：**下一个令牌预测**——LLM预训练核心目标函数，最小化预测令牌与真实令牌的交叉熵损失，自回归训练
- $SOP（Sentence \ Order \ Prediction）$：**句子顺序预测**——交换句子顺序做二分类，ALBERT使用，效果优于NSP
- $Contrastive \ Learning$：**对比学习**——拉近正样本距离、拉远负样本距离，用于多模态（如CLIP）和语义表征模型
  
## 七、训练 & 微调

- $FFT（Full \ Fine{-}Tuning）$：**全参数微调**——更新模型全部权重，效果通常最好，但显存/计算成本极高
- $SFT（Supervised \ Fine{-}Tuning）$：**监督微调**——用人工标注的输入输出数据，微调预训练模型适配任务，可用于规范推理链格式
- $PEFT（Parameter{-}Efficient \ Fine{-}Tuning）$：**参数高效微调**——仅微调少量参数，降低训练成本
- $Prefix \ Tuning$：**前缀微调**——冻结主干模型，在每一层注意力层前插入可训练的前缀向量，适配自回归生成任务
- $Prompt \ Tuning$：**提示微调**——冻结主干模型，在输入序列最前端插入可训练的虚拟提示token，适配分类/理解类任务
- $Freeze$：**冻结层**——训练时固定部分层参数不更新，节省显存与计算量
- $Warmup$：**学习率预热**——训练初期逐步升高学习率，避免初期梯度震荡，提升训练稳定性
- $LoRA（Low{-}Rank \ Adaptation）$：**低秩适配**——PEFT主流方法，通过低秩矩阵更新模型权重，公式为 $W = W₀ + B·A$
- $QLoRA（Quantized \ LoRA）$：**量化低秩适配**——LoRA+权重量化（4位NF4），消费级GPU可微调大模型，性能无损
- $FLAN（Fine{-}tuned \ Language \ Net）$：**微调语言网络**——指令微调范式，通过指令调优让模型遵循自然语言指令
- $FP16$：**16位浮点精度**——半精度，显存减半，但动态范围小，易出现溢出/下溢
- $FP32$：**32位浮点精度**——单精度，动态范围大、精度最高，无精度损失，适合小模型训练
- $BFLOAT16（Brain \ Floating \ Point \ 16）$：**16位脑浮点精度**——动态范围和FP32一致，尾数精度降低，混合精度训练常用，兼顾精度与显存
- $NF4（NormalFloat \ 4）$：**4位正态浮点**——QLoRA专用量化格式，适配LLM权重正态分布，精度损失小

## 八、对齐 & 偏好学习

- $RL（Reinforcement \ Learning）$：**强化学习**——通过奖励信号优化模型策略，RLHF第二阶段核心
- $RM（Reward \ Model）$：**奖励模型**——RLHF核心，输出标量分数区分回答优劣，基于成对偏好数据训练
- $RLHF（Reinforcement \ Learning \ from \ Human \ Feedback）$：**人类反馈强化学习**——两阶段（RM+RL）注入人类偏好，训练流程复杂
- $PPO（Proximal \ Policy \ Optimization）$：**近端策略优化**——RLHF核心RL算法，限制更新幅度保稳定，有PPO-Clip和PPO-KL Penalty两个变体
- $GAE（Generalized \ Advantage \ Estimation）$：**广义优势估计**——计算PPO的优势值，平衡偏差与方差
- $DPO（Direct \ Preference \ Optimization）$：**直接偏好优化**——纯有监督学习，替代RLHF，工程化简单，仅需调整超参数β
- $GRPO（Group \ Relative \ Policy \ Optimization）$：**群体相对策略优化**——推理模型核心RL算法，用群体相对优势值提升推理精度
- $DAPO（Distributed \ Advantage \ Policy \ Optimization）$：**分布式优势策略优化**——GRPO改进版，解决长度膨胀问题
- $Dr. GRPO$：**Doctor GRPO**——GRPO改进版，移除样本内token平均项，解决推理输出长度冗余问题
- $BoN（Best \ of \ N）$：**最优N选1**——RLHF简易替代方案，生成N个回答选最优，推理成本高
- $KTO（Knowledge{-}Enhanced \ Preference \ Optimization）$：**知识增强偏好优化**——DPO改进版，融入知识约束，减少模型幻觉
- $IPO（Implicit \ Preference \ Optimization）$：**隐式偏好优化**——无需显式偏好标签，从人类交互数据中挖掘偏好，降低标注成本
- $CPO（Constrained \ Policy \ Optimization）$：**约束策略优化**——在PPO基础上增加约束条件（如KL上限），进一步提升训练稳定性
- $KL（Kullback{-}Leibler \ Divergence）$：**库尔贝克-莱布勒散度**——衡量两个概率分布差异，用于PPO的KL惩罚项
- $Pointwise \ Preference \ Data$：**点态型偏好数据**——对单个样本标注绝对偏好分数/标签，无相对关系，独立评分，标注粒度低
- $Pairwise \ Preference \ Data$：**成对型偏好数据**——比较两个候选结果，标注相对偏好关系（A > B / A < B），RLHF/DPO核心数据，标注成本低、易规模化
- $Listwise \ Preference \ Data$：**列表型偏好数据**——对同一输入的多个候选结果给出完整排序，标注粒度最高，成本最高
- $Preference \ Modeling$：**偏好建模**——对齐任务核心，通过模型学习人类偏好（如安全性、相关性），转化为可优化的目标函数
- $Model \ Alignment$：**模型对齐**——让模型输出符合人类价值观、指令要求，核心目标是“有用、安全、无害”，RLHF/DPO均为对齐主流方法
- $RM \ Calibration$：**奖励模型校准**——修正RM输出分数偏差，避免奖励值失真，提升RL阶段优化效果

## 九、推理 & 提示

- $ICL（In{-}Context \ Learning）$：**上下文学习**——无需微调，通过示例让模型完成任务（零样本/少样本），LLM提示工程核心
- $Context{-}Aware$：**上下文感知**——推理模型的思维控制策略，按上下文自动调整推理深度
- $Inference \ Hallucination$：**推理幻觉**——模型推理时生成不符合事实、逻辑矛盾的内容，需通过CoT/SC/外部工具缓解
- $Multi{-}step \ Reasoning \ Decomposition$：**多步推理分解**——将复杂推理任务拆分为多个简单子任务，分步求解，减少推理误差（CoT/ToT核心思路）
- $CoT（Chain{-}of{-}Thought）$：**思维链**——提示模型分步推理，提升复杂任务（数学/逻辑）性能，推理模型核心范式
- $Zero{-}Shot \ CoT$：**零样本思维链**——无需人工写推理示例，仅用提示词（如“让我们一步步思考”）触发模型推理
- $Few{-}Shot \ CoT$：**少样本思维链**——给1-5个带推理步骤的示例，引导模型模仿分步思考，适配复杂推理任务
- $Scaling \ CoT$：**规模化思维链**——通过预训练/微调/RL让模型主动生成高质量CoT，而非依赖人工提示
- $ToT（Tree{-}of{-}Thought）$：**思维树**——CoT扩展，探索多推理路径，提升推理精度
- $SC（Self{-}Consistency）$：**自一致性**——生成多条推理路径，投票得到最终答案，提升CoT精度
- $Prompt \ Engineering$：**提示工程**——设计最优提示（示例/指令）引导模型输出，无需修改模型权重，提升任务效果
- $Instruction \ Tuning$：**指令微调**——用多样化指令数据微调模型，提升模型对提示指令的理解能力，适配ICL/CoT

## 十、解码 & 推理优化

- $Top{-}p/Nucleus \ Sampling$：**核采样**——选累积概率≥p的token集合采样，平衡多样性与合理性，主流解码策略
- $Temperature$：**温度系数**——调节生成概率分布，控制生成的随机性，T→0退化为贪心解码
- $KV \ Caching$：**键值缓存**——缓存历史token的KV向量，降低推理复杂度，基础推理优化方案
- $Paged \ Attention$：**分页注意力**——拆分KV缓存为显存块，提升显存利用率，降低显存碎片
- $Latent \ Attention$：**潜在注意力**——压缩KV向量，减少显存占用，推理优化方案
- $Speculative \ Decoding$：**推测解码—**—小模型快速生成候选，大模型验证，提升解码速度，无显著性能损失
- $MTP（Multi{-}Token \ Prediction）$：**多Token预测**——一次生成多个token，提升推理效率，k过大会降质量
- $BOS（Begin \ of \ Sequence）$：**序列起始符**——标记文本序列的开头
- $EOS（End \ of \ Sequence）$：**序列终止符**——标记文本序列的结尾
- $Greedy \ Decoding$：**贪心解码**——选概率最高的token，简单快速，缺乏多样性，局部最优
- $Beam \ Search$：**束搜索**——保留k个概率最高的生成路径，选全局最优，流畅度高，计算量随k增加
- $Top{-}k \ Sampling$：**Top-k采样**——仅在概率最高的k个token中采样，约束范围，灵活性低
- $Flash \ Decoding$：**快速解码**——基于分块计算与显存优化，结合Flash Attention优势，进一步提升推理速度
- $Repetition \ Penalty$：**重复惩罚**——对已生成token的概率进行衰减，缓解生成文本重复的问题，解码常用调优项
- $Length \ Penalty$：**长度惩罚**——调节生成序列的长度偏好，避免生成过短/过长文本，束搜索常用

## 十一、并行 & 显存优化

- $ZeRO（Zero \ Redundancy \ Optimization）$：**零冗余优化**——拆分参数/梯度/优化器状态，消除显存冗余
- $TP（Tensor \ Parallelism）$：**张量并行**——拆分模型张量到多GPU，适配大模型
- $PP（Pipeline \ Parallelism）$：**流水线并行**——拆分模型层到多GPU，并行处理不同序列
- $SP/CP（Sequence/Context \ Parallelism）$：**序列/上下文并行**——拆分序列维度，优化长序列处理
- $EP（Expert \ Parallelism）$：**专家并行**——MoE模型专用，将专家拆分到多GPU
- $Data \ Parallelism$：**数据并行**——拆分训练数据，多GPU复制完整模型，简单易实现但显存冗余
- $Model \ Parallelism$：**模型并行**——拆分模型参数/层到多GPU（TP/PP/SP/EP），适配万亿参数模型
- $3D \ Parallelism$：**3D并行**——结合TP+PP+数据并行，三种并行方式协同，适配万亿级及以上参数大模型（如GPT-3）
- $MoE \ Parallelism$：**MoE并行**——融合EP+TP/PP，针对MoE模型的专属并行方案，平衡专家利用率与计算效率
- $HBM（High \ Bandwidth \ Memory）$：**高带宽显存**——GPU显存类型，带宽高，速度快
- $SRAM（Static \ Random{-}Access \ Memory）$：**静态随机存取存储器**——GPU显存类型，速度更快，Flash Attention核心依赖
- $Mixed \ Precision \ Training$：**混合精度训练**——低精度（FP16/BF16）计算，高精度（FP32）保存参数，提升速度并降低显存占用
- $Gradient \ Checkpointing$：**梯度检查点**——训练时仅保存部分层激活值，反向传播时重新计算，以计算量换显存，适配深层大模型
- $Analog \ in{-}memory \ computing$：**类存内计算**——硬件优化方向，专用KV缓存单元+模拟信号计算注意力相似度，大幅降低延迟和能耗

## 十二、RAG & 检索

- $RAG（Retrieval{-}Augmented \ Generation）$：**检索增强生成**——生成前检索外部知识，缓解幻觉，核心步骤为Retrieve → Augment → Generate
- $bi{-}encoder$：**双编码器**——RAG候选检索核心，分别编码Query和Chunk，计算相似度，最大化召回率
- $cross{-}encoder$：**交叉编码器**——RAG重排序核心，联合编码Query和Chunk，提升排序精度，最大化精确率
- $NDCG@k（Normalized \ Discounted \ Cumulative \ Gain \ at \ K）$：**归一化折损累积增益**——RAG检索性能核心指标，综合相关性与排序位置
- $RR@k（Reciprocal \ Rank \ at \ K）$：**倒数排名**——RAG检索指标，衡量最快找到第一个相关Chunk的速度
- $Recall@k$：**召回率**——RAG检索指标，衡量前k个结果中找全相关内容的能力
- $Precision@k$：**精确率**——RAG检索指标，衡量前k个结果中找对相关内容的能力
- $Embedding \ Model$：**嵌入模型**——RAG核心组件，将Query和Chunk转化为低维稠密向量，决定检索相关性，代表模型BGE、Sentence-BERT
- $Vector \ Database$：**向量数据库**——RAG知识库核心存储载体，高效存储、索引Chunk嵌入向量，支持快速相似度检索（如Milvus、Pinecone）
- $Pseudo \ Document$：**伪文档**——RAG检索优化技巧，让查询嵌入与Chunk嵌入更匹配
- $Chunk \ Contextualization$：**Chunk上下文化**——RAG检索优化技巧，补充全局上下文，避免检索偏差
- $Knowledge \ Update$：**知识更新**——RAG长期优化重点，通过增量更新、定期重建知识库，确保检索到的外部知识时效性，避免过时信息误导生成
- $Hybrid \ Search$：**混合检索**——结合向量检索（语义匹配）与关键词检索（字面匹配），兼顾检索召回率与精确率，解决纯向量检索漏检问题
- $Query \ Expansion$：**查询扩展**——对原始Query进行扩充（添加同义词、相关短语），提升检索召回率，缓解Query表述模糊导致的漏检

## 十三、Agent & 工具

- $Agent$：**智能体**——自主决策、调用工具，完成端到端复杂任务的系统，RAG+工具调用的融合升级
- $ReAct（Reason+Act）$：**推理+行动**——智能体核心框架，循环推理-行动-观察
- $MCP（Model \ Context \ Protocol）$：**模型上下文协议**——规范LLM与工具的对接标准，解决互联互通问题
- $A2A（Agent2Agent）$：**智能体间通信协议—**—实现多智能体的标准化协作
- $Tool \ Calling$：**工具调用**——LLM做“大脑”解析参数，工具做“手脚”执行操作，解决LLM无法执行实际操作的问题
- $Router$：**路由模块**——多工具场景核心组件，先解析prompt筛选高相关工具，提升调用准确率
- $AgentSkill$：**智能体技能**——智能体核心能力，含ID、描述、示例，多智能体协作核心组件
- $AgentCard$：**智能体身份卡片**——含名称、地址、技能列表，支持A2A协议通信
- $AgentExecutor$：**智能体执行器**——含execute()/cancel()方法，控制任务执行流程
- $Agent{-}SafetyBench$：**智能体安全评估基准**——衡量智能体安全性能
- $τ{-}bench（Tool{-}Agent{-}User \ Interaction \ Benchmark）$：**工具-智能体-用户交互基准**——智能体专属评估基准
- $ToolBench$：**工具调用评估基准**——包含多样化工具集与任务，用于测试智能体工具调用的准确性与泛化性
- $Agent Memory$：**智能体记忆**——智能体核心组件，分为短期记忆（任务上下文）与长期记忆（历史交互、知识），支撑持续决策与多轮协作
- $Few{-}Shot \ Tool \ Use$：**少样本工具调用**——通过少量工具调用示例，让智能体快速掌握新工具的使用方法，无需额外微调
- $Tool \ Description$：**工具描述**——智能体调用工具的核心依据，明确工具功能、参数要求、输入输出格式，避免调用错误
- $Multi{-}Agent \ Collaboration$：多智能体协作——多个智能体分工协作，各自发挥优势，完成单一智能体无法处理的复杂大型任务

## 十四、评估指标


- $F1$：**F1分数**——精确率与召回率的调和平均，综合评估分类/识别效果
- $PPL（Perplexity）$：**困惑度**——衡量语言模型的预测能力，值越低越好
- $BLEU（Bilingual \ Evaluation \ Understudy）$：**双语评估替换度**——文本生成评估指标，侧重精确率
- $ROUGE（Recall{-}Oriented \ Understudy \ for \ Gisting \ Evaluation）$：——**文本生成评估指标**，侧重召回率
- $METEOR（Metric \ for \ Evaluation \ of \ Translation \ with \ Explicit \ ORdering）$：**双语评估指标**——文本生成评估指标，兼顾语言特征
- $EM（Exact \ Match）$：**精确匹配率**——评估问答/生成任务，模型输出与参考答案完全一致则计为正确，侧重输出准确性
- $MAP（Mean \ Average \ Precision）$：**平均精度均值**——检索任务核心指标，综合衡量检索结果的排序精度与召回率
- $MRR（Mean \ Reciprocal \ Rank）$：**平均倒数排名**——检索/问答任务指标，衡量多个查询中找到第一个相关结果的平均速度
- $WER（Word \ Error \ Rate）$：**词错误率**——衡量语音转写/文本生成的错误程度
- $Pass@k$：**通过率**——推理/智能体指标，k次生成至少1次成功
- $Cons@k$：**一致性准确率**——推理/智能体指标，多数投票准确率
- $\text{Pass\^ k}$：**全通过率**——推理/智能体指标，k次生成全部成功
- $ASR（Attack \ Success \ Rate）$：**攻击成功率**——智能体安全评估指标，值越低越安全
- $LaaJ（LLM{-}as{-}a{-}Judge）$：**LLM评估器**——用LLM作为法官，评估模型输出质量，分Pointwise和Pairwise两种变体
- $κ（Kappa）$：**卡帕系数**——量化评估者共识程度，剔除随机一致性，越接近1一致性越强，有Cohen's Kappa、Fleiss' Kappa等变体
- $MMLU（Massive \ Multitask \ Language \ Understanding）$：**大规模多任务语言理解**——评估通用知识
- $AIME（American \ Invitational \ Mathematics \ Examination）$：**美国数学邀请考试**——评估高阶数学推理能力
- $PIQA（Physical \ Interaction \ : \ Question  \ Answering）$：**物理常识问答**——评估常识推理能力
- $GSM8K（Grade \ School \ Math \ 8K）$：**小学数学应用题**——评估数学推理能力
- $SWE-bench（SoftWare \ Engineering \ benchmark）$：**软件工程基准**——基于真实GitHub Python问题的代码生成评估基准
- $HarmBench$：**有害行为基准**——评估模型安全性能
- $FLOPs（Floating{-}point \ Operations）$：浮点运算总次数——衡量模型计算量
- $FLOPS/FLOP/s（Floating{-}point \ Operations \ per \ Second）$：每秒浮点运算次数——衡量硬件性能
- $Factuality \ Quantitative \ Scoring$：**事实性量化评分**——解决幻觉问题，将输出拆分为独立事实，加权求和得到评分
- $Pareto \ Frontier$：**帕累托前沿**——优化权衡关系的最优解集合，为LLM落地选型提供参考
- $TOPSIS（Technique \ for \ Order \ Preference \ by \ Similarity \ to \ an \ Ideal \ Solution）$：**逼近理想解排序法**——优劣解距离法，多指标综合评估方法，用于LLM多维度选型（兼顾性能、速度、成本等）
- $Recursion \ Curse$：**递归诅咒**——LLM生成数据训练新模型导致的性能退化、幻觉增多问题，需控制生成数据比例
- $Data \ Contamination$：**数据污染**——基准测试题目出现在训练集导致评估失真，需通过哈希标记、盲测等解决
- $Goodhart's \ Law$：**古德哈特定律**——指标成目标后模型“刷分”但实际能力脱节，需多维度评估避免

## 十五、多模态 & 扩散

- $MDM（Masked \ Diffusion \ Model）$：**掩码扩散模型**——文本扩散核心，用掩码替代高斯噪声，支持推理并行
- $ViT（Vision \ Transformer）$：**视觉Transformer**——将图像切块“文本化”，用Transformer处理视觉任务，多模态基础组件
- $VLM（Vision \ Language \ Model）$：**视觉语言模型**——融合视觉与文本能力，可处理图文任务，主流两种架构（LLM解码器复用型、交叉注意力融合型）
- $DiT（Diffusion \ Transformer）$：**扩散Transformer**——图像扩散核心，用Transformer实现扩散模型
- $ARM（AutoRegressive \ Modeling）$：**自回归建模**——传统LLM生成范式，逐Token生成，推理无法并行
- $Diffusion \ LLMs$：**扩散大语言模型**——基于MDM，推理可并行，降低长文本生成延迟，效果稍低于最优ARM模型
- $Video \ Modal \ Adaptation$：**视频模态适配**——帧级切块 + 时空位置嵌入，增加时间步位置信息，适配Transformer
- $Audio \ Modal \ Adaptation$：**语音模态适配**——Conformer块（Transformer+CNN）编码，跨注意力融合LLM，适配语音时序特征
- $Code \ Modal \ Adaptation$：**代码模态适配**——代码语法结构转Token，自注意力捕捉上下文逻辑，适配代码生成任务
