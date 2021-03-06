  # Paper Notes

  > @author:  汤远航 (Yuanhang Tang)
  >
  > @e-mail: yuanhangtangle@gmail.com

  ## A Survey on Dialogue Systems:Recent Advances and New Frontiers

  > @datetime: 2021/04/03
  >
  > [weblink](https://arxiv.org/abs/1711.01731) or [click for local file](./A_Survey_on_Dialogue_Systems_Recent_Advances_and_New_Frontiers.pdf)

  This paper gives an overview to dialogue systems. More specifically, this paper categorizes DS into two main kinds: task-oriented system and non-task-oriented system. See the graph below for more details:

  ```mermaid
  graph LR
    ds(Dialogue System) --> to(task-oriented)
    to --> pm(pipeline method)
    pm --> nlu(natural Language understanding)
    pm --> dst(dialogue state tracker)
    pm --> dpl(dialogue policy learning)
    pm --> nlg(natural language generation)
    to --> to_eem(end-to-end method)
    to_eem --> eem_des(single module mapping input<br> history to system output)
    ds --> nto(none-task-oriented)
  ```

  ### Task-Oriented System

  Assist the user to complete some task

  - **NLU** (**Natural language understanding**): 
    extract information from  user utterance.
    - Maps the user utterance into **predefined slots**. A slot is a concept ID, e.g. location, date time, etc. **Slot filling** assigns each input word with a predefined semantic label. It takes a sentence as input and output a sequence of slots. one for each word in the input sentence. This can be modeled as a **sequence labeling problem** similar to the POS tagging problem.
    - Classifies the user intent into **predefined intent** or detect dialogue **domain**. This is simply a classification problem. Statistical method or deep learning method can be applied to address this issue.
    - The semantic representation generated in this step is passed to the next step.

  - **Dialogue state tracking**: 
    The user's goal may change during the conversation, and this can be modeled as a **state struture** similar to the one adopted in NAOGOLF. According to semantic representation generated in the last step, a state management system (**rule-based system**, **statistic dialog system** or **deep learning based dialog model**) mantains a state struture and categorizes the current situation into one of the predefined states (commonly called **semantic frame** or **slot**) by combining the previous system acts, dialogue history and previous dialogue states. This is the core component to ensure a robust manner.
    - **rule-based system**: Like what we do in NAOGOLF;
    - **statistical method**: Mantains a distribution over predefined slots, and output the most likely one;
    - **deep learning based methodd**: Just train it;

  - **Policy learning**:
    Conidtioned on the state representation, policy learning component generates the next system action. **Rule-based method**, **supervised learning method** or **reinforcement learning method** can be adopted to choose one of **predefined action** based on current state. This module may interates with a external database to generate meaningful action.

  - **NLG** (**natural language generation**):
    Conditioned on the chosen action, state and other representation from the upstream modules, NLG module converts this semantic symbols into natura language and present the result to the user. 
    - Conventional method typically adopt a **sentence planning system** which first convert the semantic symbols to intermediary form such as template or tree-like structures and then convert them into final response.
    - Deep learning based model: Just train it.

  - **End-to-End Method**:
    End-to-end model uses a single module to map the dialogue history to the final response.

  - **Shortcomings**:
    A task-oriented dialogue system aims at asssiting the use to complete some task, thus it is usually related to some certain domain, e.g. shopping, custom service, etc. A conventional rule-based dialogue system is usually specialized with **interdependent components**, and **does not promise portability**. **Significant human effort**, including **data collection**, labeling, rule design, template design and so on, must be devoted to create such a system. Moreover, it is **hard to design update method** for such systems since operations like **querying external database** are **non-differentiable**, and the user's feedback is hard to be **propagated to upstream modules**.

  - **Techniques**: 
    - **rule design**: 
      - Like what we do in NAOGOLF
    - **statistical methods**:
      - Output the probability of each slot for each input word
      - Output the probability for each predefined intent or domain
    - **supervised learning methods**:
      - End-to-End model: replace non-differentiable operations with differentiable layers
      - Classification model
    - **reinforcement learning methods**:
      - Model the dialogue system as an intelligent agent interacting with the user
    - **Generalization and Specialization**:
      - Generalization for portability, specialization for good performance
      - *Pre-train and fine-tune*

  ### Non-Task-Oriented System
  Take with the user on open domains; Chatting robot such as Xiaoai, Siri, etc

  - **Neural Generative Models**: sequence-to-sequence models
    - capture dialogue context
    - increase response diversity, reduce meaningless responses:
      - modify decoder
      - design better objective function
      - introduce **latent variables** to mantain a distribution
      - model dialogue topic and the user's personality
      - query knowledge database
    - learning throught interaction

  - **Retrival-based Methods**:
    - choose a response from predefined responses
    - a repsonse match problem: single turn or multi-turn

  - **evaluation**: hard to automatically evaluate; some criteria
    - forward-looking
    - informative
    - coherent
    - interactint


  ## An introduction to ROC analysis
  >@datetime: 2021/04/17
  >
  > [weblink](https://www.researchgate.net/publication/222511520_Introduction_to_ROC_analysis/link/5ac7844ca6fdcc8bfc7fa47e/download) or [click for local file](./ROCintro.pdf)
  > 
  > 我决定还是写中文, 高效方便. 这篇文章只简单看了ROC曲线和AUC的含义. 

  - **基本指标**: 
    
    ROC曲线纵轴为`TPR`, 横轴为`FPR`. 这两个指标都以预测正样本为研究对象, 这种思维其实默认了正样本具有更高的重要性. 具体的: 

$$
  TPR = \frac{TP}{P} = \frac{TP}{TP + FN}\\

  FPR = \frac{FP}{N} = \frac{FP}{FP + TN}
$$

  - **直观认识**: 
    
    `TPR` 衡量准确的正样本预测, 描述模型是否能够正确的预测正样本; `FPR`衡量错误的正样本预测, 描述模型是否会错误的将负样本预测为正样本. 两个指标综合描述了模型对正样本的预测的合理程度. 理想地, `TPR`应该尽可能高, `FPR`应该尽可能低. 模型越倾向于预测正样本, 则`TPR`越可能大, 但是`FPR`也会随之变大, 故而ROC曲线右上角的点较为"激进", 而左下角的点较为"保守"

  - **基准**: 
    
    在ROC曲线图上, 位于`y = x`左上角的点满足`TPR > FPR`, 位于右下角上的点通过反转预测标记可以映射到左上角. 位于对角线上的点可以理解为随机猜测, 正负样本的预测正样本都是是对错参半. 从这个理解可以看出, ROC曲线的基准为对角线, 也就是, **任何一个样本都以1/2的概率预测为正样本**. 

  - **绘制方法**: 
    
    对于输出置信度的模型, ROC曲线的绘制通过改变阈值进行. 具体的, 根据样本置信度从高到低排列. 开始将阈值设定为最高使得所有样本都被预测为负样本, 每次往后新增一个正样本, 并在ROC图上, 绘制直到所有样本都被预测为正样本. 这样得到的ROC曲线可以理解为对模型实际ROC曲线的近似. 

  - **AUC**: 
    
    绘制的ROC曲线下方的面积. 如果有样本置信度从高到低(各不相同)的序列 $x_1, x_2, ..., x_m$, 对应的真实标记序列 $y_1, y_2, ..., y_m$, 正样本数量 $P$, 负样本数量 $N$ 可以推导出:
    $$
      AUC = \frac{\sum_{i<j} (\mathbb{1}\{ y_i > y_j \} + 0.5 * \mathbb{1}\{ y_i = y_j \}) }{PN}	
    $$
    这其实就是: 从序列中任取一个正样本和一个负样本, 模型会给予正样本更高置信度的概率. 这个概率其实就是对**模型认为正样本比负样本更像正样本的概率**的估计.

  - **进一步讨论**: 
    
    `TPR`和`FPR`其实都是以真实标记的统计为分母, 故这两个指标其实都是反映了预测正样本符合真实标记的程度. 而精准率`PREC`定义为:
    $$
    PREC = \frac{TP}{TP + FP}
    $$
    即正确的预测正样本占预测正样本的概率. 这个指标则是反映预测正样本用于进一步分析处理的可靠度. 将`PREC`和`TPR(REC)`结合起来则可以绘制`PRC`, 得到另一个图. ROC曲线的基准是1/2随机分配, 对于不均衡样本而言, 这个基准显然并不合理, 因为我们显然可以以更高的概率预测占比较高的标记. `PRC`曲线则针对此进行了的修正. 

  - **分布意义下的ROC以及AUC**: 
    
    假定一个模型输出给定样本是正样本的概率, 那么这个概率本身具有一定的分布. 单个负样本对应的模型输出概率形成**负样本模型概率分布**, 单个正样本对应概率分布形成**正样本模型概率分布**. 在给定一个阈值的情况下, 我们将模型输出概率高于这个阈值的样本分为正样本, 低于这个阈值的样本分为负样本, 这种分类方法的一个假设是, 正样本模型概率应该普遍比负样本模型概率大; 直观的, 正样本模型概率的概率密度分布图应该比负样本模型概率的概率密度图更靠近 $x$ 轴的右侧, 而且重叠部分尽可能的小. 理论上的ROC曲线是通过这个分布计算出来的. 具体的, 假设正样本模型概率的概率密度函数为$f_{+}(p)$, 负样本模型概率的概率密度函数为$f_{-}(p)$, 联合密度分布为$f(p_+, p_-)$, 给定阈值$T$, 有:
    $$
    TPR = \int_{t > T} f_{+}(t) dt\\
    FPR = \int_{t > T} f_{-}(t) dt\\
    $$
    自然的, $T$ 越小, 两个数值都越大, ROC曲线从左到右反映了$T$变化的过程. 而$AUC$则定义为:
    $$
    AUC = \int_{p_+ > p_-} f(p_+, p_-) dp
    $$
    即正样本相比于负样本具有更高的模型输出概率的概率.

  - **分布决定了ROC**: 
    
    从上述讨论可以看出, ROC曲线是完全由单个样本的模型概率分布决定的, 而单个样本的模型概率分布则是总体的分布以及模型自身的特点决定的. 在实际的计算中, 假设正样本总体的分布, 以及负样本总体的分布保持不变, 模型不变, 在基数较大的情况下, 测试集中正负样本的采样比例不会对ROC和AUC产生太大的影响. 然而, 在样本极不均衡的情况下(往往是我们不太关心的负样本有相当高的占比), 我们往往会希望对`FPR`进行更严格的控制以减少`FP`总量的绝对数值(尽管相对比例保持不变). 因为实际应用中, 我们往往是对预测正样本(`TP+FP`)进行处理, 这时占比太高的`FP`将造成较大的资源损耗. 这时, ROC 和 AUROC 无法对模型的性能给出较好的刻画. 这就是下一篇[文章](#the-precision-recall-plot-is-more-informative-than-the-roc-plot-when-evaluating-binary-classifiers-on-imbalanced-datasets)给的分析

  - **ROC不反映类别不均衡的直观理解**: 
    
    考虑实际的计算过程. 给定一个阈值, 假设负样本的总数按一定的比例升高, 在分布不变的情况下, 置信度序列各处的负样本总数也将大致按比升高, 改变阈值过程中, 负样本出现的次数大致按比升高, 但是由于基数也按比增大, 增加的`FPR`没有太大变化. 最显著的变化是, 曲线变得相对光滑了一些. 这个结果可以从下一篇[文章](#the-precision-recall-plot-is-more-informative-than-the-roc-plot-when-evaluating-binary-classifiers-on-imbalanced-datasets)给的实验中看出. 

----------------------------------------

  ## The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets
  > @datetime: 2021/4/19
  > 
  > [weblink](https://www.researchgate.net/publication/273155496_The_Precision-Recall_Plot_Is_More_Informative_than_the_ROC_Plot_When_Evaluating_Binary_Classifiers_on_Imbalanced_Datasets) or  [click for local file](./PRC_imbalanced_dataset.pdf)
  > 
  > 主要做了很长的说明和实验, 来验证在样本极端不均衡的情况下, PRC 提供了比 ROC 更为准确有效的直观评估

  - **一些指标的理解**:
    
    - `TPR, REC`: 正样本中, 预测正确的数量. 随着阈值的减小而升高.
    - `PREC`: 预测正样本中, 预测正确的数量. 
    - `ACC`: 所有样本中, 预测正确的数量
    - `FPR`: 负样本中, 预测`错误`的数量. 随着阈值的减小而升高.

  - **评估模型的准则**:

    在考虑评估指标的时候, 我们需要考虑这些指标为什么有效, 即一个模型好坏的根本评判标准是什么. 在不同的场合, 这个评判的标准是不同的. 假设我们研究的问题有这样的特性:

    1. 正样本是我们关心的样本, 负样本则是不太关心的样本. 此时, 负样本可以理解为 "不是正样本的那些样本";
    2. 选择了一定的阈值后, 我们将会处理所有的预测正样本;
    3. 我们希望使用尽可能少的资源处理尽可能多的正样本.

    `2`和`3`表现出了一定的矛盾性. 我们想处理正样本, 但实际上我们只能处理预测正样本. 为了实现`3`所言"尽可能少的资源", 我们希望资源尽可能的少浪费在`FP`上, 因而希望`PREC`更高; 为了实现`3`所言"尽可能多的正样本", 我们希望`TPR = REC`更高. 这就使我们关心`PREC-RECALL`之间的矛盾和平衡.

  - discussion  
    
    - 翻转正负样本对AUC没有影响, 会对称的映射ROC, 又会如何影响PRC?
      - 一般而言
    - ROC在均衡情况下为什么有优势? 
      - 好算, 有固定的 baseline, 一定程度上和PRC相互制约.
    - AUPRC的含义应该如何理解？
      - 就是某种平均的估计, 数学上没有太多含义
    - 从分布的角度上将讲, 另一种直观的方法是, 将正样本和负样本的输出概率进行 KDE 后画出来看看
      - 很多时候并不总是有足够的区分度

  ## A BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS
  > @datetime: 2021/04/20
  > 
  > [weblink](https://arxiv.org/pdf/1610.02136) or [click for local file](./A_BASELINE_FOR_DETECTING_MISCLASSIFIED_AND_OUT-OF-DISTRIBUTION_EXAMPLES_IN_NEURAL_NETWORKS.pdf)
  > 
  > 这篇文章写了老长的实验来说明, 对于 **Softmax** 输出的神经网络, 类别概率可以用作区分`F`(误分类样本)和`T`(正确分类样本), `IN`(in-distribution)和`OUT`(OUT-OF-DISTRIBUTION)的一个有效指标, 同时通过设计网络结构和训练方法说明, 有更好的方法可以用来做误分类和离群点的识别. 

  - **实验思路**:

    作者通过各个领域的分类实验验证了, softmax型的输出会倾向于给正确样本更高的类别概率(即softmax输出概率的最大值), 使得正确分类的样本和误分类的样本的类别概率的分布出现差异. 通过前面两篇文章, 我们看出, 这种差异可以通过 AUROC 和 AUPRC 进行评估. 实验结果表明, 尽管误分类样本可能也有较高的类别概率, 但总体分布比正确分类的样本的类别概率小. 这个实验说明, **对于 softmax 型的输出, 其实际上并非一个具有一般意义的概率, 而只是通过软映射得到的一个概率, 其绝对数值的大小含义有限.** 在这个意义上, 下面将要讨论的基于 **能量** 的 OOD 模型则有了更好的处理方法. 

  - **指标设计**:

    作者提出的 `AUROC` 以及 `AUPRC` 已经成为一个标准, 后来又开始使用`FPR95`, 即`TPR = 95`时, `FPR`的取值.

  - **辅助编码器的设计**:

    没怎么看明白

  - **AD**: 
    - 基于已有的softmax分类器, 或者对训练集进行训练后的分类器, 没有其他需要调节的参数或者训练过程, 相当的简单明了.
    - 作者通过各个领域的分类实验验证了该方法的普适性.
    - 天然的适用于各种 softmax 单标签分类场景, 已经成为一个baseline方法.

  - **DISAD**:
    - softmax 有天然的单标签特征, 类别之间相互排斥, 可能较难用于多标签的场景. 但是总可以试试看的对吧?
    - 实验表明 softmax 可能对OOD样本赋予很高的信度
    - 

  - **DISCUSSION**
    
    - 多标签相比于单标签, OOD 的潜在含义从 "错误分类为已知的一类", 转变为 "遗漏了可能的标签" 和 "出现了新的意图". 
      - 能否面向海量标签, 或者标签组, 单独考虑每个类别的置信度. 
      - 多分类问题的标签遗漏, 是容易遗漏相似的标签还是遗漏不同的标签?
      - 逻辑上, 是否相关联的标签可以给予更多的关注?
      - 分类出的标签和标签之间的关联和排斥, 是否给 OOD 的一定的暗示?
    
    - 对单标签和多标签的差异还得找点儿文章看看
    
    - 作者提出的辅助模型究竟是个啥子东西?
    
    - 模型似乎区分出了两个概念, OOD和错分类, 两者的含义虽然不同, 但是有着相似的影响

  ## A SIMPLE UNIFIED FRAMEWORK FOR DETECTING OUT-OF-DISTRIBUTION SAMPLES AND ADVERSARIAL ATTACKS

  > @datetime: 2021/05/06
  > 
  > [click for local file](./A_Simple_Unified_Framework_for_Detecting_Out-of-Distribution_Samples_and_Adversarial_Attacks.pdf)
  > 
  > 这篇文章主要提出了一个基于预训练模型的生成式方法, 通过在神经网络的特征空间进行 GDA 获得类别条件分布, 再根据 Gaussian 得到 Mahalanobis 距离作为置信度进行 OOD 判别. 作者同时还提出了 扰动 和 特征集成 两种方法来提高模型的能力, 并针对 类别增量学习 场景提出了简单的适应方法.

  - **IDEA**:
    
    - softmax作为一个软映射分类器, 本身其实是不具备充分的概率含义的. 由 softmax 得到的概率分布可能具有一定的误导性;
    
    - softmax作为交叉熵的一个预处理, 给出了一个归一化的方法, 从而使得特征的学习成为可能. 从这个视角出发, 将神经网络理解为一个从输入空间向特征空间的映射, softmax 回归则成为一个线性分类器. 因此, 作者并不选用最后一层的输出进行建模, 而是针对倒数第二层, 即特征空间进行了概率建模;
    
    - 建模的思路是多类别的 GDA. 这个思路的合理性在于, 在各类别方差一致和均匀分布的假设下, GDA 得到的分类结果和 softmax 有形式上的等价性. 有趣的是, 通过 softmax 训练出来的特征空间进行 GDA 建模后有可能获得比 softmax 精度更高的分类器. 
    
    - 作者利用 Gaussian 分布的指数特征提出了最近类别的 Mahalanobis 距离作为信度指标.
    
    - 将神经网络每一层的输出都看作一定水平的特征, 对每一层都计算Mahalanobis距离, 并借助验证集训练logistic回归线性集成各层的结果, 得到了效果更好的判别器. 
    
    - 通过扰动进行加强. 作者给每一层的每个分量进行了相同程度的扰动, 但方向是直接由该层的Mahalanobis距离对输入的梯度方向决定的. 每一层的Mahalanobis距离对输入的梯度都需要计算. 
    
    - GDA 建模只需要使用各个类别的样本均值和所有样本的方差, 这些计算相对于样本类别而言都是线性的, 因此 GDA 天然的适合类别增量学习; 扰动幅度参数和样本类别无关; 作者假设了预训练的模型具有足够的特征编码能力, 因此不需要使用新的样本进行调整. 综上, 该方法天然的适用于类别增量学习.
    
    - 作者的实验表明, 这个方法不仅对OOD有效, 对攻击样本也有良好的分辨能力.

  - **参数**:
    - 扰动的幅度
    - 集成的隐藏层数量
    
  - **AD**:
    - 基本的idea相当的简单, 实现起来相当的方便. 无论是均值, 方差还是扰动还是logistic集成, 实现都很方便.
    - 天然的适合类别增量学习, 对攻击样本也有一定的分辨能力.
    - 对小样本和样本噪声有很强的适应力.

  - **DISAD**:
    - 每层都需要进行扰动处理, 都需要求导, 都需要使用扰动样本重新计算. 这或许会导致较大的计算开销.
    
  - **DISCUSSION**:
    
    - softmax 得到的特征空间通过其他形式简单的分类建模, 是否常常可能获得更好的分类结果?
    
    - 综合每个类别的信度, 能够集成出更强的模型? 
    
    - 如果最后一层不再使用 ReLU, 是否有可能获得更好的结果? ReLU 在理论上有可能得到任意信度的结果, 那是否能考虑在训练或者推断时对过大的输出进行抑制? 能不能通过激活函数或者分类器将远离边界的样本的输出值抑制? 或者通过施加较强的正则化控制参数大小?
    
    - 对扰动和概率密度分布的讨论:
      - 扰动是否使得概率密度高的部分受到更大的影响?
      - 扰动总是往信度增大的方向实施, 背后有什么特别的含义?

    - 作者提出的 GDA 和 softmax 的关系其实没有理论上的说服力.
    
    - Mahalanobis 距离由均值和度量矩阵决定. 这个视角下, 模型其实是原型方法的一个例子, 并预先通过Gaussian分布假设了距离的形式. 或许可以考虑度量学习的思路, 将度量矩阵当做一个学习参数使用, 距离的起点或许可以考虑某种密度中心, 或者说原型. 考虑直接结合VQ和度量学习? 会不会训练出零矩阵?


  ## LIKELIHOOD RATIOS FOR OUT-OF-DISTRIBUTION DETECTION
  > [click for local file](./Likelihood_Ratios_for_Out-of-Distribution_Detection.pdf)
  > 
  > 这篇文章提出 **主体-背景似然比** 作为区分 OOD 样本的信度指标. 其想法是, 样本包含着两部分的信息, 一是有关样本类别的语义信息, 二是和样本所属范畴相关的领域背景信息. 生成式模型有可能更多的关注了领域背景信息, 并构建了和背景信息相关的模型对样本进行了分类. 当符合这个模式的 OOD 样本出现时, 模型输出较高的信度. 基于此, 作者提出, 通过训练一个背景信息模型, 将原模型训练出来的背景信息去除, 从而获得和类别相关的语义信息, 并且以似然比的形式表示为信度指标.

  - **IDEA**:
    - 作者认为, 样本的特征可以分解为两个部分, 一是有关样本类别的语义信息, 二是和样本所属范畴相关的领域背景信息. 模型给 OOD 样本输出更高的信度, 是因为模型没有正确的提取语义信息的特征, 而是错误的提取了背景信息.
    
    - 作者用了两个例子说明. 一个 LSTM 训练出来的细菌基因似然, 出现了似然和GC碱基占比的相关性; 另一个是 PixelCNN 训练出来的 Fashion-MNIST 图像像素似然, 模型赋予非笔画的空白像素更高的似然. 
    
    - 作者认为, 如果能够去除背景信息的影响, 就能够对此进行改善. 为此, 作者通过按一定概率随机扰动的方式处理输入样本, 用相同的结构训练一个背景似然. 通过对数似然相减 (也就是似然比) 获取语义似然.
    
    - 在训练 background model 时引入 $L_2$ 正则能够提升模型效果.
    
  - **参数**:
    - 扰动概率, 作者推荐`[0.1, 0.2]`
    - $L_2$ 正则的系数
    
  - **AD**:
    - 实验结果上, 效果不错
    
  - **DISAD**:
    - 额外训练一个模型, 计算资源消耗较大
    
  - **DISCUSSION**:
    
    - 扰动是否使得概率密度高的部分受到更大的影响?
      - Google这篇文章使用的平均扰动其实就是给输入样本叠加了一个均匀分布, 这相当于对输入的概率密度函数在输入空间中进行了离散卷积操作, 即平均操作. 对于连续函数空间, 可以通过均匀分布得到类似的扰动. 
    
      - 假设样本表现为 `峰值` 分布特征, 即 `IN` 样本集中于高密度区域, 而低密度区域则是 `OUT` 样本. 比如Gaussian分布就是典型的单峰分布. 那么, 在合适的扰动幅度下, 密度分布会被 "压平". 高密度区域由于变为附近较低密度的平均, 因而密度变小; 而高密度区域附近的较低密度区域在平均过程中收益于高密度区域, 密度有所提升.
    
      - 如果扰动过大, 一个峰可能会被分割成两个; 如果扰动过小, 峰度不会有明显的变化. 综上所述, 影响因素主要就是扰动和峰度的相对大小.
    
      - 考虑到这个`相对大小`, 能不能定义某个比例处理以避免调参.
    
      - 能否设计网络模型, 将原始分布训练成合理的正态分布? 或者有一定峰度的分布? 
    
    - 另一种扰动形式是将样本往信度增大的方向扰动, 从上面这个框架来说, 这就是在加强样本分布的峰度. 

    - 深度模型总是更关注总体特征吗??


  ## ENERGY-BASED OUT-OF-DISTRIBUTION DETECTION
  > [weblink](https://arxiv.org/pdf/2010.03759.pdf) or [click for local file](./Energy-based_Out-of-distribution_Detection.pdf)
  > 
  > 这篇文章将神经网络视为能量模型, 将softmax置信度理解为有偏差的能量估计方法, 提出以能量作为分类标准, 并提出了一个基于能量的正则方法. 

  - **IDEA**:
    - 能量模型来自于热力学, 下面使用的很多公式有热力学背景.
    
    - 不同于常见的将多分类的神经网络理解为 `特征映射 + softmax回归(线性组合之后进行软映射)` 的视角, 本文作者将多分类神经网络理解为`能量映射 + 软映射`. 通常的 `编码器 + MLP + 软映射之前的线性组合` 被理解为 `能量映射`(实际上取了负号), 即模型将样本映射到一个各个类别对应的能量值.

    -  能量是样本和标签的联合函数, 可以理解为未归一化的概率密度的负值. 能量反映了样本在该类别下的稳定性, 能量越高越不稳定. 概率密度和能量的转换通过热力学方程进行
      $$
        p(x,y) \propto e^{-E(x, y)/T}\\
      
        p(x) \propto \int_{y} e^{-E(x, y)/T}\\

        p(y|x) = \frac{p(x, y)}{p(x)} = \frac{ e^{-E(x, y)/T} }{ \int_{y} e^{-E(x, y)/T} }
      $$

    - 从热力学背景出发, 样本的 `Helmholtz Free Energy` (亥姆霍茲自由能) $E(x)$ 满足:
      $$
        e^{-E(x)/T} = \int_{y} e^{-E(x, y)/T}\\
        E(x) = - T \times \log{\int_{y} e^{-E(x, y)/T}}
      $$

    - 对于神经网络而言, 模型最后一层的输出是各个类别的置信度$f_y(x)$, 信度越高, 可能性越大, 这和能量的特性刚好相反. 因此定义:
      $$
      E(x, y; f) = - f_y(x)\\
      E(x; f) = - T \times \log{\int_{y} e^{- E(x, y)/T}} = - T \times \log{ \sum_{y \in C} e^{f_y(x)/T} }
      $$
      积分被自然的替换为求和. 如果 $T = 1$:
      $$
      - E(x; f) = \log{ \sum_{y \in C} e^{f_y(x)} }
      $$
      那么, $- E(x; f)$ 就是模型 $f$ 下, $x$ 具有的能量的负值, 其值越大, 说明 $x$ 在该模型下越稳定, 越可能是 `IN`. 事实上,
      $$
      p(x) \propto \int_{y} e^{-E(x, y; f)/T} = e^{-E(x; f)/T}\\
      $$
      故 $p(x)$ 和 $-E(x)$ 正相关. 我们也可以利用概率密度的归一化得到概率密度的具体形式:
      $$
      p(x) = \frac{e^{-E(x; f)/T}}{ \int_{x \in X} e^{-E(x; f)/T}}
      $$
      定义能量模型下的总测度: 
      $$
      e^{-E(f)} = \int_{x \in X} e^{-E(x; f)/T}
      $$
      注意到这个测度是由模型决定的, 和 x 无关, 因此在信度的计算中可以忽略. 那么就有:
      $$
      p(x) = \frac{ e^{-E(x; f)/T} }{ e^{-E(f)/T} }
      $$

    - 在 softmax 框架下, 用于 OOD 检测的信度是类别概率. 我们取对数值:
      $$
      \log{p_{max}(x)} = \log{\frac{f_{max}(x)}{\sum_{y \in C} e^{f_y(x)}}} = -E(x; f) + \log{f_{max}(x)}
      $$
      可以看出, 最大 softmax 值取对数后是对样本能量的一个有偏估计. 这解释了作为 baseline 的 MSP 方法的局限性. 

    - 实验表明, 使用能量值有助于提升 baseline 方法的性能.
    
    - 能量值是基于一个预训练模型的, 即在给定了一个映射的情况下, 计算样本能量的一种方法. 作者另外提出了一个微调方法, 在给定一定的 OOD 样本的条件下, 通过 `平方hinge损失` 显式地将能量值引入误差中, 惩罚 `IN` 样本太大的能量值和 `OUT` 样本太小的能量值. 具体的, 给定一个 `IN` 样本的基准$m_{in}$, 惩罚超过这个基准的 `IN` 能量$max(0, E(x_{in}) - m_{in})$. 作者测试后选择了平方损失, 有利于得到稳定的训练结果. 故而针对 `IN` 样本的损失就是:
      $$
      \sum_{i = 1}^m (\max{(0, E(x_i) - m_{in})})^2
      $$
      类似的, `OUT` 样本的损失就是:
      $$
      \sum_{i = 1}^{m'} (\max{(0, m_{out} - E(x_i) )})^2
      $$
      加上系数作为超参数, 就有:
      $$
      CrossEntropy + \lambda \left( \sum_{i = 1}^m (\max{(0, E(x_i) - m_{in})})^2 + \sum_{i = 1}^{m'} (\max{(0, m_{out} - E(x_i) )})^2 \right)
      $$
      交叉熵用于保持训练精度, 能量损失用于判定 OOD. 实验上, 微调能够在不损失太大精度的情况下, 显著提高 OOD 判别性能. 

- **参数**:
  - 温度参数: 作者建议直接设为 1
  - 能量基准: 作者建议分别在预训练模型计算出的`IN`和`OUT`样本的能量均值附近选取调参
  - 系数$\lambda$: 调参吧
  
- **AD**:
  
- 简单好实现, 计算开销也比较小
  
- **DISCUSSION**:
  - 平方hinge损失是不是可以进行修正? 作者谈及的训练过程的稳定性是出于什么原因? 
  - 在多标签的意义下, 我们又该如何定义能量? 类别与类别之间已经不再有确定的间隔, 是否可以选取最高层的类别作为单独一类进行能量计算? 

  ## Discussion: OOD 检测的一些 IDEA

  - MSP 会赋予错分类和 OOD 样本以高信度, 但具体的, 这种信度的分布情况是否存在差异? 前面讨论的峰度和扰动对`IN`和`OUT`存在的影响能够成为一个有力的解释和判定方法?
    
  - 对于信度高得过分的样本, 应该理解为预训练模型自身的问题? 在OOD检测中, 预训练模型是否足够可信成为一个需要考虑的因素.

  - Google 的所谓 background 模型是否能够在统计分布上得到较好的解释?

  - 辅助编码器的设计, 能量边界的惩罚都是对模型的调整; Mahalanobis距离和softmax都是直接应用预训练模型; LLR则直接训了一个新模型;
  
  - 随意的扰动表现出密度的摊平, 梯度方向的扰动表现为峰度的提升;
  
  - 对于监督学习模型, 信度常常选取类别信度的最大者. 这背后似乎暗含着一种假设, 输入根据不同的类别在特征空间中聚集成为簇, 接近这些簇的某一个的样本有更高的概率出现, 和这些簇都远离的样本则是OOD的. 
  
  - LLR文章提出的模型的拟合偏差揭示了一些矛盾: 
    - 模型获取的特征能够满足设计形式, 却无法满足我们的设计意图.
  
    - softmax 可以进行高精度的拟合, 却可能拟合一些次要的特征. 
    
    - 单标签的分类模型总是对输入空间进行划分, 但是实际上输入空间远远大于分类任务的数据范围. 超出这些范围样本也看被赋予很大的信度. 这些样本可能处于离分类边界较远的区域. 比如, SVM等线性分类器会赋予更远的样本以更高的信度. 
    
    - softmax数值不够大的样本可能是位于分类边界处的样本, 而模型可能确信落入某个类别的, 离边界较远的样本正是这一类样本;
      - 有相关工作尝试扩大特征映射后的样本类别边界, 结果得到提升
      
    - 也就是说, softmax数值本身其实是一个相对数值. softmax并不给出样本落入这个类别的概率是多少, 而是给出了`相对于其他类别而言, 样本落入这个类别的、相对的概率`. 对于不属于给定类别的样本, softmax隐含的相对性成为具有误导性的指标. 
  
    - 我们能否修改网络损失或者网络结构, 激活函数等等, 对远离边界的样本进行抑制?

  - 拟合的范围: 


  ## Discussion: 多标签下的OOD检测

  - OOD 应该如何界定?
    - 遗漏了可能的标签
    - 存在标签组以外的标签
    - 错误的标记了某个标签

  ### 单标签模型如何迁移:
- 和类别无关的生成式方法: 直接套用试试看
- 和标签相关的方法:
  - 使用最高层的标签, 将多标签转化为单标签问题
    - 很难办, 效果直观的感觉就不好
    - 使用标签组里面的最大信度
    - 使用标签组里面的平均信度
    - 
- MSP方法: 最大标签的取值
  - 显然不行, 很可能某个标签有相当的信心, 但仍然存在未知的意图
- Mahalanobis距离: 以最高层标签作为单一类别
  - 标签重叠, 很难办


  ## Rethinking Feature Distribution for Loss Functions in Image Classification
  > @datetime: 2021/05/19
  >
  > [click for local file](./Wan_Rethinking_Feature_Distribution_CVPR_2018_paper.pdf)
  >
  > 提出了一个基于高斯混合密度模型的分类loss, 同时实现了分类和特征空间的密度建模. 相比于以往学习的模型, 这个模型并不是利用已有的模型的特征空间进行建模, 而是直接利用误差函数对特征空间的分布进行控制, 将其塑造为近似的高斯混合密度. 这个误差函数将分类和密度建模这两种任务同时实现, 也是相当值得关注的想法.

  - **IDEA**:
    作者希望对特征空间的的概率分布进行控制, 并为此提出了一个新的损失函数. 作者的基本假设是, 特征空间可以被塑造为近似的高斯混合密度. 因此对特征空间的密度假设为:
    $$
      p(x) = \sum_{i}p(x|z_i) p(z_i) = \sum_{i} \frac{\pi_i}{\sqrt{\left( 2\pi \right)^n \det \Sigma_i}} 
        e^{(x - \mu_i)^T \Sigma^{-1}_i (x-\mu_i)/2}
    $$
    通过Bayes公式可以得到后验概率, 通过 MLE 可以得到分类的损失函数. 假设训练集为 $\{ (x_i, y_i)| 1 \leq i \leq n\}$.
    则似然为:
    $$
      l_{cls} = \prod_{i} p(y_i|x_i) = \prod_{i} \frac{p(x_i|y_i) p(y_i)}{\sum_k p(x_i|y_k) p(y_k)}\\
      \log l_{cls} = \sum_i \log \frac{p(x_i|y_i) p(y_i)}{\sum_k p(x_i|y_k) p(y_k)}
    $$
    这里 $\pi_i, \mu_i, \Sigma_i$ 都是可训练的参数. 此时关于分类的 loss 有了雏形. 另一方面, 作者希望能够对特征空间的
    概率分布进行控制, 于是除了关于标签的条件似然之外, 还需要最大化关于训练集的似然:
    $$
      l_{lkd}(X) = p(X;\mu, \Sigma) 
      =  \prod_{i} p(x_i;\mu, \Sigma) 
      = \prod_{i} \sum_j p(x_i, z = j;\mu_j, \Sigma_j 
      = \prod_{i} \sum_j p(x_i| z = j;\mu_j, \Sigma_j)p(z = j)\\

      l_{lkd}(X,Y) = p(X, Y;\mu, \Sigma) =  \prod_{i} p(x_i, y_i;\mu, \Sigma) = \prod_{i} p(x_i| y_i;\mu_{y_i}, \Sigma_i) p(z = y_i)
    $$

    形式上, 后者比前者形式上更加的简单, 作者选用后者作为训练集的似然, 于是就得到了最后的 loss:
    $$
      L_{GM} = -l_{cls} - \lambda l_{lkd}(X,Y)
    $$

  - **Interpretation**:

    我们仍然能够类似于GDA的模型, 通过 Mahalanobis距离 对模型进行解释. 假设每个混合分量的先验概率相同, $p(z = k) = 1/K$, $K$ 为类别总数. 那么给定 $x_i$, 后验概率就简化为:

    $$
      p(y|x_i) = \frac{p(x_i|y) p(y)}{p(x_i)} \propto p(x_i|y)
    $$

    此时分类的标准就等同于条件似然. 相比于先前的 [Mahalanobis](#a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks)方法, 这里的协方差矩阵是不同的, 因此不能直接理解为Mahalanobis距离. 但是, 如果对上述结果进行形式上的变化:
    $$
    p(y|x_i) = \frac{p(x_i|y) p(y)}{p(x_i)} \propto p(x_i|y) \propto \exp( -(x_i - \mu_y)^T \Sigma^{-1}_y (x_i-\mu_y)/2 - \log \det \Sigma_y)
    $$
    这相当于带有偏置的 Mahalanobis 距离. 

  - **Improvement**:

    为了进一步的增强模型的分类性能, 作者希望能够增大样本之间的间隔. 参考上述的模型解释, 一种相当直接的做法就是, 人为地减小目标概率, 从而迫使模型针对性的增大目标概率. 一种做法是, 给目标概率的指数部分增加一个常系数, 即:
    $$
    p'(x_i|y_i) \\= p(x_i|y_i) \times e^{-m} \\\propto \det \Sigma_{y_i} ^{-1} \exp(-(x_i - \mu_{y_i})^T \Sigma^{-1}_{y_i} (x_i-\mu_{y_i})/2 - m) \\= \exp(-(x_i - \mu_{y_i})^T \Sigma^{-1}_{y_i} (x_i-\mu_{y_i})/2 - m - \log \det \Sigma_{y_i})
    $$
    记$d_k = (x_i - \mu_{k})^T \Sigma^{-1}_k (x_i-\mu_k)/2$. 为了获得正确的分类结果, 模型的参数必须满足:
    $$
    -d_{y_i} - m - \log \det \Sigma_{y_i} > -d_{k} - \log \det \Sigma_{k}, \forall k \neq y_i\\
      d_{k} -d_{y_i} > m + \log \det \Sigma_{y_i} - \log \det \Sigma_{k}
    $$
    
    于是, $m$ 就成为了调整距离间隔的一个超参数. 为了适应不同量纲的结果, 作者采用的方式是对 $d_{y_i}$ 进行一定比例的放大:
    $$
    p'(x_i|y_i)  \propto \det \Sigma_{y_i} ^{-1} \times \exp(-(1+\alpha) d_{y_i}))
    $$

    这里的 $\alpha$ 可以理解为一个惩罚系数, 分类信度越低, $d_{y_i}$ 越大, 给予的惩罚越大.
    
  - **Understanding**:

    - 我们可以将多元高斯分布理解为原型相似度的描述. 每个均值都是一个中心, 每个概率都是一个可能性, 或者通过 Mahalanobis 距离理解为信度. 

    - 利用聚类的观点, 分类簇从距离度量的角度上讲, 我们希望簇和簇之间距离相对远, 簇内部相对紧致. 作者采用和方式就是强制不同类别之间有一定的间隔. "簇和簇之间距离相对远, 簇内部相对紧致" 这个要求其实是有一定的矛盾的:

      - 如果簇和簇之间距离远, 那么即使簇内部松散, 也可以有相对较大的类间间隔
      - 如果簇内部足够紧致, 那么即使簇间距离相对比较小, 也可以有相对较大的类间间隔

      缓解这个矛盾的一个方法是, 将簇边界的距离和蔟中心的距离正相关. 

  - **DISCUSSION**

    - 如何利用已经有的标签辅助概率建模?
      - 利用标签训练模型, 实际上是在训练特征映射, 但是这种训练并不是针对密度建模设计的. 这种方式获得的模型仅仅只是满足分类任务的一个模型而已; 
      - 已有的预训练模型基本都是上述的思路. 利用预训练模型, 分不同类别进行建模, 再整合各个类别的结果, 这个就是 MSP, Mahalanobis, EM 模型的思路;
      - 本文的方法则是在训练过程中利用类别, 直接对特征空间的分布进行控制, 使得模型在确保较高的分类准确率的同时, 选择一个更适合密度建模的特征空间分布;
      
    - 单标签分类的问题中, 我们期待空间中的样本有什么分布?
      - 对于单标签问题, 我们自然期待同一类别的样本构成一个紧凑的簇, 不同类别样本的簇之间有较大的间隔. 这样可以增大同一类别的相似性,增大不同类别的差异性; 这个看法其实是和单一类别的原型聚类相关的
      - 对于一个理想的分布, 各个簇的空隙间的, 远离这些簇的, 都可以理解为 OOD 样本, 这是非概率的理解
      - 如果可以计算出 $p(x)$, 那就直接了
      
    - 多标签分类问题中, 我们期待特征空间中样本有什么分布?
      - 多标签问题是不是也可以和允许一个样本内同时属于多个簇的软聚类问题关联思考? 
      - 对于多标签的问题, 我们可能需要考虑标签和标签之间的关联性.
        - 从图的角度上讲, 标签和标签之间的关联性可以考虑通过一个带有权重的图来表示. 概率图之类的模型或许可以发挥一定的作用?
        - 一个样本同时拥有多个标签, 直觉上我们是可以认为这几个标签可能有一定的相似性. 数据提取关联性, 和推荐系统有些关联.
        - 这些问题和我们关心的 $OOD$ 问题之间有什么关联?
      - 一个简单的想法:
        - 我们从聚类的角度思考, 但是由于多标签的特性, 我们需要考虑软聚类
        - 希望相似的标签之间更接近, 不同的标签之间相距更远一些
        - 我们可以期待, 对于某个标签来说, 其样本空间呈 $Gaussian$
        - $Gaussian$的均值是这个标签类中心, 方差控制离散程度和形状

  ## OOD DETECTION FOR ML PROBLEM

  ## remain: Additive margin loss
  ## remain: MMC loss THU
  ## remain: representation of multi-label problems