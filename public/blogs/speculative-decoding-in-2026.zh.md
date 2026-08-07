---
title: "Guess and Check: How Speculative Decoding Buys Speed for Free"
date: "2026/8/6"
---

从一个大模型里吐出一个 token，要把模型的全部权重读一遍，却几乎不拿它们做什么运算。batch size 为 1 的时候，GPU 的时间基本都耗在等内存上，算力单元大半闲着。speculative decoding 的全部机会就在这个失衡里：既然卡住你的是搬运而不是计算，那么一次检查好几个 token，花的钱并不比检查一个多多少。

于是可以下这么一个赌注。找个便宜的东西先把接下来几个 token 猜出来，再把这几个猜测一次性喂给大模型，让它对每个位置同时回答一个问题：这个 token 我自己会不会也写出来？凡是它点头的，你就白赚一个，省下了本该为它单独跑的那次前向。

让这件事不止是个小聪明的，是这些猜测可以用一种方式采纳，使输出分布**精确地**保持不变。这个结论来自两篇同期独立完成的工作，ICML 2023 上的 [Leviathan et al.](https://arxiv.org/abs/2211.17192) 和 DeepMind 的 [Chen et al.](https://arxiv.org/abs/2302.01318)，两边都报了约 2–3 倍，输出与原模型逐字一致。此后所有的工作，说到底都是在把"猜"这件事做得更好。下面每一张图都出自我为这篇文章写的模拟器，所以图里的数字是量出来的，不是我讲出来的。

## 一、这条规则，以及它换来什么

记 `p` 是 target 在某个位置上的分布，`q` 是 drafter 在同一位置上的分布。drafter 提议，target 拍板：

```python
x = sample(q)                            # the draft proposes a token
if uniform() < min(1, p[x] / q[x]):      # the target checks it
    emit(x)
else:
    emit(sample(normalize(maximum(p - q, 0))))
```

承重的就两行。接受概率 `min(1, p[x]/q[x])` 的意思是：target 对这个 token 的偏好只要不低于 drafter，就无条件收下；偏好更低，就按低多少的比例收。而某个 token 一旦被拒，替补**不是**从 `p` 里重采的，而是从 residual 里采的，也就是 `p` 当中 `q` 没盖住的那部分，重新归一化。

residual 是大多数介绍会跳过的一步，偏偏就是它让整件事精确。按比例接受，等于你从 `min(p, q)` 里采了个样，比 `p` 恰好少了 `(p − q)⁺` 这一块；把被拒的那些从这块缺口里补采回来，少的正好补齐。

![The accept and reject rule](blogs/images/specdec-accept-rule.svg?v=1)

绿色是被直接收下的质量，加起来就是 acceptance rate α。蓝色是 residual 要补的那部分。这样一看，α 有个很干净的闭式：`α = Σ min(p, q) = 1 − TV(p, q)`，一减去 drafter 和 target 之间的 total variation 距离，这是原论文的推论 3.6。一个有一半时间跟 target 想到一块去的 drafter，就是一个分布离 target 半个 TV 单位的 drafter，跟它是什么架构没关系。

"这是精确的"这种话很容易点头，代码里也很容易出微妙的错，所以值得亲眼看一遍。我在八个 token 上取了一个 target 分布 `p`，故意配一个离它 0.34 个 TV 的烂 `q`，跑了两百万次单 token 的 speculative 步。

![Empirical distribution after two million speculative steps](blogs/images/specdec-lossless.svg?v=1)

drafter 错得肉眼可见，众数都押在错的 token 上。输出没有：实测输出与 `p` 的 TV 距离是 0.00046，而两百万次采样的噪声地板约 0.00071。drafter 自己那 0.34 的误差不是被削小了，是干干净净地没了。这正是 speculative decoding 在一堆推理优化里显得特别的地方。量化拿质量换速度，剪枝拿质量换速度，它什么都不换：drafter 差，你亏的是速度，永远不是正确性。

接下来算账。一轮里 drafter 出 γ 个 token，target 一次前向全部验完。第一次被拒若发生在第 k 个位置，这轮吐出 k+1 个：k 个被接受的，加上那个重采的替补；γ 个全过，就吐出 γ+1 个，多的那个是从 target 自己在最后一个位置上的分布里白拿的，因为这次验证前向早就把它算出来了。在原论文声明的简化假设下（各位置接受与否独立同分布、接受率为 α），一轮的期望产出是个截断几何分布 `(1 − α^(γ+1))/(1 − α)`，我的模拟器在测过的每一组 α 和 γ 上都能复现到采样误差以内。α = 0.75、每轮出四个的时候，21 个 token 只花了 6 次 target 前向，而不是 21 次：

![Target forward passes, autoregressive versus speculative](blogs/images/specdec-timeline.svg?v=1)

得留神这张图只数了 target 的前向次数。讲机制这是最诚实的口径，当加速比看就偏乐观了，因为打草稿本身不免费。

所以把这笔成本放回去。记 `c` 是一次 draft 前向相对一次 target 前向的成本占比，墙钟加速就是

$$
\text{speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}
$$

矛盾这就摆出来了。γ 往大调，分子涨得越来越慢，因为第 i 个 draft token 只有在它前面全过的情况下才算数；分母却是线性涨的。所以有个最优值，而且它跟着 α 走：

![Speedup against draft length for several acceptance rates](blogs/images/specdec-speedup.svg?v=1)

α = 0.9 时最优是一次出七个，约 2.4 倍；α = 0.5 时最优是一次只出一个，约 1.25 倍，出更多反倒更慢。真正要盯的数字是 acceptance rate，不是 drafter 的大小：α 既决定你能赢多少，也决定你被允许玩得多凶。

有个流传很广的说法值得就地消灭。推论 3.9 说的是：只要 **α > c**，就必然存在某个 γ 能带来提升。按常见的 α 在 0.6 到 0.8 算，drafter 只要比 target 快 1.3 到 1.7 倍就够，而不是常被引用的十倍到三十倍。同一篇论文的 Table 4 里就摆着一个 c = 0.11 的配置，drafter 只快约九倍，照样拿到 1.7 到 2.2 倍。已发表的设置基本落在 c ≈ 0.02–0.13，而且两篇原始论文谈甜点区用的都是参数量，不是延迟比。

## 二、drafter 怎么设计

上面一直把 drafter 当成一个产出 `q` 的黑盒。既然真正起作用的只有 α 和 c 两个量，那每一种 drafter 设计都是同一个平面上的一步棋，这个平面值得直接摊开看：

![The design space of drafters](blogs/images/specdec-design-space.svg?v=1)

虚线是 α = c。两个箭头是所有人手里仅有的两根杠杆：往上推，让 drafter 更常猜中；往左推，让它更便宜。这个方向上几乎每篇论文都是在拉其中一根，而有意思的那些会发现两根杠杆之间是纠缠的。

最早的提议是同系列里的另一个更小的模型。它不用改 target，任何一对都能配，但你得正好有一个在相似数据上训过的小模型，而它的 α 是多少就是多少。它的毛病是结构性的：一个独立的模型必须从零把上下文重建一遍，而 target 内部早就把这份上下文编码得妥妥当当了。

于是主线就把独立的小模型换成了一个读 target 内部状态的 head。EAGLE 让一个小的自回归 head 以 target 的 hidden feature 为条件，等于是揣着 target 自己对上下文的理解去猜，而不是靠自己那份更弱的。这纯粹是在拉 α，而且之所以划算，是因为那些 feature 本来就已经算出来了，所以这个 head 可以做得极小。

第三族转头攻 c，攻法在公式里一眼可见。如果 γ 个 draft token 得靠自回归一个个产出，那分母里的 `c` 其实是 `γ · c_单次`，于是不管 drafter 多好，最优 γ 都被死死压低。并行 drafter 把这个耦合打断了：一次前向出一整块，`c` 从此跟 γ 彻底脱钩。[DFlash](https://arxiv.org/abs/2602.06036) 用的是一个以 target 上下文特征为条件的小 block diffusion 模型，这个选择很顺，因为 diffusion 天生就是跨位置并行的。

还有一族压根不学习。SuffixDecoding 在 prompt 和历史输出上建一棵后缀树，靠匹配出草稿，`c` 逼近于零，位置在这个平面的最左端。它的 α 则完全看工作负载：在开放式散文上近乎没用，而在 agentic 这类模型大段引用自己先前输出的场景里，好用得不太讲道理。

## 三、drafter 怎么训练

把 α 往上推是个训练问题，而它有一个压倒性的失败模式：drafter 被训练在一个它推理时根本遇不到的分布上。

朴素的目标函数是：给一段真前缀，让 draft head 预测 target 的下一个 token。可推理时 drafter 要连着往前跑好几步，每一步依据的是**它自己**先前的猜测，而那些猜测有时是错的。到第三个位置上，它被要求续写的是一段训练里从没出现过的前缀。这就是普通的 exposure bias，落到指标上就是接受率沿着块往后掉。

[EAGLE-3](https://arxiv.org/abs/2503.01840) 用作者称为 training-time test 的办法修它：训练时就把多步 drafting 的过程模拟一遍，让这个 head 面对的是它推理时真会遇到的分布。同一篇还做了第二个改动，放弃预测 feature、改成直接预测 token，并报告这个组合让 drafter 能持续从更多训练数据里获益，而原来那种预测 feature 的形式做不到。

并行 drafter 有个相关但不同的毛病。既然块内每个位置都是独立预测的，就没什么东西拦着第 4 个位置跟第 3 个位置自相矛盾，于是越靠块尾，一致性越差。而这条衰减曲线的形状，比它看上去更要紧：

![Acceptance along the block](blogs/images/specdec-suffix-decay.svg?v=1)

一轮的产出是 `1 + Σ_k Π_{i≤k} α_i`，是一串**连乘**，所以靠前的位置权重远远压过靠后的，一条往下掉的尾巴造成的损失，比"平均接受率"这个数字暗示的大得多。上图三条曲线里，衰减那条整块的平均接受率并不难看，产出却只有每轮 3.81 个 token，而平坦那条是 4.33。

修这条衰减正是 [DSpark](https://arxiv.org/abs/2607.05147) 干的事：在并行主干之上叠一个极轻的序列 head，把块内位置之间的依赖重新注回去。它的 Markov 变体是把一阶转移矩阵做低秩分解，说白了就是在并行模型给出的边缘 logits 上加一个秩为 r 的偏置，用来重建一部分联合分布。把衰减压平比把平均值抬高更值钱，这也是上图第三条曲线能到 5.32 的原因。

## 四、verification 怎么设计

verification 这一步有三年时间都是一条统一施加的固定规则，而现在剩下的空间有相当一部分就在这儿。

第一个观察是：没什么规定说草稿必须是一条链。如果 drafter 在同一个位置上给出好几个候选，target 完全可以在同一次前向里把它们全查了，反正这次前向本来就是访存受限的。验证规则可以自然推广：先试第一个候选，被拒就更新 residual，再拿第二个去试。

![Chain versus tree, and acceptance against candidate count](blogs/images/specdec-tree-verify.svg?v=1)

还是同样的 p 和 q，一个候选时接受率 66%，六个候选时 87%。这里最反直觉的是：这些**依然是精确无损的**。多给几个猜测并不会让输出有偏，只是让你更少退回去用 residual。这就是 tree drafting 的思路，而 EAGLE-2 又往前推了一步，让树的形状随时变化而不是钉死的，把候选预算花在 drafter 最没底的地方。

第二个观察是：验证长度不必是常数。一段长草稿只有在它大概率活得下来时才值得验；而在高负载下，target 的 batch 容量是一种被争抢的资源，投机正在跟真实请求抢它。DSpark 的另一半就是按请求调度验证长度，依据是一个校准过的前缀存活概率估计，加上引擎的吞吐画像，好让忙起来的服务器别再把 target 的容量花在多半会被丢掉的投机上。

第三个观察不太舒服：你可以验得松一点，然后跑得更快。[Medusa](https://arxiv.org/abs/2401.10774) 在 target 上挂若干个解码 head，验证用的是它称为 typical acceptance 的做法：target 给这个 token 的概率只要越过一个跟熵有关的阈值就接受。论文对这笔交易是挑明了说的，不是含糊过去："我们确认，通常并没有必要匹配原模型的分布"，附录里还有一句更直白的："我们并不坚持输出与语言模型分布之间的精确对应"。它留了个旋钮，调大就是拿更激进的接受换速度。

值得知道的是，开山那篇先到过这里，然后选了相反的方向。它的附录 A.5 描述的正是这样一种松弛，取名 lenience，并报告它在 T5-XXL 上能到 5 倍而严格版是 3 倍，然后把它隔离了：全文其余部分一律用最严格的算法，不允许任何 lenience。同一个想法，一篇当污染物，另一篇当默认值。温度为 0 时这个区别会塌缩，因为 typical acceptance 会退化成贪心。但它确实意味着"无损"这个词得读仔细：它是关于那条拒绝规则的一个精确断言，一个改了规则的方法并不自动继承它。

## 五、把它塞进引擎

到这里为止讲的都是孤立的一个请求。而把 drafter 放进一个真的服务引擎里，会捅破好几件那个引擎本来已经安顿好的事。

第一件是 batch 的形状。正常的 decode 步喂给 target 的是 B 条序列、每条一个 token。verification 喂的是 B 条序列、每条 γ+1 个 token；草稿要是树而不是链，那就是树有多少节点就多少个。这意味着每一轮的 kernel 启动形状都在变，而它会跟想要固定形状的 CUDA graph 打架，也会跟本来就在抢同一份 token 预算的 chunked prefill 打架。

第二件是 attention mask。验证一条链很容易，因为草稿 token 是因果有序的，普通的 causal mask 就够。验证一棵**树**就不行了：同一个位置上的两个兄弟候选彼此绝不能互相看见，而两个都得看见它们共同的祖先。这需要一个把树的拓扑编码进去的自定义 mask：

![Causal mask versus tree mask](blogs/images/specdec-tree-mask.svg?v=1)

在这棵五个节点的树上，普通 causal mask 会多放行四对，让 `b` 看见它的兄弟 `a`，让 `d` 同时看见 `b` 和 `c`。这里每一处都会让某个候选去条件于一个根本不在它自己路径上的 token，从而悄悄把验证搞坏。按树的拓扑把正确的 mask 造出来，正是 [SpecInfer](https://arxiv.org/abs/2305.09781)（ASPLOS '24）真正的系统贡献，它提出的 tree-based parallel decoding 让一整棵 token 树能在一次前向里验完，报告分布式推理 1.5–2.8 倍、offloading 推理 2.6–3.5 倍。

第三件是 KV cache。那些最终被拒的 draft token，slot 是已经写进去了的，引擎得把它们回滚，而且回滚时不能扰动 batch 里其他序列的 block table。用分页缓存的引擎在这件事上占了便宜，因为回滚只是改 block table 里的记账，不涉及数据搬运。这算是分页的一个不太被提起的好处。

落到实处，两个开源引擎收敛到了类似的结构。[vLLM](https://docs.vllm.ai/en/stable/features/speculative_decoding/) 在 `speculative_config` 后面暴露一个 proposer 接口，把 `ngram`、`suffix`、`draft_model`、`mtp`、`eagle3`、`dflash` 都做成一等方法，另外它的测试集里有一个 rejection sampler 收敛测试，检查采样器的输出分布是不是真的跟 target 对得上。那个测试就是第一节那条保证的工程形态：无损是一个你可以拿来做回归测试的性质，而一个当真的引擎会为它写测试。

SGLang 则是让 [speculative worker](https://docs.sglang.ai/advanced_features/speculative_decoding.html) 和它的 overlap scheduler 一起跑，后者想把 CPU 上的调度工作藏到 GPU 执行的背后去。这个组合是真的难搞，而限制是写在文档里而不是藏起来的：overlap 目前只支持 `--speculative-eagle-topk 1`，也就是链的情形，更宽的树还在路线图上。这挺能说明问题：tree drafting 和 scheduler overlap 都想去当那个填满 GPU 空闲时间的人，于是它们会互相挡道。

最后一块是知道什么时候该收手。既然投机是在跟真实请求抢 target 的 batch 容量，那么负载上来时引擎应该能把它关掉。vLLM 可以按 running queue 的长度把 proposed length 打到零来做这件事，不过按一个[未关闭的 issue](https://github.com/vllm-project/vllm/issues/25112) 的说法，自动版本并不总能在配置的 batch size 处如期关闭。这跟 DSpark 的 confidence scheduling 是同一个念头，只不过一个是从运维那头走过来的：负载低于某条线时投机几乎白送，高于那条线，你就是在从本来也能被服务的请求那里偷容量。

## 六、收益会跑到哪里去

开头那段的前提是 GPU 有闲着的算力。这个前提会随着机器被填满而变弱，speculative decoding 的收益也跟着变弱。

![Reported speedup against batch size](blogs/images/specdec-batch-size.svg?v=1)

[有人直接量过](https://arxiv.org/abs/2310.18813)：同一套配置在 batch size 为 1 时是 2.73 倍，到 batch size 32 就只剩 1.31 倍，理由是"大 batch 本身已经把 GPU 算力占满了"。EAGLE-3 自己发表的表格是同样的形状：SGLang 上 batch 64 掉到 1.38 倍，vLLM 上 batch 56 掉到 1.01 倍。这本质上是一个延迟优化，只有机器空着的时候才看着像吞吐优化。

2026 年真正需要补的限定词是序列长度，也就是图上第四条线。[MagicDec](https://arxiv.org/abs/2408.11049) 指出，长上下文下 KV cache 的规模是 batch 乘序列长度，光把它读进来就足以让工作负载在大 batch 下依然访存受限，并报告 Llama-3.1-8B 在 batch 32 到 256 的区间里最高 2.51 倍。所以访存受限这个前提不是被"批处理"推翻的，是被"短序列加小缓存"推翻的。

收益也可能是负的。[llama.cpp 上的一个 issue](https://github.com/ggml-org/llama.cpp/issues/23752) 报告在 M1 Max 上，MTP speculative decoding 在所测的每一种配置下都是净亏损，从 25.3 tokens/s 掉到 19.3，理由是"Metal 上 draft 的前向开销超过了投机带来的收益"。这是单个用户的报告、状态还未确认，但它是个很具体的提醒：c 是你的硬件和 kernel 的属性，不是纸面上参数量的属性。

所以整个领域的形状，是被一个不等式定下来的。让 drafter 以 target 为条件、并针对它真会面对的分布去训练，把 α 抬上去；让它一次前向出一整块，把 c 压下来；然后把结果交给一个知道"机器什么时候满到不值得再玩这套"的引擎。那些发表出来的最大值，最好读作"这个方法在它最有利的那一种配置下能到多少"，而能被独立测出来的数字更接近 1.5 到 3 倍。不过真正有意思的从来不是那个倍数，而是你拿到它的时候，模型本来要说的话，一个 token 都没变。
