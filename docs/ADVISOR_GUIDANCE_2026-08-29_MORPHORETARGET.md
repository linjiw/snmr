# Advisor guidance (2026-08-29): learning-based, RobotSpec-conditioned, physics-verified retargeting

Forwarded verbatim by the owner on 2026-08-29 with the instruction: "read this guidance and see how
to improve our current work and use best of newton and isaacsim physics validation like verifiable
RL ... and lets then work towards your new plan after done the reading". Fable's assessment and the
resulting plan are in `docs/MORPHORETARGET_PLAN_2026-08-29.md`; this file is the source record and
is not edited.

---

## Owner's question

我很好奇一件事情，就是说现在这些，比如说 retargeting，Human retargeting 的这些工作，好像他们 retargeting 都是类似于算出来的一个解法，就是给 motion，然后加入他们的 retargeting 算法，然后他们算出来一个机器人的 retargeting 的结果。有没有，或者我们来 explore 一下有没有什么可能的 learning-based 的方法呢？就是我们如果能够 scale 有很大的 human motion set，然后比较干净的 dataset，然后我们也有一个可以一直验证的，或者可以物理上进行验证的，比如说 Newton，比如说 Isaac Lab，Isaac Sim，可以帮助验证 retargeting 的结果模型，那是不是相当于一个可以非常快的 RL 的问题呢？然后对于这个问题，我们能不能通过一个 learning-based 的方法，或者 RL-based 的方法，我们来学一个 retargeting。比如说你只要给定你这些机器人的参数，那我们就能学一个针对你这个机器人的 retargeting 的结果。然后我们能做到这些吗？然后我说给定一个机器人参数，就比如说如果我给你一个机器人 robot，你会要这个机器人的什么东西？比如说它可能要一段 DF，可能它的各个关节、电机那些参数或者什么东西。如果这些都是标准化的范式，然后又都可以在，都是一些成型的，然后定义很好的 robot，比如说 Isaac Lab里面之前所有的 robot 的类型，那我们是不是可以做一个这样的事情？这个变成一个，我们又有 dataset，又有一个又可以学一个模型去做这个 retargeting。Retargeting 之后，这个东西还能够在 Isaac Lab 或者 Newton 里面被实现，被模拟，甚至被 verify 它的各个的效果。我们能不能做一个设计这样的一个东西呢，来做实验？

## First advisor reply

Joe。这个方向很有意思，而且时机上你的直觉是对的——先说结论：**纯 learning-based retargeting 已经有人开始做了，但"以机器人参数为条件、能 zero-shot 泛化到任意新机器人"的通用 retargeter 目前还是一个基本开放的问题**，而且你说的几个要素（大规模干净的 human motion 数据、Isaac Lab/Newton 做物理验证）恰好是这个问题最近才变得可行的原因。我先讲一下现状，再帮你把这个系统设计拆开。

### 现在的领域格局

主流 pipeline 确实还是"算出来的"：GMR（General Motion Retargeting）、mink 这类基于优化/IK 的方法是目前 humanoid tracking 工作的标配前端。但 learning-based 的苗头已经出现了。比如最近有工作直接用神经网络把 human motion 映射成 humanoid 序列——用 1D ResNet 编码器提取特征，再过一个 Transformer 做逐帧预测，并且用 MuJoCo 的碰撞检测加载机器人 URDF 来做自穿透检测、用脚部离地高度做物理合理性过滤。还有 ReActor 这种直接用 RL 做 physics-aware motion retargeting 的工作，以及 CrossLoco 这类用 RL 框架把人的动作迁移到四足和机器人上的 cross-morphology 工作。

但注意，这些工作几乎都是**针对单一机器人训练一个 retargeter**（比如专门给 G1 训一个）。你问的那个更野心的版本——把机器人参数本身作为输入条件——对应的技术积木其实在另一个社区已经比较成熟了，就是 universal morphology control 这条线：MetaMorph 用 morphology-aware Transformer 在模块化机器人设计空间里学 universal controller，ModuMorph 用 morphology-conditioned attention 改进它，HyperDistill 再把知识蒸馏进一个 hypernetwork、按 morphology 生成机器人专属的 MLP 控制器。GET-Zero 用 graph-based embodiment transformer 把策略学习条件在手的结构图上，实现了对没见过的 morphology 的 zero-shot 迁移；灵巧手抓取那边也已经有人把每个手的 URDF 转成结构化的 morphology token 来获得 embedding。humanoid 方向最近也有把不同机器人的关节映射到一个语义对齐的全局关节空间（比如定义一个 32 维的 canonical joint ordering），再在上面构建 embodiment graph 来显式编码形态结构的做法。

所以你的 idea 本质上是把这两条线接起来：**morphology-conditioned architecture × motion retargeting × sim-based verification**。这个交叉点目前还没有被做透，是有真实 novelty 空间的。

### 一个关键的概念澄清

你的描述里有一个地方需要先拆开，不然实验设计会混乱：**retargeting 和 tracking 是两层**。Retargeting 输出的是一条运动学参考轨迹（joint angles + root trajectory），它本身不能直接"在 Isaac Lab 里跑"——把它扔进物理仿真需要一个 tracking controller 去执行。所以"物理验证"这件事有两种含义：

1. **弱验证（运动学/准物理）**：joint limit、自碰撞、脚不打滑不悬空、速度加速度不超限、支撑多边形内的准静态平衡。这些不需要 controller，可以又快又可微地算。
2. **强验证（动力学）**：把 retargeted 轨迹交给一个 tracking policy，看它能不能真的在仿真里跟上（tracking error、摔倒率）。这才是最终 ground truth，但它引入了一个新依赖——你需要一个 tracker，而 tracker 本身的好坏会污染对 retargeter 的评价。好消息是现在有 UniTracker、GMT、BeyondMimic 这类 general motion tracker，可以当作一个相对固定的"verifier"来用。

想清楚这一点后，我建议的定位是：**retargeter 学的是运动学映射，但训练信号里注入物理**——而不是让 RL 直接在动力学层面学 retargeting（那其实就退化成 H2O 那种端到端 tracking 了，反而失去了 retargeting 作为可复用中间产物的价值）。

### 系统设计草案

**输入表示（回答你"要机器人的什么"）**。核心就是 URDF/MJCF 能给的东西，按每个关节 token 化：kinematic tree 的拓扑（parent-child 边）、每个 joint 的类型和轴向、joint limits（位置/速度/力矩）、link 长度和质量/惯量、end-effector 和 foot 的标记、以及一组 human-robot 的 keypoint 对应关系（哪个 link 对应人的手腕/脚踝/髋，这个可以半自动从命名和拓扑推断）。架构上就是每个关节一个 token，用 topology-aware attention（把 kinematic tree 的邻接当 attention bias）——这正是 embodiment-aware transformer 的做法：kinematic token、沿运动学边做 message passing 的 topology-aware attention bias、加 per-joint 属性条件。电机动力学参数（armature、damping、力矩曲线）对第一阶段不是必需的，但如果你想让 retargeter 输出"这个机器人力矩不够就自动放慢/收敛动作幅度"这种行为，就要加进去。

**训练范式：我建议三阶段，而不是一上来就 RL。**

第一阶段，**amortized optimization / 蒸馏**：对一大批机器人（下面讲怎么来）跑现成的优化式 retargeter（GMR 等），生成 (human motion, robot params) → robot motion 的配对数据，监督训练 conditioned 网络。这一步的价值是把"每条 motion 跑一次优化"变成一次 forward pass，天花板是优化器本身的质量。

第二阶段，**自监督 + 可微物理先验精调**：不依赖标签，直接用可微 FK 算 keypoint matching loss（scaled 之后和人的 keypoint 对齐）、加上可微的软约束（joint limit、脚接触一致性、smoothness、自碰撞的 SDF 近似）。这一步可以让模型超过第一阶段那个优化器 teacher。Newton 在这里有一个独特的角色：它是基于 Warp 的、支持可微仿真，所以理论上你可以把一小段真实 rollout 的物理残差也做进 loss 里，这是 Isaac Gym 时代做不到的。

第三阶段，**sim-in-the-loop 的验证信号**：用一个冻结的 universal tracker 在 Isaac Lab 里大规模 rollout retargeted 结果，把 tracking 成功率/误差作为信号回流。这里我反而不建议做成经典 policy-gradient RL——对一个输出整条轨迹的序列模型做 RL，credit assignment 很差、方差很大。更实用的形式是 **rejection sampling / 仿真过滤 + 迭代自训练**（有点像 RLHF 里的 best-of-n + SFT）：retargeter 采样多个候选，仿真验证，把通过的高质量样本加回训练集。或者用 tracking error 做 per-frame 的 reward-weighted regression。这样"仿真作为 verifier"的思想保留了，但训练稳定得多。

**数据和泛化的真正瓶颈：机器人数量，不是 motion 数量。** Human motion 侧你有 AMASS、LAFAN1、Motion-X，量是够的。但 Isaac Lab 里现成的 humanoid 也就一二十个（G1、H1、GR-1、T1 之类），对于学"以机器人为条件的泛化"来说太少了——模型会直接记住这十几个机器人。所以这个项目里最重要的一个工程组件其实是**程序化 morphology 增广**：以现有 humanoid 为种子，随机扰动 link 长度、质量分布、joint limit、DoF 数量（比如腰 1 自由度 vs 3 自由度、有无 ankle roll），程序化生成几千个"合法的假 humanoid"。GenLoco 在四足上、UNIMAL benchmark 在模块化机器人上都验证过这个思路是 zero-shot 泛化的关键。这样真实机器人（包括训练时没见过的）就落在增广分布内部。

**评价设计**：held-out 机器人上的 zero-shot 表现是核心指标——(a) 对比 per-robot 优化式 retargeting 的 keypoint 误差和物理指标（脚滑、穿透、limit violation）；(b) 冻结 tracker 下的 tracking 成功率；(c) 推理速度（神经网络应该比逐帧优化快几个数量级，这也是它实用价值的卖点：real-time teleoperation、大规模数据集离线 retarget）；(d) 一个很有说服力的 ablation 是"给一个全新机器人，优化式方法要人工调 config 和权重，我们的方法 URDF 扔进去直接出结果"。

### 需要提前想清楚的几个坑

一是 **retargeting 本身是 ill-posed 的**：同一段人的动作在一个机器人上有多个合理解（尤其 DoF 不匹配时），纯回归会学出"平均解"导致动作发糊。所以生成式建模（diffusion / flow matching，条件在 morphology token 上）可能比确定性回归更合适——事实上已经有 graph-conditioned diffusion 做 retargeting 的苗头了。二是 **keypoint 对应关系的定义**在跨形态时是隐含的人工先验，怎么让它也自动化（或从拓扑学出来）本身就是问题的一部分。三是 verifier 的偏差：tracker 对某些机器人本身就弱，会把 tracker 的问题错误归因给 retargeter，可能需要 per-robot 归一化的成功率基线。

总体上我觉得这是个设计得当就能在一两个学期内出结果的项目：第一阶段的蒸馏版本几周就能有 demo，真正的科学贡献在第二、三阶段（超越 teacher、zero-shot 到新 URDF）。如果要收缩范围先做 MVP，我会建议：固定 5 个真实 humanoid + 500 个程序化变体，AMASS 子集，GMR 当 teacher，先只做到"held-out 机器人上媲美 per-robot 优化"，物理验证先用弱验证指标，强验证放到第二篇。你现在手头有倾向用哪个 tracker 当 verifier，或者已经有偏好的机器人集合了吗？这会影响 canonical joint space 怎么定义。

## Second advisor reply (full review of the first)

Joe，我重新把这份 advice 按照**论文新颖性、工程依赖、实验可证伪性**完整审了一遍。我的判断是：

> **方向是对的，但原方案还把"已有工作""工程 baseline"和"真正科学贡献"混在了一起。**
> 我们可以从 GMR 起步，但绝不能把项目定义成"用神经网络蒸馏 GMR"。真正值得做、也更符合你们现有积累的，是：
>
> **显式读取 RobotSpec、无需 robot-specific head/prompt/retargeting loss tuning、对整台未见 humanoid 做 zero-shot retargeting，并通过 physics verification 与 failure-localized repair 提高动态可执行性。**

这一区分非常重要。因为截至 **2026 年 8 月**，GMR 已经可以在 CPU 上实时运行；HoloRetarget 和 IKMR 又把单机器人 neural retargeting 推到了很高吞吐；NMR 已经做了 physics-refined paired data 加 Transformer；AdaMorph 已经覆盖多种 humanoid；G-DReaM 已经把 robot skeleton 编码成 graph-conditioned diffusion。单纯"更快""learning-based""multi-robot"都不再足以单独支撑一篇强论文。([GitHub][1])

真正还没有被做透的是：**不靠静态 robot ID、不靠每机器人独立 output head、不为新机器人重新训练或优化 prompt，而是直接根据机器人运动学、质量惯量、执行器和控制属性产生适配结果。** AdaMorph 使用每个机器人的 learned prompt 与 embodiment-specific adapter；G-DReaM 更接近我们的方向，但仍依赖人工 correspondence，新机器人要进行额外适配，而且当前 robot condition 主要是运动学结构，论文也明确把 contact、balance、torque 等动力学因素列为不足。([arXiv][2])

---

### 一、对原 advice 的逐项审查

#### 1. Retargeting 与 tracking 必须分开：完全正确

这个判断应该保留，而且要成为实验设计的基本原则。

Retargeter 输出的是 reference：

$$
Y_R=
\{x^{root}_{0:T},q_{0:T},\dot q_{0:T},c_{0:T}\},
$$

tracker 才把 reference 变成 torque 或 position target，并在闭环动力学中执行。

所以不能把"tracker 跟不上"直接等同于"retargeting 不可行"。我们至少要同时保留：

* **controller-independent verification**：joint limit、collision、penetration、contact、inverse dynamics、torque margin；
* **controller-dependent verification**：冻结 tracker 后的 tracking error、fall rate、full-clip survival；
* **downstream utility evaluation**：分别用不同 retargeting 数据重新训练相同 tracker。

这三者回答的是不同问题，不能用一个 rollout success 数字全部替代。

#### 2. "GMR 蒸馏 → 自监督约束 → simulator feedback"三阶段：方向正确，但 GMR 只能是起点

GMR 的实现仍然读取目标机器人专属的 IK JSON config，包括 human–robot match table、position/orientation weight、offset、scale、root definition，然后逐帧调用 Mink 求解 IK。换句话说，它非常适合作为 teacher 和强 baseline，但它的结构本身仍然携带大量 robot-specific prior。

因此：

* **可以**用 GMR 生成初始 paired data；
* **不能**把复现 GMR 结果当作最终贡献；
* **不能**让最终模型在 inference 时仍依赖 GMR seed，否则新机器人依然需要 GMR config；
* **不能**用 GMR 标签训练 dynamics-aware model，然后声称模型理解 motor torque、mass 或 latency。

最后一点尤其关键：GMR 的输出基本只由 kinematics 和人工 IK objective 决定。同一个 skeleton，如果仅把 motor torque 降低一半，GMR teacher 仍会输出几乎相同的 motion。于是网络最理性的行为就是**忽略你输入的 torque、mass 和 latency**。

所以 GMR 提供的是：

> **kinematic prior，不是 dynamics-conditioned target。**

真正的 dynamics-conditioned label，必须来自 SPIDER、PhySINK、trajectory optimization、local physics repair，或者 simulator 对多个候选的 ranking。SPIDER 和 PHUMA 已经分别提供了 physics-based retargeting 与 physics-constrained dataset pipeline，因此更适合作为较小规模的 high-quality teacher，而不是让我们从零写完整动态优化器。([GitHub][3])

#### 3. "Universal tracker 作为 verifier"：思路正确，名字和协议要改

目前多数所谓 general/universal tracker，主要是：

> 一个机器人跟踪很多 motion，

而不是：

> 一个完全相同的 policy 对任意新 morphology 都有效。

Holosoma 当前支持 G1、T1、多 simulator 和 whole-body tracking，是很好的 verifier/tracker 模板；但我们仍然应该为不同机器人使用**相同架构、相同训练 recipe、相同预算下训练的 per-robot tracker**，而不是把某一台机器人上训练好的 policy 直接当作所有 morphology 的绝对裁判。

最终协议应是：

1. 每个 robot 有一个冻结的 tracker；
2. 所有 retargeting 方法使用同一个该机器人 tracker；
3. tracker 本身先在强 teacher data 上达到最低合格线；
4. 同时报告 controller-independent feasibility；
5. 再做"用不同 retargeted dataset 重新训练 tracker"的 downstream experiment。

这样即使某台机器人 tracker 比较弱，我们也不会把 tracker weakness 错误归因给 retargeter。

#### 4. "Newton differentiable physics 直接回传梯度"：原 advice 说得太乐观

Newton 非常值得用，但第一版不能把"可微 humanoid contact rollout"设为关键依赖。当前 Isaac Lab–Newton integration 仍是积极开发中的 beta 路线，主力 GPU solver 是 MuJoCo Warp；而 MuJoCo Warp 当前本身并不提供我们可以稳定依赖的 differentiability。([Isaac Sim][4])

因此第一篇里 Newton 最合适的角色是：

* 大规模 batched rollout；
* PhysX 与 MJWarp 的 cross-solver verification；
* 检查 reference 是否依赖单一 simulator artifact；
* 生成 failure report；
* 在未来经过单独 spike 验证后，再把有限的 differentiable component 加进来。

这反而更稳。我们不需要把论文押在一个尚未稳定的梯度接口上。

#### 5. 程序化 morphology augmentation：必须做，但不能"随便随机 500 个机器人"

原 advice 认为机器人数量是瓶颈，这是对的。但如果独立随机 link length、mass、inertia、joint limit 和 torque，会制造出大量根本不物理一致的"假机器人"。模型最后可能学会 simulator/data generator 的怪异模式，而不是 morphology规律。

第一阶段只生成 **same-topology coherent variants**：

* link length 做相关缩放；
* mass 随体积与密度合理变化；
* inertia 根据质量与几何重新计算；
* COM 保留在 link 内；
* foot size、leg length、torso length 联动；
* actuator strength 和 velocity limit 在合理范围内变化；
* 所有 variant 必须通过 standing pose、joint range、self-collision 和基本 PD stability 检查。

同一 seed robot 的 joint names 和 topology 保持不变，这样可以直接复用 GMR 的配置生成 teacher data。等模型证明能处理连续 morphology variation，再加入"缺少 ankle roll""waist 从 3DoF 变 1DoF"之类的离散 topology change。

#### 6. Flow matching / diffusion：可能有用，但不能第一阶段就上

Retargeting 确实是多解问题，但在我们尚未证明：

* RobotSpec graph 有效；
* held-out robot 能泛化；
* dynamics features 没被忽略；
* verifier 能给出稳定信号；

之前就加入 flow matching，只会让失败更难归因。

第一版应使用 deterministic model。只有出现下面证据后，才升级到 flow：

* 同一 human motion–robot pair 存在多组明显不同、都可行的 Pareto solutions；
* deterministic regression 出现平均姿态、动作幅度衰减；
* best-of-\(N\) candidate 明显优于 single candidate；
* diversity 不只是随机 jitter，而是真正不同的 contact/time-warp strategy。

在那之前，一个小型 mixture head 或 4 个 latent proposal 就够了。

#### 7. 强物理验证不能放到"第二篇"

如果第一篇的 claim 是"physics-aware"或"trackable"，却只报告 foot skate、penetration 和 joint limit，reviewer 会立刻问：

> 为什么不直接放进 simulator 跑？

所以第一篇至少必须包含：

* 所有 robot 的 controller-independent verification；
* 核心 3–4 台 robot 的 strong rollout；
* 一个 cross-simulator subset；
* downstream tracker utility。

硬件可以是 optional final validation，但 strong simulated dynamics 不能推迟。

---

### 二、最终项目定义

#### Working title

**MorphoRetarget: RobotSpec-Conditioned, Physics-Verified Retargeting for Unseen Humanoids**

#### 核心研究问题

> 给定一段 human motion 和一份标准化 RobotSpec，一个没有 robot-specific output head、没有静态 robot prompt、没有针对新机器人重新训练的统一模型，能否为训练中完全未见过的 humanoid 生成语义保真且物理可跟踪的 motion？

形式化为：

$$
\hat Y_R
=
F_\theta(X_H,\mathcal R,\mathcal O),
$$

其中：

* \(X_H\)：human motion；
* \(\mathcal R\)：机器人标准化规格；
* \(\mathcal O\)：retargeting objective，例如优先保持 feet contact、hand trajectory 或动作节奏；
* \(\hat Y_R\)：root motion、variable-DoF joint trajectory、contact schedule、local time warp 和 feasibility confidence。

#### 第一篇的明确边界

为了让结果可解释，第一篇只做：

* humanoid；
* 约 19–32 body DoF；
* flat ground；
* foot-ground contact；
* 无物体 interaction；
* 不处理 dexterous fingers；
* 不输出 torque policy；
* 新机器人允许提供一个非常小的 semantic manifest；
* 不声称"raw URDF 零人工直接支持任意机器人"。

##### 不属于第一篇的内容

* humanoid → quadruped；
* arbitrary topology；
* object manipulation；
* hand-object contact；
* universal morphology-conditioned tracker；
* 大型 diffusion foundation model；
* 纯 simulator-gradient training。

这个收缩不是保守，而是在保护最核心的科学命题：**RobotSpec-conditioned zero-shot embodiment generalization**。

---

### 三、论文真正的三项贡献

#### Contribution 1：显式 RobotSpec-conditioned variable-DoF retargeter

与 robot ID、learned prompt 或 per-robot adapter 不同，模型读取：

* kinematic tree；
* joint axis 与 limits；
* link geometry；
* mass、COM、inertia；
* torque/velocity limits；
* PD gains、latency、control frequency；
* contact surfaces；
* coarse semantic anchors。

模型的输出也不是固定 29DoF vector，而是对每个 joint token 输出 trajectory，因此可以处理不同 joint 数量。

#### Contribution 2：真正的 leave-one-robot-out benchmark

不是随机留出 motion，而是完整留出：

* robot asset；
* 该 robot 的所有 paired labels；
* 该 robot 的 synthetic variants；
* 该 robot 的 learned prompt；
* 该 robot 的 adapter。

测试时只提供 RobotSpec 和 minimal semantic manifest。

此外加入 **dynamics twins**：

* geometry 与 kinematic tree 完全相同；
* 只改变 torque、velocity、latency、mass 或 COM；
* 检查模型是否真的读取 dynamics，而不是只识别 skeleton。

#### Contribution 3：Physics verification → localized repair → self-distillation

模型生成后不是笼统地说"物理更合理"，而是：

1. verifier 定位失败原因与时间区间；
2. repair 只修改该局部；
3. 保持区间外 trajectory 不变；
4. 成功修复结果回流训练；
5. 随迭代测量 repair invocation rate 是否下降。

这会把你们已经建立的 **Generate → Verify → Repair** 能力变成整个项目最有辨识度的部分。

---

### 四、从哪个仓库起步：最终决定

#### 不要 fork GMR 作为整个项目主仓库

应该新建一个独立仓库，例如：

```text
morphoret/
```

GMR、PHUMA、SPIDER、Holosoma 都作为外部 worker 或 pinned dependency。原因是它们分别绑定了不同 simulator、Python 环境、asset format 和 robot-specific assumptions。把所有代码塞进一个环境，后面会被依赖冲突和隐式格式转换拖住。

| 组件                | 在本项目中的角色                                                      | 是否作为主干         |
| ----------------- | ------------------------------------------------------------- | -------------- |
| GMR               | kinematic teacher、manual-config oracle baseline               | 否，外部 adapter   |
| HoloRetarget      | G1 inference-speed baseline                                   | 否              |
| PHUMA / PhySINK   | G1/H1-2 physics-constrained data、custom robot setup reference | 否              |
| SPIDER            | physics-refined teacher、sampling baseline                     | 否，且注意 CC BY-NC |
| Holosoma          | G1/T1 tracking verifier、multi-sim bridge                      | 否              |
| Isaac Lab / PhysX | 第一 strong rollout backend                                     | 否，独立环境         |
| Newton / MJWarp   | cross-solver verifier 与 batched rollout                       | 否，独立环境         |
| `refeas`          | L1 feasibility screen、failure taxonomy、已有 repair              | 是，作为内部模块接入     |
| MorphoRetarget    | schema、dataset、model、training、evaluation                      | **是**          |

GMR 当前非常适合作为第一 teacher：支持多种 humanoid 和 human motion format，而且 MIT license 也比较友好。HoloRetarget 当前则仍是固定 G1 29DoF 输出，更适合成为 G1 speed baseline，而非 universal architecture 的基础。

SPIDER 已经提供 G1/H1/T1 等 physics-based pipeline 和 MuJoCo Warp workflow，但其代码采用 CC BY-NC；因此不要复制进主仓库，保持为 optional external baseline。

PHUMA 可以直接提供 G1/H1-2 数据和 custom URDF/XML setup，但其部分原始 human motion 受上游许可限制，所以我们的公开 release 只发布转换脚本、manifest 和可依法发布的数据索引，不重新分发 SMPL 模型或受限原始 motion。

---

### 五、最先完成的不是模型，而是数据契约

这一点对你们尤其重要。你们现有 G1 corpus 已经遇到过三种很危险、但训练曲线表面完全可能"看起来正常"的 silent corruption：

* body order 错位；
* 多种 FPS 被当作一种；
* root height 使用相对值而没有正确还原。

现有库后来确认有 **10,822 clips / 43.6 hours**，并为 10,705 个 clips 建立了 physics feature atlas；这些经验说明这个项目的第一科学组件应该是**严谨的 motion/robot contract**，而不是马上写 Transformer。

#### 5.1 `HumanMotionSpec`

统一训练频率暂定 50 Hz，但始终保存原始 FPS 和 resampling provenance。

```text
HumanMotionSpec
├── sequence_id
├── source_dataset
├── source_fps
├── target_fps
├── coordinate_convention
├── human_height / shape
├── root_position[T,3]
├── root_rotation_6d[T,6]
├── root_linear_velocity_heading[T,3]
├── root_yaw_velocity[T,1]
├── body_local_position[T,B,3]
├── body_local_rotation_6d[T,B,6]
├── body_velocity[T,B,3]
├── contact_probability[T,C]
├── phase / event labels
├── confidence_mask[T,B]
└── source_hash
```

模型输入最好同时包含 joint rotations 与 body positions。只有 rotations 容易把 human proportions 混进动作；只有 positions 又会丢失局部方向信息。

#### 5.2 `RobotSpec v0.1`

RobotSpec 不应只是一个 URDF path，而应该是：

$$
\text{RobotSpec}
=
\text{Asset}
+
\text{Kinematics}
+
\text{Geometry}
+
\text{Dynamics}
+
\text{Actuation}
+
\text{Semantics}
+
\text{Runtime}.
$$

一个简化 schema 可以是：

```yaml
schema_version: morphoret.robot.v0.1

asset:
  source_format: urdf
  sha256: ...
  family: unitree_g1

frames:
  length_unit: meter
  up_axis: z
  forward_axis: x
  quaternion: wxyz

global:
  total_mass: ...
  standing_height: ...
  arm_span: ...
  control_dt: 0.02
  simulation_dt: 0.002

semantics:
  root_link: pelvis
  torso_link: torso
  head_link: head
  left_hand_link: left_wrist
  right_hand_link: right_wrist
  left_foot_link: left_ankle_roll
  right_foot_link: right_ankle_roll
  symmetry_pairs: [...]
  allowed_contact_links: [...]

contacts:
  left_sole_frame: ...
  right_sole_frame: ...
  foot_dimensions: ...

control:
  mode: position_pd
  latency_seconds: ...
  default_kp: [...]
  default_kd: [...]

links:
  - parent
  - local_transform
  - mass
  - center_of_mass
  - inertia
  - collision_proxy
  - semantic_role

joints:
  - parent_link
  - child_link
  - type
  - axis
  - lower_limit
  - upper_limit
  - velocity_limit
  - torque_limit
  - armature
  - damping
  - nominal_position
```

##### Minimal manual annotation允许什么？

允许用户第一次加入新机器人时确认：

* pelvis/root；
* torso；
* head；
* left/right hands；
* left/right feet 和 sole frames；
* symmetry；
* 哪些 links 可以接触环境。

不允许用户为每个 robot 再填写：

* 每个 keypoint 的 loss weight；
* 每个 motion 的 scale；
* 一整套人工 orientation offset；
* robot-specific model head；
* robot-specific prompt embedding；
* retraining data。

否则我们只是把 GMR config 换了一个名字。

#### 5.3 无量纲化

为了让模型学习比例规律：

* length 除以 standing height；
* mass 除以 total mass；
* inertia 除以 \(mL^2\)；
* torque 除以 \(mgL\)；
* velocity 根据 control rate 与 limb scale 标准化。

模型输入中不提供 robot name、file path 或 hash，也不使用绝对 joint index embedding，防止它偷偷记住 robot identity。

---

### 六、数据集设计

#### 6.1 Human motion 来源

第一阶段：

* **AMASS**：主要训练源，包含 40 小时以上、300 多位 subject 和 11,000 多段统一 SMPL motion；
* **LAFAN1**：OOD motion benchmark，约 4.6 小时、77 个长序列；
* **你们现有 10,822-clip G1 bank**：G1 anchor、failure regression 和 hard-case challenge；
* **PHUMA G1/H1-2**：physics-conditioned paired data；
* **Motion-X++**：只在第二阶段加入，先经过 curation，避免大规模噪声把问题淹没。([AMASS][5])

AMASS 与 Motion-X++ 不能仅按随机 window 划分。所有来自同一原始 sequence、subject 或派生 crop 的样本必须进入同一 split，防止相邻帧泄漏。

#### 6.2 第一批真实 robot

建议分成两层。

##### Tier A：必须有 strong physics rollout

* Unitree G1 29DoF；
* Unitree H1-2；
* Booster T1 29DoF；
* 再加入一台能稳定运行相同 tracking recipe 的 humanoid。

##### Tier B：先做 kinematic 与 L1 verification

* Unitree H1；
* Talos、GR-1、Kuavo 或 Berkeley Humanoid Lite 中选择资产最干净的两台。

MVP 可以先用 4 台，完整论文目标是 5–6 台真实 morphology。关键不在机器人数量本身，而在其中至少有三次完整的 **leave-one-real-robot-out**。

#### 6.3 程序化 variants

先对每个 training robot 生成 16 个 variant 做 pilot，之后根据 holdout performance 扩到每个 family 32–64 个，而不是一开始盲目做 500 个。

每个 clip 不与所有 variants 做笛卡尔积。可以让每个 human clip 随机覆盖 4–8 个 morphology，以更低成本得到均衡配对。

##### Variant 参数

```text
leg / arm length
torso height and width
foot length and width
link mass and COM
joint limits
motor torque limits
motor velocity limits
PD stiffness and damping
latency
control frequency
```

##### 生成规则

* 几何缩放与 mass/inertia 联动；
* T-pose 必须无自碰撞；
* feet 必须能落地；
* default PD 下至少能稳定站立；
* variant descriptor 使用 farthest-point sampling，避免生成大量近似重复机器人。

##### Holdout 防泄漏

如果 hold out T1：

* T1 asset 不参与训练；
* T1 的任何 teacher labels 不参与训练；
* T1 的所有 synthetic variants 不参与训练；
* 不允许把 T1 prompt 学出来；
* normalizer 不能单独使用 T1 数据拟合。

测试时模型只能读取 T1 RobotSpec 和 minimal semantic manifest。

---

### 七、Teacher 数据不是一个答案，而是一组候选

对于每个 `(human motion, robot)` pair，保存多个 candidate：

```text
Candidate A: raw GMR
Candidate B: GMR + contact projection
Candidate C: PHUMA / PhySINK
Candidate D: SPIDER refinement
Candidate E: local trajectory repair
Candidate F: time-warped / amplitude-adjusted alternatives
```

不要把这些 trajectory 直接求平均，因为不同解可能分别使用不同 joint/contact strategy，平均后反而不可执行。

每个 candidate 都保存：

* semantic metrics；
* joint/velocity limits；
* contact consistency；
* collision/penetration；
* inverse-dynamics torque margin；
* strong rollout result；
* teacher/version/asset hash；
* failure intervals；
* Pareto rank。

最终训练可以：

1. 对 clean teacher 做 supervised regression；
2. 对多个 candidates 做 preference/ranking；
3. 对多解 pair 使用 best-of-\(K\) 或 mixture objective；
4. 对 verifier 修复结果做 self-distillation。

---

### 八、模型架构

#### 8.1 总体结构

```text
                    ┌────────────────────────┐
Human Motion ──────▶│ Human Temporal Encoder │──────┐
                    └────────────────────────┘      │
                                                    ▼
                                            Cross Attention
                                                    ▲
                    ┌────────────────────────┐      │
RobotSpec Graph ───▶│ Robot Graph Encoder    │──────┘
                    └────────────────────────┘
                                                    │
                                  ┌─────────────────┴───────────────┐
                                  ▼                                 ▼
                         Variable-Joint Decoder              Global Heads
                                  │                                 │
                         q, qdot per joint             root/contact/timewarp
```

#### 8.2 Human encoder

建议第一版：

* 6 层 Transformer；
* hidden size 256；
* body-aware embedding；
* temporal positional encoding；
* 4 秒训练 window；
* 50 Hz；
* overlap inference；
* sequence boundary continuity loss。

参数规模控制在约 20–40M，足够表达问题，也适合你当前 16GB 级 GPU，不需要一开始训练大模型。

#### 8.3 Robot graph encoder

使用 **link–joint bipartite graph**：

* link token：geometry、mass、COM、inertia、semantic role；
* joint token：axis、limits、actuator、parent/child transform；
* graph edge：kinematic relation；
* global token：height、mass、control frequency、latency。

加入：

* kinematic depth；
* root distance；
* shortest-path attention bias；
* left/right symmetry embedding；
* contact-role embedding。

不能加入：

* robot ID；
* joint array absolute index；
* asset filename；
* robot-specific trainable embedding。

#### 8.4 Variable-DoF decoder

对每个时间 \(t\) 和 joint \(j\) 构造 token：

$$
z_{t,j}
=
\phi(H_t,R_j).
$$

然后使用 factorized attention：

1. 同一 joint 跨时间 temporal attention；
2. 同一时间沿 kinematic graph spatial attention；
3. global root/contact token 与所有 joint 交互。

这比将 `(T × J)` 全部做 full attention 更节省，也保留了 joint permutation equivariance。

输出：

```text
root heading velocity
root height
root orientation
joint position q[t,j]
joint velocity qdot[t,j]
left/right contact logits
local time-warp rate
feasibility confidence
optional repair uncertainty
```

对于 bounded revolute joint，模型预测归一化变量：

$$
q_{t,j}
=
q^{mid}_j
+
\tanh(u_{t,j})
\frac{q^{max}_j-q^{min}_j}{2},
$$

这样 joint limit 是参数化保证，而不只是一个可能被 loss 忽略的 penalty。

#### 8.5 Time warp 是 dynamics-aware retargeting 的关键变量

如果机器人 motor 比较弱，最合理的 adaptation 往往不是把姿态全部改掉，而是：

* 局部放慢；
* 调整 contact timing；
* 降低极端动作幅度；
* 更早开始重心转移。

因此让模型输出一个正的 phase rate：

$$
\dot{\tau}_t=\mathrm{softplus}(a_t),
$$

再积分得到 source phase 到 robot phase 的映射。语义误差按 phase alignment 计算，而不是简单要求同一 wall-clock frame 完全相同。

---

### 九、训练目标

第一版总损失可以写成：

$$
\mathcal L =
\lambda_{\text{sup}}\mathcal L_{\text{teacher}}
+
\lambda_{\text{fk}}\mathcal L_{\text{semantic}}
+
\lambda_{\text{contact}}\mathcal L_{\text{contact}}
+
\lambda_{\text{smooth}}\mathcal L_{\text{smooth}}
+
\lambda_{\text{geom}}\mathcal L_{\text{geometry}}
+
\lambda_{\text{rank}}\mathcal L_{\text{physics-rank}}
+
\lambda_{\text{cal}}\mathcal L_{\text{calibration}}.
$$

#### Teacher supervision

* joint position/velocity Huber loss；
* root velocity/orientation；
* contact BCE；
* time-warp supervision，仅对 physics-refined candidate 可用。

#### Semantic loss

不只比较 joint angle，而是比较：

* normalized body anchors；
* end-effector path；
* link orientation；
* pelvis/torso trajectory；
* motion phase；
* contact event。

#### Geometry loss

* foot penetration；
* self-collision proxy；
* ground clearance；
* joint acceleration和 jerk。

#### Physics ranking loss

给定两个 candidates \(Y_a,Y_b\)，simulator 判断 \(Y_a\) 更可执行，同时两者 semantic fidelity 相近，则训练：

$$
s_\psi(Y_a,\mathcal R)>s_\psi(Y_b,\mathcal R).
$$

这里的 \(s_\psi\) 是 learned feasibility critic。这样 generator 不需要直接通过 simulator 反向传播，也可以逐渐学到 physics preference。

#### Spec dropout 与 anti-memorization

训练时随机：

* 去掉部分 noncritical dynamics fields；
* 扰动 joint serialization order；
* mask robot family；
* mirror motion 与 RobotSpec；
* 对同 morphology 使用不同 asset name。

这样可以检验模型依赖的是物理参数还是隐式 robot identity。

---

### 十、分阶段训练，而不是直接 RL

#### Stage A：固定机器人 amortization baseline

先只做 G1：

$$
X_H \rightarrow Y_{G1}.
$$

目的不是论文贡献，而是确认：

* loader 正确；
* FK parity 正确；
* model 能拟合 teacher；
* full-sequence stitching 不产生 discontinuity；
* inference pipeline 能接 Isaac Lab。

这一阶段要同时实现一个简单 per-robot Transformer，后面它是 upper-bound baseline。

#### Stage B：Kinematic RobotSpec model

训练多个 robot + same-topology variants，但暂时不加入 mass、torque 等 dynamics，只输入：

* graph；
* link geometry；
* joint axis；
* limits；
* semantic anchors。

做第一次 leave-one-robot-out。

如果 graph model 连 robot-ID/nearest-prompt baseline 都不能战胜，说明 morphology distribution 或 representation 还不够，不应该进入 physics stage。

#### Stage C：Dynamics-aware model

加入：

* mass/inertia；
* torque/velocity limits；
* PD/latency；
* control frequency；
* time warp。

但不能继续只用 GMR label。对 dynamics twins 建立候选集：

```text
time scale: 0.70, 0.85, 1.00, 1.15, 1.30
root-height residual
motion-amplitude residual
contact advance/delay
selected-joint spline residual
```

在每个 dynamics spec 下运行 verifier，选出语义—可执行性 Pareto candidates，训练 critic 和 generator。

这一步是验证 dynamics conditioning 的核心。

#### Stage D：Simulator preference alignment

流程类似 best-of-\(N\) + self-training：

1. generator 生成 \(N=4\) 或 \(8\) 个 candidate；
2. cheap critic 和 L1 verifier筛选；
3. 少数进入 strong rollout；
4. simulator 返回 success 与 failure report；
5. 高质量 candidate 加入 replay；
6. 再做 preference learning / supervised fine-tuning。

这比对整个 sequence model 直接 PPO 稳定得多。

#### Stage E：Localized repair 与 distillation

只对失败片段修复，成功后蒸馏回 generator。

#### Stage F：可选的 generative model

只有当 deterministic model 的多解平均问题被实验确认后，再把 decoder 换成 conditional flow matching 或 small mixture generator。

---

### 十一、Physics verifier 的四层设计

#### L0：Schema 与数值一致性

必须自动检查：

* NaN/Inf；
* quaternion convention；
* coordinate frame；
* joint ordering；
* body ordering；
* source/target FPS；
* root absolute/relative position；
* units；
* asset/spec hash；
* FK parity。

必须有以下 property tests：

* 随机打乱 joint serialization，再映射回来，结果不变；
* URDF/MJCF 对同一姿态的关键 link FK 一致；
* mirror 两次恢复原 motion；
* resample 到目标 FPS 再还原时误差受控；
* PhysX 与 MuJoCo/Newton joint name mapping 显式验证。

#### L1：无 tracker 的快速物理筛选

检查：

* joint position/velocity/acceleration；
* self-collision；
* ground penetration；
* foot skate；
* contact mismatch；
* COM/support margin；
* inverse-dynamics torque；
* contact wrench feasibility；
* unsupported vertical wrench。

你们现有的 `refeas` 已经证明这条路线非常值得复用：screen 大约每 clip 1 个 CPU 秒，contact-projection repair 大约每个 recoverable clip 3 秒；在 2,443 个 flagged clips 上，约 **65.8% 可自动恢复**。这不是一个概念组件，而是已经有真实规模证据的基础设施。

#### L2：冻结 tracker 的 strong rollout

每个 robot 使用相同 tracker architecture 与 training recipe，冻结后评估所有 retargeters。

输出：

* full-clip survival；
* fall time；
* normalized link tracking error；
* root tracking error；
* contact timing error；
* foot slip；
* torque saturation；
* velocity saturation；
* energy；
* robustness under friction/mass/delay randomization。

#### L3：Downstream data utility

分别使用：

* GMR data；
* GMR + repair；
* neural output；
* neural + repair；
* physics optimizer output；

训练完全相同的 tracking policy，比较：

* sample efficiency；
* final success；
* OOD motion；
* perturbation robustness；
* training stability；
* failure-category distribution。

这会回答最实在的问题：

> 我们生成的数据是否真的比 GMR 更适合作为 robot learning data？

---

### 十二、Failure-localized repair

Verifier 应输出结构化报告：

```json
{
  "passed": false,
  "failure_intervals": [
    {
      "start_frame": 142,
      "end_frame": 176,
      "type": "torque_deficit",
      "severity": 0.81,
      "links": ["left_foot", "pelvis"],
      "joints": ["left_knee", "left_hip_pitch"]
    }
  ]
}
```

#### Repair operator library

| Failure type                    | 首选 operator                       |
| ------------------------------- | --------------------------------- |
| root floating / foot mismatch   | 已有 contact projection             |
| joint/velocity limit            | local joint spline                |
| torque deficit                  | local time warp + root adjustment |
| premature contact               | contact timing shift              |
| self-collision                  | selected limb residual            |
| unstable COM                    | pelvis/root residual              |
| genuine ballistic impossibility | refuse / reject，而不是硬修             |

Repair parameterization 不直接修改每一帧，而是修改少量 B-spline knots：

$$
a=
\{\Delta q_{\mathrm{knot}},
\Delta x_{\mathrm{root}},
\Delta \tau,
\Delta c\}.
$$

首先使用 CEM、MPPI 或 sampling 搜索，而不是 PPO。因为 action dimension 很低、horizon 很短、成功条件明确，这比完整 trajectory RL 更适合。

#### RL 放在哪里才合理？

等积累了足够多的 repair tuples 后，再定义一个 3–5 步的 repair MDP：

* state：human context、RobotSpec、failed candidate、failure report；
* action：operator choice 与 spline parameter delta；
* transition：局部 rollout；
* reward：首先通过 hard constraints，其次保持 semantic，最后最小化修改。

RL 学的是**如何提出修复**，而不是从零生成整个 motion，也不是直接控制 torque。

#### Self-distillation

每个成功 tuple：

$$
(X_H,\mathcal R,Y_{\text{failed}},V,Y_{\text{repaired}})
$$

加入 replay。之后测量：

* 第一次生成 pass rate；
* average repair iterations；
* strong verifier calls per clip；
* repair invocation rate；
* unrecoverable rate。

如果模型真的学会了 repair knowledge，后几轮训练后 repair invocation rate 应持续下降。

---

### 十三、实验设计

#### E0：Data contract 与 FK parity

这是进入任何模型实验前的硬门槛。

* GMR/MuJoCo；
* canonical schema；
* Isaac Lab/PhysX；
* Newton/MJWarp；

对同一 robot pose 的 link transforms、root frame、joint ordering 必须一致。

#### E1：G1 amortization

比较：

* GMR；
* HoloRetarget；
* per-robot Transformer；
* MorphoRetarget 固定 G1 模式。

这里只验证工程正确性与 speed-quality curve，不作为主要 novelty。

#### E2：Seen motion / seen robot

确认 full model 没有因为共享参数显著损害普通 performance。

#### E3：Unseen motion / seen robot

使用 LAFAN1、dynamic category 和你们已有 hard set。

#### E4：Seen motion / unseen robot

完整留出一台 robot，验证 RobotSpec generalization。

#### E5：Unseen motion / unseen robot

这是 primary evaluation：

* 新 human sequence；
* 新 robot；
* 无 adapter；
* 无 prompt optimization；
* 无 labels；
* 无 fine-tuning。

#### E6：Dynamics twins

保持 geometry 和 topology 相同，只修改：

* torque 50% / 75% / 100% / 125%；
* velocity limit；
* latency；
* total mass；
* pelvis COM。

比较：

* kinematic-only model；
* full RobotSpec model；
* full model without time warp；
* oracle physics refinement。

如果 full model 对 50% torque 与 125% torque 产生完全相同输出，说明 dynamics feature 被忽略，核心 claim 不成立。

#### E7：Verifier、repair 与 self-distillation

报告：

* raw generator pass rate；
* + candidate ranking；
* + specialized repair；
* + learned repair；
* + self-distillation round 1/2/3。

#### E8：Cross-simulator verification

在相同 initial state、reference 和 controller 下比较 PhysX 与 Newton/MJWarp：

* pass/fail agreement；
* failure time；
* ranking consistency；
* 哪类 motion 对 solver 最敏感。

Cross-solver disagreement 本身也要报告，而不是隐藏。

#### E9：Downstream tracker utility

同样的 tracker、同样的 seeds、同样训练预算，仅改变 motion data source。

#### E10：安全硬件 subset

最后从 strong rollout 中选择：

* low velocity；
* low impact；
* no jump；
* high torque margin；
* multi-randomization pass；

在 G1 上验证一小组 motion。硬件实验是加分项，不是前面所有结论的唯一支撑。

---

### 十四、Baselines 与 ablations

#### 外部 baselines

* GMR manual config；
* PHUMA / PhySINK；
* SPIDER；
* HoloRetarget，仅 G1 speed；
* per-robot NMR-style Transformer；
* G-DReaM-style kinematic graph baseline，在可复现范围内。

#### 最关键的内部 ablation chain

Per-robot model → + shared backbone with robot ID → + kinematic graph → + geometry → + mass/inertia
→ + actuator/control → + physics critic → + local repair → + repair distillation.

还要有：

* no semantic anchors；
* no contact head；
* no time warp；
* no morphology augmentation；
* random versus coherent variants；
* random versus stratified motion sampling；
* robot joint-order permutation test。

你们内部已经有 SNMR——一个由 classical IK teacher 蒸馏、共享 latent 服务五种 humanoid 的 retargeting network。因此新项目不应该再把"shared latent across robots"写成主贡献；它应作为 internal baseline，而新贡献放在显式 RobotSpec、held-out embodiment 和 dynamics intervention 上。

---

### 十五、指标与统计协议

#### 语义指标

* normalized keypoint error；
* link orientation geodesic error；
* root path error；
* end-effector path；
* contact F1；
* phase-aligned motion error；
* motion amplitude retention。

#### 运动学与几何指标

* joint-limit violation；
* velocity/acceleration violation；
* self-collision rate；
* ground penetration；
* foot skate；
* jerk。

#### 动态指标

* full-clip survival；
* fall rate/time；
* tracking error；
* torque margin；
* saturation fraction；
* robust success under randomization；
* energy；
* contact impulse。

#### 系统指标

* clips/s；
* latency；
* batch throughput；
* new robot annotation fields；
* new robot setup time；
* simulator rollouts per successful clip；
* repair invocation rate。

#### 统计单位

实验单位必须是 **clip**，不是 frame。

* 对 clips 做 paired bootstrap 95% CI；
* 每台 robot 单独报告；
* robot macro-average，不能让 G1 大数据量淹没其他 robot；
* tracker training 至少 3 seeds；
* retargeter核心结果也运行多个 seeds；
* failure category 做分层结果；
* primary metric 在 full run 前冻结。

推荐的 primary claim 形式不是"我们的总分更高"，而是：

> 在 held-out robots 上，MorphoRetarget 相对 GMR 达到 semantic non-inferiority，同时显著提高 robust full-clip tracking success，且无需 robot-specific retargeting tuning。

---

### 十六、仓库结构

```text
morphoret/
├── pyproject.toml
├── README.md
├── LICENSE
├── envs/
│   ├── gmr.lock
│   ├── phuma.lock
│   ├── spider.lock
│   ├── train.lock
│   ├── isaaclab.lock
│   └── newton.lock
├── third_party/
│   └── pinned_revisions.yaml
├── morphoret/
│   ├── spec/        (schema, parse_urdf, parse_mjcf, parse_usd, semantics, normalize, variants)
│   ├── human/       (smplx, bvh, canonicalize, resample, contacts)
│   ├── teachers/    (gmr_adapter, phuma_adapter, spider_adapter, holoretarget_adapter)
│   ├── data/        (records, build_pairs, candidates, provenance, splits, shards)
│   ├── models/      (human_encoder, robot_graph_encoder, factorized_decoder, heads, feasibility_critic)
│   ├── losses/      (supervised, semantic, contact, geometry, smoothness, ranking)
│   ├── verify/      (schema_checks, refeas_adapter, isaaclab_physx, newton_mjwarp, tracker_protocol, reports)
│   ├── repair/      (failure_localizer, parameterization, contact_projection, cem, proposal_network, distill)
│   └── eval/        (metrics, leave_one_robot_out, dynamics_twins, cross_solver, downstream_tracking)
├── configs/  (robots, datasets, teachers, models, experiments)
├── scripts/
└── tests/    (quaternion conventions, joint order, permutation equivariance, fk parity, resampling, root height, mirror, asset hash)
```

不同环境之间只交换 JSON / NPZ / Parquet / HDF5；不跨环境 import simulator internals。每个 artifact 保存
human_source_hash, asset_hash, robot_spec_hash, teacher_commit, teacher_config_hash, simulator_build,
tracker_checkpoint_hash, split_version, verification_version.

---

### 十七、Gate-driven 执行计划

这些数字是**进入下一阶段的工程决策阈值**，不是预先声称的论文结果。

| Gate            | 必须证明的事情                  | 建议门槛                                                           |
| --------------- | ------------------------ | -------------------------------------------------------------- |
| G0 Contract     | 所有 frame/joint/FK 一致     | FK parity、order、FPS、root tests 全过                              |
| G1 Amortization | neural baseline 能替代 GMR  | semantic error 距 GMR 不超过约 5%，violation 不增加                     |
| G2 Embodiment   | graph condition 真能泛化     | held-out robot 显著胜过 ID/nearest-prompt，并达到 GMR 大部分 trackability |
| G3 Dynamics     | model 真读取 motor/dynamics | torque twin 上 saturation/fall 降低，semantic degradation 受控       |
| G4 Repair       | 局部修复有实际价值                | strong success 增加至少约 10 个百分点，语义损失低于约 5%                        |
| G5 Utility      | 输出是更好的训练数据               | 相同 tracker recipe 下匹配或超过 GMR data                              |
| G6 Cross-sim    | 结果不依赖一个 solver           | 排名大体一致，分歧有可解释 taxonomy                                         |

##### Kill criteria

* graph model不比 robot-ID/nearest prompt好：说明 morphology data 或 representation 不够；
* dynamics twins 下输出完全不变：说明 dynamics conditioning 失败；
* 新 robot 仍需大规模人工 weight tuning：zero-shot claim 失败；
* L1 指标很好但 tracker 大面积失败：先诊断 tracker 或缺失 dynamics；
* repair 总把动作改成保守站立：semantic objective 与拒绝机制失效；
* cross-solver 排名完全反转：不能直接做 simulator-grounded claim。

---

### 十八、第一轮最有信息量的实验

不要先跑完整 AMASS。第一轮就做一个能杀死或支持核心假设的实验：

Train robots: G1, H1, H1-2, each with 8 coherent variants. Held-out robot: T1 29DoF.
Human motions: 100 stratified clips (walk, turn, run, squat, crouch, kick, dance, jump, asymmetric
arm motion, recovery-like motion).

对比模型: (1) GMR-T1 manual oracle; (2) T1-specific supervised model, upper bound; (3) shared model +
robot ID; (4) nearest training robot prompt; (5) kinematic RobotSpec graph; (6) full RobotSpec graph;
(7) full model + physics critic.

第一轮必须回答的四个问题:
A. 新 T1 只给 RobotSpec，能否输出合法 trajectory？如果不能，先修 schema、graph 或 training diversity。
B. Kinematic graph 是否胜过 nearest-prompt？如果不能，暂时不碰 dynamics。
C. 将 T1 torque 降到 50%，full model 是否改变 time warp 或姿态？如果不改变，说明 dynamics information 没有进入输出。
D. Physics verifier 是否能定位具体失败区间，而不是只给 clip-level fail？如果不能，repair loop 无法成立。

---

### 十九、最终建议

这个项目应当**从 GMR 开始，但不能建在 GMR 里面**。

> **GMR 提供广覆盖的 kinematic prior；PHUMA/SPIDER 提供小规模 physics-quality supervision；你们现有 `refeas` 与 contact projection 提供高性价比的 failure detection/repair；Isaac Lab/PhysX 提供第一 strong verifier；Newton/MJWarp 提供跨 solver 验证；一个新的 variable-DoF RobotSpec-conditioned model 负责真正的 zero-shot generalization。**

**第一条开发分支不应该叫 `train_transformer`，而应该叫：**

```text
robot-spec-contract-and-fk-parity
```

当这个基础完全可靠后，第一篇论文的主轴就保持一句话不动：

> **A single retargeter reads an unseen humanoid's physical specification, rather than its identity, and produces motion that survives independent physics verification.**

[1]: https://github.com/YanjieZe/GMR
[2]: https://arxiv.org/html/2601.07284v2
[3]: https://github.com/facebookresearch/spider
[4]: https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html
[5]: https://amass.is.tue.mpg.de/
