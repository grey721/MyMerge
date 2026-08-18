# AgentMerge

面向固定 Agent 配置与提示词的多 Agent LoRA 融合搜索框架。每个 Agent 从
同一个全局 adapter 池中获得独立的稀疏融合策略；系数既可按整个 adapter，
也可按层、层组、模块类型或“层×模块类型”表达，搜索器以整个 Agent 系统的
端到端指标进行多目标优化。

当前版本实现：

- 类型化 YAML/JSON 配置与一致性校验；
- 固定 DAG 拓扑，以及可选的共享 scratchpad 工具循环；
- 可替换的模型后端、benchmark、评分器和优化器接口；
- Mock、Transformers + PEFT，以及可选的 vLLM CausalLM 后端；
- 支持不同 LoRA rank 的候选级按需临时融合，并将融合系数直接烘焙进
  rank-sum PEFT Adapter；
- 按拓扑阶段执行的 benchmark micro-batching，PEFT 后端按融合向量批量生成；
- 非负、归一化、可变 top-k 的可插拔系数表示与搜索空间；
- 包级解耦的 NSGA-II、MOEA/D-DE、固定/自适应分辨率搜索、优化器注册表、共享评价缓存和并行评价；
- JSONL 系统评价、质量/最差组性能/实测 token/延迟/融合秩代理指标；
- 每代 checkpoint、最终种群与 Pareto 前沿导出；
- 一张 GPU 一个常驻模型副本的候选级多卡并行评价。

## 路径与文件职责

当前工作区和公共模型资源的实际位置如下。配置文件中的相对路径始终相对于
该配置文件自身解析，不依赖执行命令时所在的目录。

```text
/202422801010/
├── models/
│   └── Qwen3-8B/                 # 公共基础模型，只读共享，不属于本项目源码
└── AgentMerge/                   # 项目根目录
    ├── experiments/              # 模型、Agent 系统和 benchmark YAML
    ├── data/                     # 小型工程 benchmark
    ├── scripts/                  # 数据构建和模型下载辅助脚本
    ├── src/agentmerge/           # Python 主包
    │   ├── core/                 # 配置、FusionPlan 与共享领域对象
    │   ├── optimizers/           # NSGA-II、MOEA/D-DE、双阶段算法与注册
    │   ├── evaluation/           # benchmark、Agent 系统、评价缓存与诊断
    │   ├── runtime/              # 实验生命周期与多 GPU 调度
    │   └── backends/             # Mock/PEFT/vLLM、物化和 Adapter 审计
    ├── tests/                    # 单元测试和集成测试
    ├── references/               # 论文、开源仓库、Adapter 池和调研记录
    ├── outputs/                  # 自动生成的实验结果，不是源码
    ├── pyproject.toml            # 包元数据、依赖、CLI 和 pytest 配置
    ├── setup.py                  # setuptools 安装入口
    └── README.md                 # 本文档
```

真实 Qwen3-8B 路径是
`/202422801010/models/Qwen3-8B`，对应配置中的
`../../models/Qwen3-8B`。`.conda/`、`.pytest_cache/`、`__pycache__/` 和
`src/agentmerge.egg-info/` 都是本地环境或安装生成物，不承载实验逻辑。

### 核心源码

| 路径 | 作用 |
|---|---|
| `src/agentmerge/__init__.py` | 对外公开的 Python API 与版本号。 |
| `src/agentmerge/cli.py` | `agentmerge validate/search/audit/smoke-real/admit-adapters` 命令入口。 |
| `src/agentmerge/core/config.py` | 当前 YAML/JSON schema、路径解析、类型转换和严格字段校验。 |
| `src/agentmerge/core/domain.py` | 融合权重、生成请求/结果、评价结果和 Agent trace 等共享数据对象。 |
| `src/agentmerge/core/policies.py` | 算法无关的 `FusionPlan`、融合策略和模块粒度定位。 |
| `src/agentmerge/evaluation/system.py` | 固定 DAG 与共享 scratchpad 工具循环、阶段式批量执行、工具环境注入和统一系统构建。 |
| `src/agentmerge/evaluation/benchmarks.py` | JSONL/synthetic benchmark、评分器、候选系统评价和运行时指标。 |
| `src/agentmerge/evaluation/code_worker.py` | HumanEval/MBPP 生成代码的隔离子进程、AST 限制、资源上限和基础测试执行。 |
| `src/agentmerge/evaluation/cache.py` | 以 `FusionPlan` 内容寻址的评价去重、缓存和批量调度。 |
| `src/agentmerge/evaluation/admission.py` | base、全局 one-hot 和可选 role-wise Adapter 准入评价。 |
| `src/agentmerge/evaluation/diagnostics.py` | 真实 PEFT/vLLM 后端的加载、one-hot、融合与释放冒烟诊断。 |
| `src/agentmerge/runtime/distributed.py` | 一卡一进程的常驻模型、多候选异步多 GPU 调度。 |
| `src/agentmerge/runtime/lifecycle.py` | TERM/HUP 清理退出、Linux 父进程死亡联动与 Worker 生命周期保险。 |
| `src/agentmerge/runtime/experiment.py` | 组装组件，管理搜索生命周期、事件和结果落盘。 |

### 后端与优化器

| 路径 | 作用 |
|---|---|
| `src/agentmerge/backends/base.py` | 模型推理后端抽象和候选 session 接口。 |
| `src/agentmerge/backends/algebra.py` | 后端无关的融合代数接口、`exact_delta` / `factor_linear` 实现和不可变执行计划。 |
| `src/agentmerge/backends/resources.py` | 源/融合 Adapter 资源与后端融合工厂的统一生命周期协议。 |
| `src/agentmerge/backends/pool.py` | 稳定 Adapter key、阶段坐标索引、引用计数释放和常驻池状态机。 |
| `src/agentmerge/backends/peft_pool.py` | PEFT 私有源 Adapter 资源对象及可逐个释放的常驻池。 |
| `src/agentmerge/backends/mock.py` | 无需 GPU 的确定性测试后端。 |
| `src/agentmerge/backends/peft.py` | Transformers + PEFT 模型加载、源 Adapter 常驻、分词和批量生成。 |
| `src/agentmerge/backends/materialization.py` | 候选级临时 Adapter 按需物化、复用、统计、异常清理与释放。 |
| `src/agentmerge/backends/vllm/` | 可选 vLLM 0.10 后端、常驻源 tensor、内存 LoRA 注册和候选生命周期。 |
| `src/agentmerge/backends/factory.py` | Mock、PEFT 和 vLLM 后端的统一构建入口。 |
| `src/agentmerge/backends/audit.py` | 不加载基础模型权重的 Adapter 池兼容性和元数据预检。 |
| `src/agentmerge/optimizers/base.py` | `ask/tell`、搜索结果、observer 接口及双阶段日志坐标适配。 |
| `src/agentmerge/optimizers/pareto.py` | 支配关系、非支配排序和拥挤距离。 |
| `src/agentmerge/optimizers/registry.py` | 优化器注册与按配置构建。 |
| `src/agentmerge/optimizers/contracts.py` | 算法无关的实向量搜索空间、Genome 和 Translator 接口。 |
| `src/agentmerge/optimizers/genome.py` | 全局/层级融合染色体及通用个体数据结构。 |
| `src/agentmerge/optimizers/representation.py` | 染色体编解码、融合粒度和约束 repair。 |
| `src/agentmerge/optimizers/translation.py` | 全局/层级染色体到 `FusionPlan` 的独立翻译器，以及共享阶段投影。 |
| `src/agentmerge/evaluation/metrics.py` | benchmark 指标目录、任务级质量聚合与可选目标名称。 |
| `src/agentmerge/optimizers/fitness.py` | 将任意评价指标编码为优化器统一使用的最小化 fitness，并负责校验与结果还原。 |
| `src/agentmerge/optimizers/initialization.py` | 根据 Adapter 准入元数据构造共享初始锚点。 |
| `src/agentmerge/optimizers/nsga2/config.py` | NSGA-II 专属配置解析和校验。 |
| `src/agentmerge/optimizers/nsga2/components.py` | 向组合算法公开一个完整单阶段 NSGA-II 的装配边界。 |
| `src/agentmerge/optimizers/nsga2/variation.py` | NSGA-II 专属实向量交叉、连续变异和稀疏变异。 |
| `src/agentmerge/optimizers/nsga2/optimizer.py` | NSGA-II 的采样、选择、交叉、变异和环境选择。 |
| `src/agentmerge/optimizers/nsga2/__init__.py` | NSGA-II 包出口及构建函数。 |
| `src/agentmerge/optimizers/two_stage_nsga2/` | 独立的 Two-stage NSGA-II 包，负责全局→层级编排、精英投影和 Translator 切换。 |
| `src/agentmerge/optimizers/de/` | 单阶段分解式多目标 DE；分解、参考方向、差分算子和种群更新相互独立。 |
| `src/agentmerge/optimizers/two_stage_de/` | Adapter→Layer Two-stage DE，负责 Pareto anchor 投影与阶段切换。 |
| `src/agentmerge/optimizers/progressive_de/` | 固定 Adapter 池上的递归层分区、阶段投影和逐步自由度释放 MOEA/D-DE。 |
| `src/agentmerge/optimizers/adaptive_resolution_de/` | 固定嵌套层分区、successful-DE 方向监控、边际效用与严格 FE reserve 驱动的自适应自由度释放。 |

### 配置、数据与脚本

| 路径 | 作用 |
|---|---|
| `experiments/smoke.yaml` | 单进程 synthetic benchmark 的快速全链路测试。 |
| `experiments/multigpu_mock.yaml` | 无 CUDA 的多进程调度链路测试。 |
| `experiments/qwen3_peft_template.yaml` | 新建 Qwen3-8B PEFT 搜索实验的模板，Adapter 仓库标识需要替换或核验。 |
| `experiments/qwen3_pool_v01.yaml` | 全局候选 Adapter 池的准入配置。 |
| `experiments/qwen3_pilot_v01.yaml` | 已筛选 Adapter 子池的多目标试运行配置。 |
| `experiments/qwen3_de_pool_research_v01.yaml` | 64 例 search split 上的普通 DE/个性化池消融配置。 |
| `experiments/qwen3_vllm_research_v01.yaml` | 相同模型、Agent、Adapter 池和数据的可选 vLLM 后端配置。 |
| `experiments/qwen3_vllm_factor_linear_research_v02.yaml` | 224 例、8 任务组的困难多域搜索配置，覆盖数学、代码、逻辑、规划、游戏和多跳 QA。 |
| `experiments/qwen3_vllm_factor_linear_research_v03.yaml` | 默认 loop 论文对标配置；共享 scratchpad 上分离 planner/arguments/final。默认优化语言推理、数学和代码三类目标；BFCL 仍参与评价并暴露指标，但不是默认优化目标。 |
| `data/demo.jsonl` | 通用 JSONL 格式示例。 |
| `data/admission_v01.jsonl` | 固定 16 例的 MATH-500 + HotpotQA 工程准入集。 |
| `data/research_v01_{search,validation,test}.jsonl` | 互斥的 64/128/256 例 MATH-500 + HotpotQA 研究切分。 |
| `data/research_v01_manifest.json` | 固定选择 seed、来源窗口和三个切分的 case ID。 |
| `data/research_v02_{search,validation,test}.jsonl` | 互斥的 224/320/576 例困难多域研究切分，每个切分内 8 个任务组严格平衡。 |
| `data/research_v02_manifest.json` | v02 的论文对标、来源、难度过滤、生成规则和全部 case ID。 |
| `data/research_v03_{search,validation,test}.jsonl` | 互斥的 384/648/1152 例文献常用任务切分，语言推理仅使用 6-NLI；MBPP 公开目标函数名和首条测试示例，但使用全部测试评分；每个切分内四个目标领域严格等量。 |
| `data/research_v03_manifest.json` | v03 的来源、评价协议、领域映射、分组计数与全部 case ID。 |
| `data/mock_multigpu.jsonl` | 多进程 Mock 评价输入。 |
| `scripts/build_admission_v01.py` | 按固定公开数据 ID 重建 `admission_v01.jsonl`。 |
| `scripts/build_research_v01.py` | 排除准入题后确定性重建 research_v01 三个切分。 |
| `scripts/build_research_v02.py` | 确定性重建 v02；远程数据分页缓存，规划/Sudoku/Sokoban 本地生成并由确定性执行器评分。 |
| `scripts/build_research_v03.py` | 确定性重建 v03；公开数据分页缓存，并读取本地固定版本的 BFCL 数据。 |
| `scripts/download_hf_shards.py` | 支持 HTTP Range、并行分块和 SHA-256 校验的模型分片下载器。 |

### 参考资料、测试与输出

| 路径 | 作用 |
|---|---|
| `references/papers/` | 本地论文 PDF，仅用于调研。 |
| `references/codes/` | 克隆的论文开源仓库，不参与 AgentMerge 包导入或运行。 |
| `references/codes/BIG-Bench-Hard/` | BBH 官方任务数据与提示词仓库。 |
| `references/codes/evalplus/` | HumanEval/MBPP 评价实现参考；当前基础测试由项目内隔离子进程执行。 |
| `references/codes/ImPart/` | ImPart 开源实现；MBPP 通过 EvalPlus 执行，提示中包含公开 assert。 |
| `references/codes/LED-Merging/` | LED-Merging 开源实现；其 MBPP 生成提示显式包含函数测试示例。 |
| `references/codes/gorilla/` | BFCL 官方仓库；v03 使用 V4 的 `simple_python` 与 `live_simple` 静态子集。 |
| `references/codes/LRAgent/` | 共享 scratchpad、规划/参数角色分离和工具 Observation 回灌的设计参考；不作为运行时依赖。 |
| `references/open_source_model_merging_papers.csv` | 开源模型融合论文和仓库索引。 |
| `references/model_merging_reuse_review.md` | 可借鉴算法与工程实现的调研记录。 |
| `references/engineering_backlog.md` | 尚未采纳或待验证的工程想法。 |
| `references/adapters/qwen3-8b/pool_v0.1.yaml` | Qwen3-8B 候选 Adapter 池清单和筛选状态。 |
| `references/adapters/qwen3-8b/metadata/` | 已下载 Adapter 的配置、权重和来源元数据；`reserve_*` 仅为候补。 |
| `references/adapters/qwen3-8b/admission_*.md` | Adapter 准入方法与结果记录。 |
| `tests/` | 配置、拓扑、评价、融合正确性、异常清理、优化器和多 GPU 调度测试。 |
| `outputs/<experiment>-<timestamp>/` | 每次运行独立生成的配置快照、事件、评价、checkpoint 和 Pareto 结果。 |

`references/` 中的论文代码和记录不是运行时依赖；真正参与实验的基础模型、
Adapter 路径和 benchmark 必须以所运行的 `experiments/*.yaml` 为准。

## 快速开始

基础软件链路不需要 Transformers、PEFT 或 GPU。安装核心包与测试依赖后可以
运行完整的合成冒烟搜索：

```bash
pip install -e '.[test]'
agentmerge validate experiments/smoke.yaml
agentmerge search experiments/smoke.yaml \
  --algorithm nsga2 \
  --search_granularity global \
  --populations 8 \
  --generations 2
pytest
```

合成 benchmark 只用于验证软件和演化算法，不能作为论文结果。搜索产物写到
`outputs/<experiment>-<UTC timestamp>/`，包括：

- `resolved_config.yaml`
- `output_manifest.json`（Agent/Adapter 轴、指标顺序和输出安全策略；不含模型或 Adapter 权重）
- `runtime.json`
- `run.log`（人类可读日志；代际使用 `GEN`，个体仅输出精简的 `CAND` 完成行）
- `events.jsonl`（机器可读的候选提交/完成、物化/释放统计、代际、checkpoint、失败等事件）
- `history.jsonl`
- `evaluations/<genome_id>.json`（内容寻址、每个候选只保存一次；仅保存 genotype、指标、聚合统计和运行统计）
- `checkpoints/generation_*.json`
- `pareto_front.json`（仅保存 Pareto 候选引用、rank 和拥挤距离）
- `final_population.json`（仅保存最终种群候选引用、rank 和拥挤距离）
- `resolution_trajectory.json`（仅自适应分辨率算法生成；记录各阶段起止 FE、维度和 promotion 原因）

输出层明确禁止保存模型权重、Adapter 权重、逐题 prediction、逐题 target 和完整问题答案。
评价器只保存分组分数、分数分布、输出长度/token 统计、生成上限触达次数和重复输出次数。
因此，若模型生成了连续的 `0` 或其他重复文本，结果文件会标记异常统计，但不会写入整段文本。

第二版 checkpoint 不再重复嵌入每个候选的逐样本预测，只保存
`genome_id + evaluation_ref + rank + crowding_distance`。完整评价位于
`evaluations/`，因此同一候选跨代保留时不会重复写入大块结果。

### YAML 与 CLI 的职责

`experiments/*.yaml` 固化基础模型、Adapter 池、Agent 提示词与拓扑、benchmark
和默认目标。优化器、目标覆盖、搜索层级、预算和演化算子均可由 CLI 指定：

```bash
agentmerge search experiments/qwen3_pilot_v01.yaml \
  --run-name pilot-seed-7 \
  --algorithm nsga2 \
  --search_granularity global \
  --seed 7 \
  --populations 24 \
  --generations 10 \
  --device 1 2 4 5 \
  --batch_size 8 \
  --mutation_probability 0.2
```

每个 JSONL `group` 会自动成为 `task_quality.<group>` 指标。重复使用
`--objective METRIC:DIRECTION` 可在不复制 YAML 的情况下替换整组目标，例如：

```bash
agentmerge search experiments/qwen3_vllm_factor_linear_research_v01.yaml \
  --objective task_quality.math500:maximize \
  --objective task_quality.hotpotqa:maximize \
  --objective fusion_rank_proxy:minimize \
  --algorithm de \
  --search_granularity adapter \
  --populations 12 \
  --generations 20 \
  --device 0 1 2 3 4 5
```

`maximize/minimize` 可简写为 `max/min`。`agentmerge validate` 会列出
`available_metrics`，并在加载基础模型前拒绝不存在的任务或指标。

Agent 执行方式也属于 YAML 固化的系统定义。省略 `kind` 时默认使用
`tool_loop`；固定 DAG 是需要显式声明 `kind: fixed` 的可选执行方式。共享
scratchpad 循环可写为：

```yaml
agents:
  - name: planner
    role: planner
    system_prompt: Select the next action or tool.
    prompt_template: "Task:\n{task}\n\nShared scratchpad:\n{scratchpad}"
  - name: arguments
    role: argument generator
    system_prompt: Generate arguments for the selected action.
    prompt_template: "Task:\n{task}\n\nShared scratchpad:\n{scratchpad}"
  - name: final
    role: final answerer
    system_prompt: Produce the final answer from the shared scratchpad.
    prompt_template: "Task:\n{task}\n\nShared scratchpad:\n{scratchpad}"

topology:
  loop_agents: [planner, arguments]
  max_steps: 2
  max_context_tokens: 7680
  output_agent: final
```

每轮按 `planner → arguments → ToolEnvironment` 执行，Observation 回写后再进入
下一轮；最后由 `output_agent` 作答。默认 `MetadataToolEnvironment` 从 JSONL
样本的 `tool_results` 元数据回放结果。真实检索器、代码执行器或任务环境应通过
`ToolEnvironment` 注入，因此该执行方式不依赖 BFCL。循环内部仍按角色批量处理
同一 micro-batch，候选 Adapter 也仍只在整个候选评价 session 内物化一次。
Planner 产生 `Finish` 或 `Finish[...]` 时停止继续调用工具；超过 `max_steps`
或 prompt 超过 `max_context_tokens` 时分别记录 `max_steps` / `context_limit`；
空输出和 `No response` 记录为 `generation_error`。工具异常不会终止整个样本，
而会作为错误 Observation 回灌，供下一步修正。这些状态只保存聚合计数，不保存
逐题回答。

多个任务也可在 benchmark 的 `options.quality_domains` 中按领域分组。评价器会
先计算任务均值，再对领域内任务做宏平均，并暴露为
`domain_quality.<domain>`。默认的 `research_v03` 优化语言推理、数学和代码
三个质量目标；工具调用指标仍会计算，但需要用 CLI 显式选择才成为目标。具体
映射保存在该实验 YAML 中。v02 仍保留，仅用于复现实验，不再是运行脚本默认
数据集。

双阶段直接写为：

```bash
agentmerge search experiments/qwen3_pilot_v01.yaml \
  --algorithm two_stage_nsga2 \
  --search_granularity global layer \
  --populations 12 12 \
  --generations 8 12 \
  --num_layers 36 \
  --device auto
```

分解式多目标 DE 使用 `--algorithm de`；Adapter→Layer 双阶段 DE 使用
`--algorithm two_stage_de`。DE 参数直接使用 `--differential_weight`、
`--crossover_probability`、`--mutation_probability`、`--neighborhood_size`、
`--neighborhood_mating_probability`、`--max_replacements` 和
`--decomposition`。默认采用 `DE/current/1/bin`、多项式变异和
Tchebycheff 分解。

`progressive_de` 在固定 Adapter 池上逐步释放层级自由度。典型 Qwen3-8B
调度为 `1→2→4→8→16→32→36` 个连续层组；每次递归二分当前最大区间，最终到达
逐层系数。阶段切换把完整已评价种群无损复制到子区间，不重新评价，也不加入局部
扰动。所有阶段使用标准 `DE/current/1/bin` 与 MOEA/D 分解，不包含多面体启发
算子、个性化池或 Adapter 筛选。

`adaptive_resolution_de` 使用固定 `1→4→8→16→32→36` 嵌套层级，但不预先固定
各阶段代数。它只复用正常 DE trial：按 reference-vector sector 统计 successful
displacement 的方向一致性，并将相对 scalarized gain/真实 FE 作为边际效用；二者
同时饱和且达到当前阶段最低 FE 后才自适应 promotion。独立的 future-reserve 硬
约束保证最终进入逐层空间并获得 `--min_stage_fe` 中预留的最低预算。promotion
按位复制完整 population 与 archive，保留 objective/ideal point，消耗 0 FE。
总预算和阶段最低预算分别由 `--total_budget` 与 `--min_stage_fe` 显式给出。

当前研究配置固定使用 6 个 Adapter，并支持 PEFT 与 vLLM V1。

一个 `--populations`/`--generations` 值会广播到全部阶段；也可以按阶段分别给值。
其他参数包括 `--workers`、`--checkpoint-every`、`--max-cases`、
`--runtime-mode`、`--evaluation-timeout-seconds`、`--max-retries` 和
`--output-dir`。NSGA-II 参数直接使用
`--crossover_probability`、`--sparsity_mutation_probability` 和
`--mutation_sigma`，不经过通用参数字符串。非法值会被严格拒绝。每次搜索的
`resolved_config.yaml` 只保存 YAML 所属的实验定义；实际算法、阶段和 CLI 预算
保存在 `output_manifest.json.search`，避免把搜索策略重新混入实验 YAML。

## 使用 Qwen3-8B

Qwen3 所需的 Transformers 版本不再向当前系统的 Python 3.8 提供安装包。
核心搜索框架仍兼容 Python 3.8；真实模型实验建议单独创建 Python 3.10 环境：

```bash
conda create -n agentmerge-qwen3 python=3.10 -y
conda activate agentmerge-qwen3
pip install -e '.[peft]'
```

复制并修改 [Qwen3 配置模板](experiments/qwen3_peft_template.yaml)。所有 adapter
必须指向同一个 `Qwen/Qwen3-8B` 基础 checkpoint，并在运行前核对：

1. PEFT task type 是生成式 CausalLM；
2. base model、tokenizer 和词表完全一致；
3. 全池 target modules 完全一致且存在于 Qwen3；
4. 不使用 `rank_pattern`、DoRA、adapter bias 或 `modules_to_save`；
5. adapter rank 有效，若在 YAML 中声明则与 PEFT 配置一致；
6. adapter 许可证和训练数据已经审查，且不污染最终 benchmark。

替换 adapter 路径后可先运行只下载小型 PEFT 配置、不加载模型权重的准入
检查：

```bash
agentmerge audit experiments/qwen3_peft_template.yaml
```

该检查会拒绝错误的 base model、非 `CAUSAL_LM` adapter、rank 声明不一致、
异构 target modules、`rank_pattern`、DoRA、adapter bias 和
### 公共基础模型资源

基础模型权重、tokenizer 和 AgentMerge 的加载入口是分开的。项目只保存
`ModelBackend` API 与实验配置，基础模型可以放在项目上一级的公共目录，多个
项目或多个实验共享同一份只读快照。例如目录可以组织为：

```text
/202422801010/
├── AgentMerge/
└── models/
    └── Qwen3-8B/
        ├── config.json
        ├── model-*.safetensors
        └── tokenizer.json
```

配置写成：

```yaml
model:
  backend: peft                 # 当前本地 Transformers + PEFT 入口
  name_or_path: ../../models/Qwen3-8B
  base_model_id: Qwen/Qwen3-8B # Adapter 兼容性身份，不是文件路径
  options:
    model_kwargs:
      local_files_only: true
    tokenizer_kwargs:
      local_files_only: true
```

`name_or_path` 支持绝对路径、相对于配置文件的路径和 Hugging Face 模型 ID；
`base_model_id` 只用于校验 Adapter 是否来自同一基础模型。切换后端不影响
Agent、拓扑、染色体和演化搜索。

### 可选 vLLM 后端

vLLM 使用独立环境，避免其固定的 Torch/CUDA 依赖改写现有 PEFT 环境：

```bash
conda create -n agentmerge-vllm python=3.10 -y
conda activate agentmerge-vllm
pip install -e '.[vllm,test]'
```

实验 YAML 中只需将模型后端和稳定的引擎参数改为：

```yaml
model:
  backend: vllm
  name_or_path: ../../models/Qwen3-8B
  base_model_id: Qwen/Qwen3-8B
  dtype: bfloat16
  generation:
    max_new_tokens: 256
    temperature: 0.0
    enable_thinking: false
    seed: 42
  options:
    fusion_algebra: exact_delta
    generation_kwargs:
      repetition_penalty: 1.05
    vllm:
      lora_transport: memory
      engine_mode: v1
      tensor_parallel_size: 1
      source_adapter_device: cuda
      max_model_len: 8192
      gpu_memory_utilization: 0.85
      max_lora_rank: 256
      max_loras: 1
      max_cpu_loras: 12
      max_num_seqs: 32
      enable_prefix_caching: true
```

内存模式固定兼容 vLLM `>=0.10,<0.11`，`engine_mode` 可选 `v1`（默认）或
`v0`。V1 使用同进程离线 EngineCore：保留 V1 scheduler、chunked prefill、prefix
cache 和模型执行路径，同时避免让 CUDA LoRA tensor 穿越进程边界。每个
AgentMerge GPU worker 持有一个基础模型和一份源 Adapter tensor 池；共享
`FusionAlgebra` 将 policy 编译为不可变执行计划，PEFT 与 vLLM 只是平行的计划执行后端。
`fusion_algebra: exact_delta`（默认）以 rank 拼接精确表示加权 delta；改为
`factor_linear` 时，按 LoRAHub/PEM 语义分别线性融合 A/B，异构 rank 零填充到
最大 rank，并有意保留因子乘积产生的交叉项。混合 policy 首次使用时生成临时
LoRA tensor、注册一个 LoRA ID，在该候选全部 benchmark 批次中复用，
候选结束后卸载并释放。全局 one-hot 源 Adapter 注册后跨候选复用。逐层、层组和
模块粒度融合使用同一代数，不会退化成全局平均系数。vLLM 会预编译模块路径和
tensor key，候选热路径预分配最终 A/B 后原位写入，不创建 `torch.cat` 或缩放 B
中间 tensor；同一次生成调用还可在 `max_loras` 容量内批处理不同 LoRA。配置
`source_adapter_device: cuda` 时，所有原始 A/B 因子全程常驻 GPU；vLLM 对源
LoRA folding scaling 所需的 B 副本按首次注册创建，不会原地污染源池。每个外层
GPU worker 还会获得独立的 torch.distributed 端口范围，避免并发启动竞态。

`VllmTensorLoraTransport` 是融合资源与 vLLM 引擎之间唯一的注册边界。源池和候选
融合只调用 `make_request/register/unregister`，不依赖 V0/V1 的具体 Engine 类。
V1 模式固定设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；多 GPU 并行仍由
AgentMerge 的持久 worker 调度，每张卡各自运行一个 V1 引擎。

源资源由统一池接口管理。配置中的 Adapter 名称是稳定的 `adapter_key`；池仅按
配置顺序分配用于当前阶段稠密数组的 0-based `coordinate_index`，运行时 vLLM
LoRA ID 则由后端单独分配，三种身份不再混用。PEFT 的
`_PeftResidentAdapter` 持有一份模型内源 LoRA，vLLM 的
`_VllmResidentAdapter` 持有一份源 tensor、请求和注册状态。每个私有对象都能
独立、幂等 `close()`；池按 `adapter_key` 关闭零引用对象。池状态严格遵循
`BUILDING → SEALED → PRUNED → CLOSED`：只有 BUILDING 可以追加源 Adapter，
SEALED 后阶段坐标冻结，物理释放不会重排全局轴，因此染色体、checkpoint 和结果
仍可比较。`runtime_snapshot()` 会同时记录 key、coordinate、configured、resident
和 released 集合。

候选临时 Adapter 不进入源池。共享 `FusionAdapterFactory` 协议由 PEFT/vLLM
分别实现，接收不可变 `FusionPolicy` 并返回同样可独立关闭的
`FusedAdapterResource`；融合数学仍由 `FusionAlgebra`/materializer 负责。
`candidate_session()` 只管理“首次物化、候选内复用、结束释放”的租约生命周期。
如果 vLLM 注销失败，资源和 tensor 会保持可追踪状态，并在下一候选开始前重试，
不会把未实际释放的显存错误标记为空闲。

这里的 `tensor_parallel_size=1` 是有意约束：当前多卡运行时已经是一卡一个常驻
worker，使用 `--device 0 1 2 3 4 5` 做候选级数据并行。vLLM 在每个拓扑阶段内
对同一候选的题目做 continuous batching；尚未让多个候选同时驻留在一张卡上。
首次接入应先执行 `audit` 和 `smoke-real`，再与 PEFT 后端核对答案、指标、token
计数和候选释放状态。

主搜索固定 `enable_thinking: false`，融合执行也固定为候选级按需物化，不提供
运行时策略选择。一个候选首次使用某个 Agent 的融合向量时，直接把各常驻源
LoRA 的 A/B 因子写入一个临时 rank-sum Adapter；该 Adapter 在当前候选的全部
benchmark micro-batch 中复用，候选结束后统一释放。实现不调用
`add_weighted_adapter`，也不保留跨候选的融合 Adapter。源 Adapter 可以具有
不同 rank，但激活 Adapter 越多，临时融合秩和运行成本通常越高。

融合生命周期由独立接口管理：算法无关的 `SearchSpace` 只处理染色体、编解码
和 repair；独立 `PlanTranslator` 把染色体翻译成不可变 `FusionPlan`；benchmark、
Agent 系统和多 GPU worker 只接收 `FusionPlan`；`FusionAlgebra` 生成后端无关的
执行计划；`ResidentAdapterPool` 提供常驻源资源；`AdapterMaterializer` 再把计划
编译成候选临时 Adapter。
`candidate_session()` 负责候选级复用和释放。更换 GA、ES 或 DE 时可以使用完全
不同的染色体与 repair，只需同样输出 `FusionPlan`，评价和后端无需修改。

```bash
agentmerge validate experiments/qwen3_peft_template.yaml
agentmerge search experiments/qwen3_peft_template.yaml
```

真实权重首次接入时，建议先在一张空闲 GPU 上运行后端冒烟检查。它不会运行
benchmark 或演化搜索，只验证基础模型、one-hot adapter、双 adapter 按需融合，
以及候选结束后的临时 adapter 释放：

```bash
agentmerge smoke-real \
  experiments/qwen3_vllm_research_v01.yaml \
  --device 1 \
  --max-new-tokens 16 \
  --one-hot-limit 1
```

`--device` 是物理 GPU 编号；进程内部会把它映射为 `cuda:0`。冒烟报告会输出
模型加载时间、生成延迟、显存和融合缓存状态。首次 on-demand 融合包含一次性
的 rank-sum adapter 物化开销，同一候选内的后续 batch 应直接复用该临时 adapter。

冒烟通过后，可在不启动优化器的情况下评测基础模型与全局 one-hot adapter：

```bash
PYTHONPATH=src python -m agentmerge.cli admit-adapters \
  experiments/qwen3_pool_v01.yaml \
  --device 1 2 4 5
```

默认产生 `1 + adapter 数量` 个固定候选；加入 `--role-wise` 后还会评测每个
Agent 单独启用每个 adapter 的组合。结果写入独立的
`outputs/*-one-hot-admission-*/admission_results.json`，不会进入 NSGA-II。
该目录同时保存本次准入实验的 `run.log`。

NSGA-II 的初始种群策略位于算法包的 `initialization.py`，不通过 CLI 配置。
默认初始化器读取 Adapter 的 `metadata.role_admission`：例如
`critic_positive` 会自动生成“仅在 critic 启用该 Adapter”的 one-hot 种子；
没有正向准入标记的 Adapter 不生成角色种子。Two-stage NSGA-II 只在第一阶段
运行该初始化器，第二阶段由第一阶段 Pareto 精英投影初始化。

模板中的 Adapter 仓库标识是占位符，替换前不能执行真实搜索。

## 阶段式批量推理

JSONL benchmark 不再逐题完整运行多 Agent 系统，而是按 micro-batch 和拓扑
阶段执行。例如 planner → solver → critic 会先批量生成一组题目的 planner
输出，再批量生成 solver 和 critic 输出。批大小由以下配置控制：

```yaml
benchmark:
  options:
    inference_batch_size: 8
```

`ModelBackend.generate_many` 是后端无关的批量接口；不支持批处理的后端自动
退化为逐条生成。PEFT 后端会将融合向量、生成长度、温度和随机种子相同的请求
合并为一次 `model.generate`，不同融合向量仍保持隔离并按输入顺序返回。
批量调用的墙钟时间平均分摊到每条 `GenerationResult.latency_seconds`，完整批次
耗时同时保存在 metadata 的 `batch_latency_seconds` 中。

评价指标中的 `latency_seconds` 是完整候选 session 墙钟时间按样本分摊后的值，
覆盖 prompt 构造、分词、临时 Adapter 物化、全部 Agent 生成和候选结束释放；
纯模型批次耗时另存为 `model_latency_seconds`。每个候选的物化次数、复制字节数、
物化/释放耗时同时写入评价详情的 `_runtime.fusion` 和 `events.jsonl`。

## 多卡执行

Qwen3 模板默认：

```yaml
runtime:
  mode: multi_gpu
  gpu_ids: [0, 1, 2, 3, 4, 5]
  start_method: spawn
  preload_workers: true
  evaluation_timeout_seconds: 3600
  max_retries: 1
```

主进程只运行所选优化器和写 checkpoint。每个 GPU 对应一个长期存活的子进程，
各自加载一份 Qwen3 和全局 adapter 池；多个候选按 GPU 实际完成速度动态分配。
模型不会随候选重新加载。

这是候选级数据并行，不是模型切分。Qwen3-8B 可以单卡容纳，因此应保持
`device_map: auto`：每个子进程通过 `CUDA_VISIBLE_DEVICES` 只看见自己绑定的
一张物理卡。CLI 的 `--workers` 只控制本地 evaluator 的线程数，在
`multi_gpu` 模式下并发数由 `runtime.gpu_ids` 决定。

如果部分 GPU 正被占用，应从 `gpu_ids` 删除，而不是依赖 `device_map: auto`
跨越其他卡。子进程启动失败、adapter 不兼容或候选评价失败会返回主进程；
可用 `max_retries` 控制重试，用 `evaluation_timeout_seconds` 检测停滞。

无需 CUDA 即可验证多进程调度链路：

```bash
agentmerge search experiments/multigpu_mock.yaml \
  --algorithm nsga2 \
  --search_granularity global \
  --populations 8 \
  --generations 2
```

## 配置扩展点

主要对象边界如下：

```text
ExperimentRunner
  ├─ ComponentFactory
  │    ├─ ModelBackend
  │    ├─ FusionPlanEvaluator
  │    └─ OptimizerRegistry
  │          └─ optimizer package → Optimizer
  ├─ shared representation/codec/repair + optimizer-private operators → FusionPlan
  ├─ fixed DAG / shared-scratchpad tool loop → AgentSystem(FusionPlan)
  ├─ CandidateEvaluationService → cache + batch/multi-GPU evaluation
  ├─ Pareto utilities → dominance + sorting + crowding
  └─ SearchObserver → checkpoints
```

新增后端可继承 `ModelBackend`；新增 benchmark 可实现 `FusionPlanEvaluator`；新增
演化算法可实现 `Optimizer`，并向 `OptimizerRegistry` 注册构建
函数。评价侧不接收染色体，因此新算法可以自行决定表示、融合粒度、目标转换
和约束修复，不需要修改 `ComponentFactory`、benchmark 或 GPU 调度器。

当前有六个独立优化器包：

- `src/agentmerge/optimizers/nsga2/`：单阶段 NSGA-II；
- `src/agentmerge/optimizers/two_stage_nsga2/`：全局→层级 Two-stage NSGA-II；
- `src/agentmerge/optimizers/de/`：单阶段 MOEA/D-DE；
- `src/agentmerge/optimizers/two_stage_de/`：Adapter→Layer Two-stage DE。
- `src/agentmerge/optimizers/progressive_de/`：递归层分区与逐步自由度释放 MOEA/D-DE。
- `src/agentmerge/optimizers/adaptive_resolution_de/`：预算感知、自适应 promotion 的嵌套分辨率 MOEA/D-DE。

单阶段 NSGA-II 内部：

- `config.py`：`NSGA2Config`，负责本算法默认值、类型转换和参数校验；
- `components.py`：提供给 Two-stage NSGA-II 的单阶段装配边界；
- `variation.py` / `optimizer.py`：交叉、变异、选择和环境选择；
- `__init__.py`：校验完整实验配置、创建本算法私有组件并构建优化器。

搜索请求不写入 YAML，直接从 CLI 装配：

```bash
agentmerge search experiments/qwen3_pilot_v01.yaml \
  --algorithm nsga2 \
  --search_granularity global \
  --populations 24 \
  --generations 10 \
  --mutation_probability 0.2
```

DE 包中的 `PopulationUpdateStrategy` 将多目标生存策略与差分变异、评价调度、
搜索空间和 Translator 分离。当前实现是 `DecompositionUpdate`；后续尝试
Pareto/GDE3 风格更新时，可以替换该策略而不修改模型评价链路。CLI 解析后会把
参数放入内存中的算法命名空间；实验 YAML 不包含它。

GA、ES 和 DE 不强制共享染色体协议，只需把候选解码为 `FusionPlan`。NSGA-II
和当前 DE 复用公共的有界实向量 representation，但分别维护自己的变异和种群
更新；未来算法也可以实现其他 Genome/SearchSpace。例如对 Qwen3-8B 的36层按
每6层一组搜索：

```bash
agentmerge search experiments/qwen3_pilot_v01.yaml \
  --algorithm nsga2 \
  --search_granularity layer_group \
  --num_layers 36 \
  --group_size 6
```

内置表示为 `global`、`layer`、`layer_group`、`module_type` 和
`layer_module`。候选级按需物化支持全部表示；逐层或逐模块系数会在临时
rank-sum Adapter 的对应模块切片中分别烘焙。

代际优化器执行采用 `ask() → completion-order evaluation → tell()` 边界。多 GPU
调度器在候选完成后立即回收对应 worker 并提交队列中的下一项；NSGA-II 和
MOEA/D-DE 都在整代候选完成后执行种群更新，因此并行调度不会改变其代际算法
语义。`progressive_de` 在每个递归细化阶段复用相同代际 DE，精确继承的表型通过
阶段局部评价缓存复用，不重复消耗真实评价。

## Benchmark 定位

当前 `synthetic` 和 `data/admission_v01.jsonl` 的作用是验证融合、评分、多卡
调度和搜索链路，不能作为论文主结果。`admission_v01` 是从公开 MATH-500 与
HotpotQA 中固定抽取的小型工程准入集，并不是复用某篇 Agent-LoRA 论文的完整
题库。

没有立即绑定相似论文常用 benchmark，主要因为基础模型、免训练 adapter 池及
其训练数据当时尚未冻结；若先选题库，容易出现 adapter 训练集与搜索/测试集
重叠。同时，完整多 Agent 演化搜索的成本远高于单模型评测，需要先用小集合
确定单次评价成本和一天内可承受的搜索预算。

正式实验建议分层：

1. 语言推理使用 MNLI、QNLI、SNLI、RTE、SICK 和 SciTail 组成的
   6-NLI，并固定互斥的 search/validation/test 样本；
2. 用 MATH-500、HotpotQA 等未参与搜索的数据测试跨任务泛化；
3. 在模型和代码能力允许时，再增加 AIME、LiveCodeBench 或规划环境；
4. 所有题目按 case ID 固定，记录来源、版本、切分和 adapter 数据污染审计。

## 多目标建议

固定激活 adapter 数量时，连续系数本身通常不会明显改变 token 数。因此：

- 部署稳健性基线推荐
  `quality ↑ + worst_group_quality ↑ + fusion_rank_proxy ↓`；
- 分析任务偏好时推荐每个 `task_quality.<group> ↑` 加
  `fusion_rank_proxy ↓`，总体质量和最差组质量仅作为报告指标；
- 任务较多时，可用 `domain_quality.<domain> ↑` 控制目标数量，并保留每个
  `task_quality.<group>` 作为诊断指标；
- 若 `min_active_adapters < max_active_adapters`，可优化
  `active_adapters ↓` 或实测 `latency_seconds ↓`；
- `total_tokens` 只应在输出确实随 adapter 改变且生成上限固定时作为目标。

真实研究运行前还需要确定 benchmark、筛选/训练 adapter 池，并增加重复种子和
统计检验；当前 demo 数据仅用于接口联调。
