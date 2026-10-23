# CHIME-MAS

## CHIME 单智能体复现（τ²-bench + DeepSeek-V4.1-Flash）

论文：CHIME: Credit-Aware Hierarchical Memory Evolution for Long-Horizon Agentic Planning（arXiv 2609.02074）。官方代码未发布，本仓库按论文公式自行实现。

## 结构

| 文件 | 对应论文 | 状态 |
| --- | --- | --- |
| `chime/memory.py` | Eq.2/3 分层记忆，Eq.5 检索重排，Eq.7/8/9 价值更新，剪枝，合并/插入 | 已实现，13 个单元测试通过 |
| `chime/gate.py` | Eq.6 信用归因门（同骨干自反思） | 已实现，3 个已知答案探针全部判对 |
| `chime/embed.py` | 检索向量 | 临时替代：哈希 n-gram 向量；论文用 Qwen3-Embedding-0.6B |
| `chime/tau2_agent.py` | 先规划后执行的 agent，接入 τ² | 已写，未运行（需安装 tau2） |
| `run_stream.py` | 7:3 任务流，训练段演化、评估段冻结 | 已写，未运行 |
| `tests/` | `test_memory.py` 单元测试，`gate_probe.py` 归因门探针 | — |

## 超参（Table 7）
N=20，K=3，λ=3，α=1，β=0.2，θ_conf=0.5，θ_merge=0.88，n_min=3，θ_prune=−0.1，temperature=0。

## 论文没写清楚、由本实现补全的部分
1. 所有 prompt（planner、executor 附加段、归因门、合并）：按论文文字描述自写，见代码。
2. 归因标签为 none 时：不更新任何层（Eq.7 未给分支）。
3. δ=0 时是否做在线平均：Eq.8/9 与正文矛盾，`Config.update_on_zero_delta` 开关，默认按公式。
4. 复用次数 n：每次被检索使用即 +1。
5. 合并：LLM 将两条经验改写为一条，v、n 保持不变。
6. 检索 query：plan 层用用户首条消息，exec 层用首条消息 + 计划。
7. 每段对话只规划一次，不重规划。
8. Table 7 中正文未解释的阈值 0.3 / 0.1 / 0.03：暂不使用，留作消融。
9. "w/o Attribution Gate" 消融：论文未给细节，`chime_nogate` 用"所有经验写入两层、γ=1、不标误导记忆"近似。

## 运行
```
uv sync            # 在 tau2-bench 目录
python -m unittest discover -s tests
python tests/gate_probe.py
python run_stream.py --domain airline --variant chime --order-seed 0 --limit 4
```
密钥放在 `.env`（`DEEPSEEK_API_KEY`），不进代码。
