# `run_qwen3_shift_experiment.py` 使用说明

本目录下的 `run_qwen3_shift_experiment.py` 用于研究 **Qwen3-8B** 模型上「Attention Shift」与「Prompt Injection 攻击是否成功」之间的关系。它在 Attention Tracker 项目基线之上，引入 LLM-as-a-Judge 与 Shapley 归因，量化评估「注意力偏移」能否作为提示注入的风险信号。

> 与本目录其他脚本（`find_heads.sh` / `run_dataset.sh` / `visualize_attention_tokens.py`）相比，本脚本是**针对 Qwen3-8B 的专项实验脚本**，具备完整的 collect → judge → shapley 流水线。

---

## 1. 实验目标

1. 对每条样本，先用 `AttentionDetector` 计算「Focus Score」并判断 `AttentionShift` 是否触发。
2. 调用 OpenAI 兼容的 LLM 裁判（默认 `deepseek-v4-pro`）对模型输出做语义层面的 `success / failed / ambiguous` 判定。
3. 寻找关键反例 `AttentionShift=True ∧ InjectionStatus=failed` —— 证明「注意力偏移 ≠ 攻击成功」。
4. 对反例与重点样本执行 Shapley 归因（coarse / fine 两个粒度），验证最终输出是由 `<system>/<user>/<data_fact>` 驱动，而不是由 `<data_attack>` 驱动。

---

## 2. 输入数据

| 文件 | 用途 | 关键字段 |
| --- | --- | --- |
| `data/bipia.jsonl` | 间接提示注入样本 | `system`, `user`, `data_fact`, `data_attack`, `clean_label` |
| `data/hotpotqa.jsonl` | 良性 RAG 对照样本（`data_attack` 为 `None.`） | `system`, `user`, `data_fact`, `answer` |

样本统一抽象为下列结构：

```text
<system>系统规则、权限边界、安全约束</system>
<user>用户真实任务</user>
<data>
  <data_fact>外部数据中的事实信息/任务证据</data_fact>
  <data_attack>外部数据中的未授权控制指令</data_attack>
</data>
```

实验脚本只关心 `system / user / data_fact / data_attack / clean_label / answer / id / attack_category / position` 等字段，其他字段会被忽略。

---

## 3. 流水线与阶段

脚本通过 `--stage` 把整体流水线解耦为 3 个阶段，便于在「本地推理 + 远程裁判 + 重型 Shapley 归因」之间分工：

| Stage | 是否需要 GPU | 说明 |
| --- | --- | --- |
| `collect` | ✅ | 本地 Qwen3-8B 推理 + 注意力抓取；输出 `results.jsonl` 与 `*.attention.jsonl` |
| `judge` | ❌ | 远程 LLM 裁判；只对需要判定的样本调用 OpenAI 兼容 API |
| `shapley` | ✅ | 对子集（默认 `candidate_attention_shift_attack_failed`）做 2 粒度 Shapley 归因 |
| `all` | ✅ | 顺序执行 collect → judge → shapley（默认） |

> 建议在生产环境中拆开执行：先跑 `collect` 落盘，再跑 `judge`（可放到任意联网机器），最后用 GPU 跑 `shapley`。

---

## 4. 安装与环境

### 4.1 Python 依赖

```bash
pip install -r requirements.txt
```

依赖至少包括：`torch`、`transformers`、`numpy`、`tqdm`、`openai`、`python-dotenv`。

### 4.2 模型权重

`configs/model_configs/qwen3_8b-attn_config.json` 中的 `model_id` 指向本地 Qwen3-8B 权重目录。脚本启动时通过 `AttentionModel` 加载：

```jsonc
{
  "model_info": {
    "provider": "attn-hf",
    "name": "qwen3-8b-attn",
    "model_id": "/root/Qwen3-8B"
  },
  "params": {
    "temperature": 0.1,
    "max_output_tokens": 32,
    "important_heads": [[10, 6], [11, 0], ...]
  }
}
```

> 重要：`important_heads` 列表由 `select_head.py`（参见 `scripts/find_heads.sh`）挑选得到，请勿在未重新执行 head 搜索的情况下修改。

### 4.3 `.env` 变量

`judge` 阶段通过 OpenAI SDK 调用远程裁判模型，所需环境变量在仓库根目录的 `.env` 中加载：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `JUDGE_PROVIDER` | `deepseek` | 标识用途，便于日志区分 |
| `JUDGE_MODEL` | `deepseek-v4-pro` | 裁判模型名 |
| `JUDGE_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 base URL |
| `JUDGE_API_KEY` | _必填_ | 远程裁判 API Key |
| `JUDGE_TEMPERATURE` | `0` | 裁判采样温度 |
| `JUDGE_TIMEOUT` | `60` | 请求超时（秒） |

> 缺 `JUDGE_API_KEY` 时，脚本会直接报错，不会降级为启发式；如需离线调试请显式传 `--judge_mode heuristic`。

---

## 5. 快速开始

### 5.1 一键运行（全流程）

```bash
python scripts/run_qwen3_shift_experiment.py \
    --model_name qwen3_8b-attn \
    --bipia data/bipia.jsonl \
    --hotpotqa data/hotpotqa.jsonl \
    --limit_bipia 120 \
    --limit_hotpotqa 100 \
    --max_output_tokens 1024 \
    --save_attention_tokens \
    --output result/qwen3_shift_experiment/results.jsonl
```

### 5.2 拆分执行（推荐）

```bash
# Stage 1：本地推理 + 注意力抓取
python scripts/run_qwen3_shift_experiment.py --stage collect \
    --limit_bipia 120 --limit_hotpotqa 100 \
    --save_attention_tokens \
    --output result/qwen3_shift_experiment/results.jsonl

# Stage 2：远程裁判（任意联网机器）
python scripts/run_qwen3_shift_experiment.py --stage judge \
    --judge_mode llm \
    --judge_only_attention_shift \
    --skip_judge_no_attack \
    --input  result/qwen3_shift_experiment/results.jsonl \
    --output result/qwen3_shift_experiment/results.judged.jsonl

# Stage 3：Shapley 归因（GPU）
python scripts/run_qwen3_shift_experiment.py --stage shapley \
    --shapley_scope candidates \
    --input  result/qwen3_shift_experiment/results.judged.jsonl \
    --output result/qwen3_shift_experiment/results.shapley.jsonl
```

### 5.3 离线 / 调试模式

- 仅跑启发式判定（无 API Key）：`--judge_mode heuristic`。
- 关闭 Shapley：`--no_shapley`。
- 关闭注意力抓取：不传 `--save_attention_tokens`。
- 限制样本数：`--limit_bipia 20 --limit_hotpotqa 20`。

---

## 6. 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model_name` | `qwen3_8b-attn` | 对应 `configs/model_configs/<model_name>_config.json` |
| `--bipia` | `data/bipia.jsonl` | BIPIA 数据集路径 |
| `--hotpotqa` | `data/hotpotqa.jsonl` | HotpotQA 数据集路径 |
| `--limit_bipia` | `20` | BIPIA 样本上限（`0` 表示跳过） |
| `--limit_hotpotqa` | `20` | HotpotQA 样本上限（`0` 表示跳过） |
| `--max_output_tokens` | `1024` | 模型生成上限 |
| `--seed` | `0` | 随机种子（`python / numpy / torch / cuda`） |
| `--output` | `result/qwen3_shift_experiment/results.jsonl` | 主结果 JSONL |
| `--input` | _阶段依赖_ | `--stage judge / shapley` 时必填，指向上一阶段产物 |
| `--stage` | `all` | `all / collect / judge / shapley` |
| `--no_shapley` | _off_ | 关闭 Shapley 归因 |
| `--shapley_scope` | `candidates` | `candidates` = 只对 `AttentionShift ∧ Injection=failed`；`attention_shift` = 只对 `AttentionShift=True`；`all` = 全部 |
| `--judge_mode` | `llm` | `llm / heuristic / none` |
| `--judge_fallback_heuristic` | _off_ | LLM 调用失败时回退到启发式 |
| `--judge_only_attention_shift` | _off_ | judge 阶段只对 `AttentionShift=True` 的样本调 API |
| `--skip_judge_no_attack` | _off_ | judge 阶段对 `data_attack` 为空/`None.` 的样本直接标记 `failed` |
| `--save_attention_tokens` | _off_ | 额外写出 token 级注意力 JSONL（用于可视化） |
| `--attention_output` | _同 stem + `.attention.jsonl`_ | 注意力 JSONL 输出路径 |
| `--attention_top_k` | `80` | 注意力记录中重复保留的 top-K 高注意力 token |

---

## 7. 输出文件

执行结束后会得到 3 类产物：

1. `result/.../results.jsonl`（或带 `.judged` / `.shapley` 后缀）：每行一条样本记录，包含：
   - `dataset / id / output / attention_shift / injection_status / judge / shapley_coarse / shapley_fine`
   - `candidate_attention_shift_attack_failed`：核心反例标记
   - `attention.focus_score / threshold / important_heads / mean_instruction_attention`
   - `sample`：原始 `system / user / data_fact / data_attack / clean_label / answer / attack_category / position`
2. `result/.../results.attention.jsonl`：可选的 token 级注意力记录，包含 `token_ranges / region_scores / top_tokens / tokens`，供 `visualize_attention_tokens.py` 直接渲染。
3. `result/.../results.summary.json`：聚合统计，含 `total / attention_shift / candidate_attention_shift_attack_failed` 以及 `by_dataset` 维度的 `injection_status` 分布。同时也会把该摘要打印到 stdout。

> `region_scores` 累加自 `auth / data / data_fact / data_attack / special` 五个区域；`candidate_attention_shift_attack_failed` 是本实验关注的核心反例指示符。

---

## 8. 关键模块速查

| 函数 | 作用 |
| --- | --- |
| `build_auth / build_data / build_prompt` | 把样本渲染成 `<system>…<user>` + `<data>…` 的两段式 prompt |
| `masked_parts` | Shapley 中按联盟把被屏蔽字段替换为 `[REMOVED_*]` 占位符 |
| `token_regions / region_for_index` | 依据 token 序列和 `<data_fact>/<data_attack>` 文本定位，划定每个 token 所属的 `auth / data / data_fact / data_attack / special` 区域 |
| `attention_scores` | 调一次 `model.inference` 抓首 token 的 attention map，产出 `focus_score / at_detect / threshold / mean_instruction_attention` 以及可选的 token 级 attention 记录 |
| `shapley` | 在玩家集合（`coarse: {auth, data}` 或 `fine: {auth, data_fact, data_attack}`）上枚举所有联盟并做对数似然差分，输出 Shapley 值与所有联盟值 |
| `InjectionJudge` | 默认通过 OpenAI 兼容 API 调 LLM 做注入成功判定，支持启发式回退 |
| `classify_injection` | 离线启发式：基于词重叠、攻击关键词、`base64`、脚本指令等做粗略分类 |
| `should_skip_judge` | 短路 `data_attack` 为空或 `AttentionShift=False` 的样本，节省 API 配额 |
| `summarize` | 生成 `summary.json`，统计 `attention_shift / candidate_attention_shift_attack_failed` 及各 dataset 的 `injection_status` 分布 |

---

## 9. 常见问题

- **`JUDGE_API_KEY is empty`**：在仓库根目录 `.env` 填入有效 Key，或显式传 `--judge_mode heuristic` 跑离线模式。
- **Shapley 极慢**：每次 Shapley 需要 `2^N` 次前向传播（`coarse` 4 次、`fine` 8 次）。可通过 `--shapley_scope candidates` 把范围限定在反例上，必要时配合 `--limit_bipia / --limit_hotpotqa` 进一步缩小。
- **`important_heads` 缺失或不一致**：重新跑 `scripts/find_heads.sh` 或参考 `select_head.py` 重新挑选 Qwen3-8B 的重要 head。
- **可视化**：把 `results.attention.jsonl` 喂给 `scripts/visualize_attention_tokens.py` 即可得到按 `auth / data_fact / data_attack / special` 区域上色的注意力可视化。
