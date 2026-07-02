# Translation Notes — Attention Tracker (NAACL-Findings 2025)

## 翻译策略

- **翻译风格**：中英混合，中文为主，关键术语保留英文原词
- **图表处理**：未单独裁剪图片，论文 PDF 关键信息已在 paper.md 中以表格/文字描述
- **翻译粒度**：Abstract、§1 Introduction、§3 Distraction Effect、§4 Detection、§5 Experiments 完整；Related Work 仅按类别梳理
- **置信度**：高（PDF 文本可选择）

## 核心术语对照

| 术语 | 选择 | 备选 | 理由 |
|------|------|------|------|
| Prompt Injection (PI) Attack | 提示注入攻击 | 提示词注入 | 与安全社区标准译法一致 |
| Distraction Effect | 分心效应 | 注意力转移效应、干扰效应 | "分心"对应 "distraction"，最简洁直观 |
| Important Heads | 重要头 | 重要注意力头 | "重要头"是 attention head 的标准简称 |
| Focus Score (FS) | 焦点分数 | 注意力分数 | "焦点"突出"集中在指令上"的含义 |
| Training-free Detection | 训练免费检测 | 无需训练的检测 | "免费"比"无需训练"更醒目 |
| Open-Prompt-Injection | — | 开放提示注入 | 保留原名，业界标准称呼 |
| deepset prompt injection | — | deepset 提示注入 | 保留原名（数据集名） |
| AUROC | ROC 曲线下面积 | AUC | 分类任务标准指标 |
| Known-answer Detection | 已知答案检测 | — | 沿用原文术语 |
| LLM-based Detection | 基于 LLM 的检测 | — | 沿用原文术语 |
| Trained Detector | 训练型检测器 | — | 沿用原文术语 |
| Induction Heads | 归纳头 | — | Olsson et al. 2022 标准译法 |
| Successor Heads | 继任头 | — | Gould et al. 2024 标准译法 |
| Function Vectors | 函数向量 | — | Todd et al. 2024 标准译法 |
| Lookback Lens | 回顾透镜 | 回溯镜头 | "透镜"更贴近 lens 的隐喻 |
| Naive Ignore Attack | 朴素忽略攻击 | 简单忽略攻击 | "朴素"是 ML 术语标准译法（naive） |

## 关键数据点核对

| 数据点 | 原文 | 笔记中 | 核对状态 |
|--------|------|--------|---------|
| AUROC 提升上限 | up to 10.0% | 10.0% | ✓ |
| 训练免费方法平均提升 (Open-Prompt-Injection) | 31.3% | 31.3% | ✓ |
| 训练免费方法平均提升 (deepset) | 20.9% | 20.9% | ✓ |
| 校准数据量 | 30 random sentences | 30 条 | ✓ |
| 默认 k 值 | k=4 | k=4 | ✓ |
| 最优 k=4 head 占比 | 0.3% | 0.3% | ✓ |
| k=4 AUROC (Llama3-8B + deepset) | 0.986 | 0.986 | ✓ |
| 模型参数量范围 | 1.5B–9B | 1.5B–9B | ✓ |
| 模型数 | 5 | 5 | ✓ |
| 额外推理开销 | zero | 零 | ✓ |

## 引用与设计概念溯源

| 概念 | 出处 |
|------|------|
| Prompt Injection | Perez and Ribeiro 2022; Greshake et al. 2023 |
| 注意力机制 | Olsson et al. 2022 (Induction heads); Gould et al. 2024 (Successor heads); Todd et al. 2024 (Function vectors) |
| 用 attention 检测行为 | Chuang et al. 2024 (Lookback Lens); Lyu et al. 2022 (AttenTD) |
| 用 attention 操控模型 | Zhang et al. 2024b |
| Open-Prompt-Injection 基准 | Liu et al. 2024b (USENIX Security 24) |
| deepset 数据集 | HuggingFace 2023 |
| 训练型检测器 | ProtectAI.com 2024; Meta 2024 (Prompt-Guard) |
| 任务偏移检测 | Abdelnabi et al. 2024 (activations) |
| 指令层次训练 | Wallace et al. 2024 (Instruction Hierarchy) |
| 提示型防御 | Jain et al. 2023; Hines et al. 2024 (Spotlighting) |
| StruQ | Chen et al. 2024 (与本文 §5 互补) |

## 图表位置参考（如果后续需要裁剪）

> 当前未单独裁剪图片。下次如需可视化，可按以下位置裁剪：

- `fig1_overview.png` — p.2310 左栏 (流程图)
- `fig2_attention_maps.png` — p.2312 整页 (注意力图)
- `fig3_distraction_strength.png` — p.2313 右上 (攻击强度对比)
- `fig4_qualitative.png` — p.2315 右上 (定性分析)
- `fig5_heads_generalization.png` — p.2316 上半 (跨数据集泛化)
- `fig6_data_length.png` — p.2316 左下 (数据长度影响)
- `table1_main_results.png` — p.2316 中部 (主结果)
- `table2_head_selection.png` — p.2316 右下 (head 选择)

## 局限与未翻译部分

- 完整 References 未翻译
- Appendix A.1-A.7 仅在 paper.md 中以概要形式引用，未完整翻译
- 公式 (1)(2)(3) 已在 paper.md 中给出 LaTeX 形式

## 论文核心结论一句话

> Attention Tracker **首次从注意力机制角度**解释 PI 攻击为何成功——**"分心效应"** (specific heads 把注意力从原指令转移到注入指令)；基于此提出**训练免费、零额外推理**的检测器，在 5 个 LLM × 2 数据集上取得 **0.97–1.00 AUROC**，且在 1.5B 小模型上同样有效。
