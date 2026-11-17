# 用户指南：使用自己的研究文件

这里按实际操作顺序组织：先定义输入与身份，再提取或聚合，然后训练、预测和解释输出。脚本位于仓库的 `examples/real/`，读取用户提供的路径、坐标和结局；每次使用新的输出目录。

## 选择工作流

| 已有输入 | 阅读顺序 | 最终产物 |
|---|---|---|
| NIfTI、ROI 与临床表 | 输入契约 → 影像文件工作流 → 训练与预测 | 特征、QC、患者划分、模型与外部预测 |
| 已有 CSV / Parquet 特征 | 输入契约 → 训练与预测 → 结果解读 | 三任务评价、`.rad` 模型与预测表 |
| 普通 patch 或外部 embedding | 输入契约 → 病理与 embedding → 训练与预测 | QC、层级特征、真实 tile 坐标图与患者预测 |

```{toctree}
:maxdepth: 1
:caption: 按操作顺序阅读

输入契约与身份 <../reference/user-guide/input-contracts>
影像与临床文件工作流 <../reference/user-guide/file-workflows>
三任务训练与外部预测 <../reference/user-guide/training>
普通病理与外部 embedding <../reference/user-guide/pathology>
结果解读与排错 <../reference/user-guide/results-and-errors>
真实公开表格说明 <../reference/user-guide/public-tabular>
本地样例与输出边界 <../reference/user-guide/local-samples>
教程实际验证范围 <../reference/user-guide/validation>
```

尚无自己的文件，可从[教程路线](../tutorials/index.md)开始。需要函数、参数与 CLI 配置时使用 [API 参考](../reference.md)。算法定义、Golden、基准实验和工程证据集中在[验证与开发](../development/index.md)，与操作步骤分别组织。
