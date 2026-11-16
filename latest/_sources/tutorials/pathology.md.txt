# 轻量病理：Patch、聚合与坐标图

<span class="data-badge synthetic">05、06：固定合成数据</span>

**输入**：普通图像 patch 或外部 embedding、明确的 tile ID、patient→slide→tile 层级，以及来源可证明的坐标和尺度。此路线不提供通用 WSI 解码、自动分割或深度训练。

| 顺序 | 操作与检查 | 输出 |
|---|---|---|
| 1 · [Patch 到患者](05_patch_patient.ipynb) | 查看 QC 原始分数与排除原因；tile→slide→patient 聚合 | QC 表、层级特征、实际 tile 特征覆盖图 |
| 2 · [Embedding 与覆盖图](06_embedding_overlay.ipynb) | 打乱行序后按 tile ID 对齐；验证坐标、覆盖计数和实际 tile 分数 | 对齐后的 embedding、聚合结果、tile 概率与覆盖图 |
| 3 · [自己的 patch 与 embedding](../reference/user-guide/pathology.md) | 提供 manifest、原始身份/坐标；缺失物理尺度时保留像素单位 | 可供原生建模的患者特征表和可追溯图像结果 |

```{toctree}
:maxdepth: 1

05_patch_patient
06_embedding_overlay
```

覆盖图必须来自实际 tile 特征或 tile 预测，不能把患者概率广播到局部。重叠 patch 的实例数只能按 patch 报告，没有全局实例 ID 时不能相加为全切片细胞数。空切片、身份与坐标错误见[结果解读与排错](../reference/user-guide/results-and-errors.md)。
