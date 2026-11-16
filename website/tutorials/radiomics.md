# 影像组学：从 NIfTI 到特征表

<span class="data-badge synthetic">01、04：固定合成数据</span>

**输入**：标量 2D / 3D NIfTI、同网格 ROI、明确的样本与患者 ID，以及显式离散化参数。先从小数据理解几何，再增加区域、模态与时间点。

| 顺序 | 操作与检查 | 输出 |
|---|---|---|
| 1 · [基础提取](01_extract.ipynb) | 查看 shape、spacing、完整 affine；核对 ROI 和提取参数 | Original 特征、质量记录、有效配置与 ROI 预览 |
| 2 · [多区域与时间点](04_regions_delta_habitat.ipynb) | 明确瘤周距离单位；按键连接区域/模态/时间点；解释局部生境 | 带区域标识的特征、Delta 与生境结果 |
| 3 · [换成真实文件](../reference/user-guide/file-workflows.md) | 提供 cohort CSV 与真实路径；核对几何、ROI、单位和失败记录 | 可连接真实临床数据的特征表与 manifest |

```{toctree}
:maxdepth: 1

01_extract
04_regions_delta_habitat
```

首次接触真实 NIfTI，可先走[公开 CT 路线](real-data.md)。已有特征表和真实结局，再进入[建模与统计](modeling.md)。参数与算法子集见 [API 参考](../reference.md)，适用配置见[支持矩阵](../reference/support-matrix.md)。
