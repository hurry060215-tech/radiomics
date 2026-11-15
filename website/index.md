# Radiomics：从输入文件到可审查的研究结果

独立 Rust 核心、Python API 与绘图、无Python依赖的CLI。面向标量NIfTI影像组学、临床表建模，以及普通病理图像和外部embedding的轻量分析。

**先选择自己的入口。** 安装后可以离线运行合成入门教程；公开CT教程使用真实公开影像，记录许可和SHA256；自己的研究数据使用文件驱动工作流。

| 你现在有的输入 | 建议入口 | 可检查的结果 |
|---|---|---|
| 第一次使用，没有数据 | [快速开始](quickstart.md) | 固定可核对的小型输出 |
| 想看真实NIfTI/ROI | [公开CT逐步教程](tutorials/07_public_ct.ipynb) | 几何、ROI图、特征和质量记录 |
| 自己的影像和临床CSV | [自己的文件工作流](reference/user-guide/file-workflows.md) | 原生提取、患者分组建模、外部预测 |
| 普通patch、mask或embedding | [病理与embedding指南](reference/user-guide/pathology.md) | QC、层级聚合、真实tile分数覆盖图 |
| 想了解支持到哪一步 | [支持矩阵](reference/support-matrix.md) | 原生、外部适配、实验范围和暂缓内容 |

## 两条研究流程

1. NIfTI＋ROI＋临床表 → 几何/QC → Original特征 → 患者分组筛选与建模 → 分类/回归评价 → `.rad`保存与无标签外部预测。
2. 普通patch／外部embedding／已有mask → QC与特征 → tile→slide→patient聚合 → 复用原生模型 → 结果表与真实tile分数覆盖图。

核心计算和CLI不调用Python参考算法。Python训练、筛选和临床接口保留兼容默认，教程均显式指定 `backend="rust"`。不执行自动分割、深度模型训练、通用WSI解码或配准。

教程按“输入和来源 → 完整代码 → 中间检查 → 输出和解读 → 换成自己的数据”组织。合成分数只说明软件流程运行成功；单例公开CT没有训练结局，不能据此编造分类研究。

```{toctree}
:maxdepth: 2
:caption: 开始使用

安装 <installation>
快速开始 <quickstart>
数据 <datasets>
教程 <tutorials/index>
```

```{toctree}
:maxdepth: 2
:caption: 自己的数据与方法参考

输入契约 <reference/user-guide/input-contracts>
研究流程 <reference/user-guide/file-workflows>
病理 <reference/user-guide/pathology>
常见问题 <reference/user-guide/results-and-errors>
API <reference>
验证 <validation>
贡献 <contributing>
```
