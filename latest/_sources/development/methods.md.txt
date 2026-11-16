# 方法、契约与技术报告

软件技术报告汇集架构、数据契约、明确算法子集、双轨正确性、患者泄漏控制与部署证据。它描述研究软件的验证范围，不代表临床验证或论文录用。

```{toctree}
:maxdepth: 1

软件技术报告 <../reference/technical-report>
原生能力边界 <../reference/native-boundaries>
Rust范围与后端选择 <../reference/rust-backend-decision>
独立标量精度审计 <../reference/scalar-accuracy-audit>
```

## 按问题查定义

| 方法问题 | 详细定义 |
|---|---|
| NIfTI 世界坐标、距离单位与重采样 | [图像与 ROI](../reference/image-native.md) |
| Original 子集、离散化和纹理方向 | [原生特征](../reference/features-native.md)、[提取配置](../reference/native-extraction.md) |
| 预处理统计量与原生目标函数 | [预处理](../reference/preprocessing-native.md)、[模型](../reference/models-native.md) |
| 分组 CV、OOF、融合与训练来源 | [研究工作流](../reference/research-native.md) |
| 临床推断、ICC 与任务评价 | [临床统计](../reference/clinical-native.md)、[可靠性](../reference/reliability-native.md)、[评价](../reference/evaluation-native.md) |
| 病理坐标、QC 与层级权重 | [像素工具](../reference/pathology-pixels.md)、[层级聚合](../reference/pathology-aggregate.md) |

双轨验证分别记录 PyRadiomics 3.0.1 兼容性与所声明 IBSI 定义的符合性，不用放宽容差混合两套规则。适用数据、已知差异及固定范围见[验证说明](../validation.md)。
