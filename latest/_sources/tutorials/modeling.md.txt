# 建模与统计：训练、评价与预测

<span class="data-badge synthetic">02、03：固定合成数据</span>

**输入**：数值特征、明确列角色的临床变量与结局、样本/患者身份、预声明队列。教程标签用于软件演示；实际研究必须提供真实结局。

| 顺序 | 操作与检查 | 输出 |
|---|---|---|
| 1 · [影像与临床二分类](02_image_clinical_binary.ipynb) | 同患者病灶共享划分；预处理与筛选只在训练内拟合；核对 OOF 来源 | 划分摘要、二分类指标与图、OOF 和融合结果 |
| 2 · [多分类与回归](03_multiclass_regression.ipynb) | 显式任务与类别顺序；分别解释分类概率与连续预测 | 多任务评价、`.rad` 模型、保存加载与外部预测 |
| 3 · [使用自己的表格](../reference/user-guide/training.md) | 明确 feature / clinical / outcome / metadata；连接患者键并核对外部队列 | 可审查训练配置、真实留出评价与无标签预测 |

```{toctree}
:maxdepth: 1

02_image_clinical_binary
03_multiclass_regression
```

没有影像但想检查真实文件建模，可读[公开 UCI 数值表教程](08_public_tabular.ipynb)及其记录身份限制。临床推断、统计定义与原生模型约定见 [API 参考](../reference.md)；患者泄漏控制和性能实验见[验证与开发](../development/index.md)。
