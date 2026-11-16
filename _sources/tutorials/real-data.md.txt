# 真实数据：CT 提取与表格建模

<span class="data-badge real">07、08：真实公开数据</span>

这两篇是独立入口：一篇检查真实影像提取，一篇检查真实数值记录建模。两者不构成共享患者队列，不连接成诊断研究。

## 路线 A · 一例公开 CT 与 ROI

**输入**：固定来源的 IBSI PAT1 CT / GTV-1，CC-BY-NC-3.0；下载前确认许可，原始文件保留在 Git 工作树外。

1. [阅读来源、许可和 SHA256](../datasets.md)，下载或复用已核验缓存。
2. [运行 07 · 公开 CT](07_public_ct.ipynb)，检查真实 shape、affine、ROI 截面与世界坐标。
3. 检查特征、QC 与 manifest；按[文件工作流](../reference/user-guide/file-workflows.md)替换自己的 NIfTI 与 mask。

**输出**：可追溯文件、ROI 预览、特征表与提取配置。这例 CT 没有训练结局，教程止于提取与解释。

## 路线 B · 真实细胞核测量表与标签

**输入**：UCI WDBC 的 569 条既有数值记录及 B/M 标签，CC-BY-4.0。

1. 核对来源、固定文件摘要、列角色与预声明记录留出。
2. [运行 08 · 公开数值表](08_public_tabular.ipynb)，原生 LR 训练、评价和保存加载。
3. 检查预测表、ROC 与模型；按[三任务训练指南](../reference/user-guide/training.md)替换自己的特征和真实结局。

**输出**：固定记录划分、原生模型、留出指标与无标签预测。变量来自既有细胞核测量，不是本包影像提取结果；source ID 只证明唯一记录，不能证明患者独立。

```{toctree}
:maxdepth: 1

07_public_ct
08_public_tabular
数据与许可 <../datasets>
```
