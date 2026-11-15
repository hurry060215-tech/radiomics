# 用自己的影像、临床和病理数据完成分析

本组教程面向已经有研究数据、希望明确知道“该准备哪些文件、运行哪条命令、看哪个结果”的使用者。程序读取你提供的结局和队列，不从影像文件名、病理颜色、患者顺序或特征值猜标签。示例字段名可以替换，真实研究中的标签定义、时间窗和研究对象需要由你预先确定。

两条路径使用同一个 Rust 计算核心：

1. **医学影像**：自己的 NIfTI＋ROI → 特征表 → 按样本或患者键连接真实临床/结局 → 原生训练、患者分组交叉验证 → 留出评价 → 保存模型 → 新队列无标签预测。
2. **轻量病理**：普通 RGB 图像/已解码 patch 或外部 embedding → QC 与明确坐标/ID → tile→slide→patient 聚合 → 同一原生训练和预测流程。

建议按以下顺序阅读：

| 页面 | 解决的问题 |
|---|---|
| [输入契约](input-contracts.md) | 每个文件一行代表什么；ID、患者、队列、特征、临床变量、结局怎样分开 |
| [影像＋真实临床数据](file-workflows.md) | 自己的 cohort、NIfTI/ROI、提取参数、临床连接及完整运行命令 |
| [三任务训练与外部预测](training.md) | 二分类、多分类、回归；分组 CV；`.rad` 保存加载；不带结局的预测 |
| [普通病理与外部 embedding](pathology.md) | 自有图像 manifest、层级聚合、embedding 行 ID 和真实 tile 覆盖图 |
| [结果解读与排错](results-and-errors.md) | 输出文件怎么看；替换成自己数据时常见错误和处理方式 |
| [教程实际验证范围](validation.md) | 运行过的三任务、CSV/Parquet、保存加载、负面场景与测试数据来源 |
| [真实公开表格](public-tabular.md) | 569条真实UCI数值记录、固定留出、原生LR与保存加载，明确记录组和临床验证边界 |

从仓库根目录运行 `examples/real/` 下的脚本。它们是可审查的使用范例，安装包的公共接口仍然是 `import ov_radiomics as rad`。下载源码的用户可使用：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\python.exe examples/real/train_table.py --help
```

Linux/macOS 的 Python 可执行文件通常为 `.venv/bin/python`。如果使用已提供的 wheel，先在自己的环境安装该 wheel，再用同一 Python 运行源码中的教程脚本。Python API 需要 NumPy/pandas，绘图还使用 Matplotlib；这些脚本不调用 sklearn、statsmodels、SimpleITK 或 PyRadiomics。Rust CLI 可在完全没有 Python 的环境独立使用，见[CLI 文档](../cli-workflows-native.md)。

运行教程会生成数据表、模型和图。每次指定**新的输出目录**；脚本拒绝覆盖已有目录。真实模式不下载医院数据，不生成伪造结局，不训练深度模型，也不自动分割或配准。原生支持、可选适配和暂缓内容分别列在[支持矩阵](../support-matrix.md)。

如果只是想先观察软件行为，可以运行 `examples/native/` 中明确标注的合成教程。那些标签来自固定软件验收样例，不是医学证据，不能作为真实模式的数据准备方法。
