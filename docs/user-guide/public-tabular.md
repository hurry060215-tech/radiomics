# 08：569条真实公开记录的原生二分类

这篇可逐步执行的Notebook补充单例公开CT教程：使用 UCI Breast Cancer Wisconsin (Diagnostic) 的真实诊断标签和既有数值表格，完成固定留出、原生训练、模型保存加载与不带标签的预测。数据来自细针穿刺细胞核图像测量，**不是本包提取的radiomics特征**。[官方数据与许可](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)；[DOI：10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)。

数据按CC-BY-4.0使用，保留 Wolberg、Mangasarian、Street等的来源署名和改动说明；数据许可与软件MIT分开。原始 `wdbc.data`、`wdbc.names` 留在仓库外缓存，不放进Git或安装包。所用固定文件摘要、完整特征顺序、下载地址和运行协议在 `examples/real/public_tabular_source.json`，源文件若改变则拒绝，不更新预期迎合下载或模型结果。

## 从文件开始运行

先生成未执行Notebook到新的本地文件：

```powershell
python tools/build_public_tabular_notebook.py --output outside-lessons/08_public_tabular.ipynb
```

在完整源码和对应原生Python环境下打开它，从上到下逐个Shift+Enter。Notebook直接读取源文件、检查数据、声明划分并调用公开API，不用runpy包住训练脚本。首次运行从UCI官方下载约129KB，后续复用校验通过的缓存。也可以先下载：

```powershell
python examples/real/download_public_tabular.py --output E:/my-radiomics-cache/uci-wdbc
```

`--from-cache` 可指定已核验原始文件的目录，供离线复制。缓存必须在Git工作树之外。发现同名文件内容不同会报错，拒绝覆盖，不从镜像自动接受另一版本。

交互运行会寻找仓库并创建新临时输出。批量执行者可在执行前注入：

```python
TUTORIAL_REPO_ROOT = "D:/my-project/radiomics"
TUTORIAL_DATA_ROOT = "E:/my-radiomics-cache/uci-wdbc"
TUTORIAL_RUN_ROOT = "E:/my-radiomics-results/wdbc-run-01"  # 必须尚不存在
```

执行者需要安装Notebook内核环境及本包；计算始终显式 `backend="rust"`。不依赖sklearn下载器或Python模型。生成器只写指定Notebook，不修改网站或共享配置；发布到教程站由主流程集成。

## 预先声明的计算与划分

| 项目 | 本教程固定选择 |
|---|---|
| 真实输入 | 569条source记录，30个数值测量；标签B/M来自原第二列 |
| 特征角色 | 仅30项原测量；sourceID、diagnosis和split hash不进入预测器 |
| 划分 | 每真实类别按SHA256(`uci-wdbc-holdout-v1\|`＋sourceID)、sourceID排序；前floor(0.8×该类记录数)训练，其余test |
| 记录/患者组 | 一source记录一个本地group ID；原始资料没有提供可验证的重复患者链接，不能声称569位独立患者 |
| 模型 | 原生Logistic，L2 lambda=0.1；Standard scaling，训练内填补/缩放 |
| 筛选/调参 | 不筛选、不搜索；一个预定候选，内层计划不伪造调参分数 |
| 分组CV | 外层3折/内层2折，seed20261115；patient API的group以sourceID声明 |
| 类别/阈值 | `[B,M]`，M阳性，阈值0.5；不根据test选择 |
| 评价 | OOF与train/test分别输出，CI=None；固定test ROC |
| 无标签演示 | 移除同一test的diagnosis、保持test身份，保存加载前后完整PredictionTable一致 |

源ID只能证明记录身份；可能存在的隐藏受试者关联不是本数据可检查的范围。用于自己的多病灶/多切片研究时，必须使用真实共享patient_id，不能按行生成“患者”。这里的test也不是新中心或外部临床队列；无标签预测演示不会把它重命名为external来夸大验证。

## 输出与解释

Notebook输出完整 `split-assignments.csv`、`study.csv`、`training/split_plan.json`、OOF/队列预测与分数、`model.rad`、留出ROC、无标签输入和重新加载预测、来源许可/输入摘要以及 `receipt.json`。其中运行marker为 `real-public-tabular-no-clinical-validity`，明确区别于合成教程和公开CT无结局教程。

模型分数只描述这一固定数据/划分的行为；不复现或比较原论文的算法结论，不把AUC作为医学诊断建议或外部泛化证据。真实数据并不自动等于临床验证，固定留出也不替代新的多中心队列。

当前协议和源码先冻结，实际执行后的产物/receipt会作为单独版本证据补充，不在看到AUC后改seed、阈值、特征、训练比例或模型参数。
