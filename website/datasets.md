# 教程数据与许可

| 数据 | 来源/许可 | 用途 | 不可作出的结论 |
|---|---|---|---|
| 六套入门教程的数值、NIfTI和RGB patch | 本仓库固定合成数据，相应fixture声明CC0 | 离线学习、数值与流程复现 | 真实诊断效能、临床泛化 |
| IBSI PAT1 CT与GTV-1 | IBSI公开镜像，原始数据CC-BY-NC-3.0 | 真实NIfTI输入、几何/ROI预览、特征 | 多患者建模、IBSI CT A–E全面符合 |
| UCI WDBC 569条记录 | Wolberg/Mangasarian/Street等；CC-BY-4.0 | 原生LR、固定记录留出、保存加载和图 | 本包radiomics提取、不同记录必定为独立患者、外部临床验证 |
| 用户自己的影像、临床表、mask、embedding | 使用者负责访问授权和去标识化 | 实际研究流程 | 软件自动提供外部验证或伦理批准 |

## 公开CT

第07篇从[IBSI数据仓库](https://github.com/theibsi/data_sets)固定提交 `6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`获取一例CT和mask。原始来源DOI为[10.17195/candat.2016.08.1](https://doi.org/10.17195/candat.2016.08.1)。原始数据署名和非商业限制与本项目MIT软件许可证分别适用。

下载器要求显式确认许可，把原始文件放在Git工作树外，核对完整字节数和SHA256，同时保留原始README/LICENSE、来源说明和零结局的队列CSV。离线可以复用已核验的同一缓存，不关闭校验。

| 文件 | SHA256 |
|---|---|
| image.nii.gz | `909129fdf99d5f5fd17bc568b1834d31edce8663ea6746c6020720bfc1d7bbcb` |
| mask.nii.gz | `0e9243e473b1255c60e6460df465d0b07b65436d6e0c691fe3d17bca0594ae78` |

这例数据没有为本教程提供患者结局。教程止于提取与解释，不复制同一患者构造训练/测试集，也不随机捏造恶性、复发或生存标签。

公开CT Notebook中的已执行预览图属于相应NC数据的派生展示，受同一数据许可限制。
基础软件源码包不包含该已执行页；文档准备工具会在缺页时重建仅含原创代码的页面，
不自动下载或执行。完整Git仓库及文档预览保留带来源说明的实际运行展示。

## 合成演示与真实输入的关系

入门Notebook可以检查24体素ROI、固定患者划分和覆盖图计数。真实文件工作流复用同一API，但输入由用户提供。运行成功、特征数正确和模型分数较高，是三个不同问题；实际研究仍需独立结局、训练集内拟合和患者级外部验证。

## 真实公开表格

第08篇使用[UCI Breast Cancer Wisconsin Diagnostic官方数据](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)，
DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)，CC-BY-4.0。
其30项变量来自细针穿刺细胞核图像的既有测量，不是本包的NIfTI/纹理特征。
`wdbc.data` SHA256固定为 `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`。

原始CSV和说明保留外部缓存；Notebook保留来源、署名、改动和划分协议。
真实B/M标签用于预声明的分层记录留出，不从特征生成结局。
569个source ID只证明唯一记录，不足以证明患者独立或排除隐藏受试者关联。
具体参数、列顺序和解释见[真实表格教程说明](reference/user-guide/public-tabular.md)。
