# 本地样例：从图像／ROI 到原生建模

09、10 是两篇中文源码 Notebook，读取用户自己提供的本机文件。仓库只提供教程代码，
不附带数据、不下载样本、不发布已执行的切面图或个体预测。网页构建不执行这两篇。
运行前确认自己有权使用输入；本地读取权限不自动意味着有权再分发数据及其衍生产物。

## 输入契约

`--data-root` 指向以下目录的父目录，输入和结果目录都必须在任何 Git 工作树之外。
这一版本面向已经有完整图像、ROI、标签、固定划分和既有特征表的样例，不是通用导入器。

```text
<data-root>/
  CT/
    images/0.nii.gz ...
    masks/0.nii.gz ...
    label.csv
    label_multicenter.csv
    rad_features.csv
```

- 图像与 mask 是同名的 `.nii.gz`；文件名以整数开头，用于稳定排序。
  `ID` 必须是完整文件名，例如 `0.nii.gz`，不能省略后缀。两组文件与标签 ID 集合须完全相同。
- `label.csv` 有 `ID,label`；`label_multicenter.csv` 有 `ID,label,group`。
  ID 唯一，两份标签逐 ID 一致；标签为原始字符串 `0`、`1`，group 为 `train`、`val`。
- `rad_features.csv` 有唯一 `ID` 和107个既有数值特征；09会另外生成55列原生特征。
  两个视图逐 ID 对齐，不靠 CSV 行顺序匹配。ID、label、group 不进入预测器。
- 二值 mask 标签只允许0、1，label 1不可为空。图像和 mask 必须同形状、坐标单位均为毫米。
  其他 ROI 标签、多类结局和自定义特征数量需要另行定义教程协议。

`CT` 是输入目录约定，不构成模态或 HU 标定证据。`multicenter` 文件名也不能证明真实中心；
教程保留它的 train／val 分组，不重新选择划分。原始0/1标签未说明临床含义，不擅自命名疾病。
`patient_id=ID` 仅用于来源记录分组，不能证明患者独立；自己的多病灶研究必须提供真实患者关系。
临床／治疗列的时间顺序未知，因此本教程不将它们加入模型。

## 执行

使用已经安装本包原生 wheel 以及 `nbformat`、`nbclient`、`ipykernel` 的 Python 环境。
从仓库根目录运行，`DATA_ROOT` 和 `RESULTS_DIR` 由使用者设置为仓库外的目录：

```bash
python tools/run_local_sample_tutorials.py --data-root "$DATA_ROOT" --output "$RESULTS_DIR"
```

PowerShell 对应使用 `$env:DATA_ROOT` 和 `$env:RESULTS_DIR`。
输出目录必须不存在，且不得与输入目录重叠。执行器先校验位置和输入结构，再创建结果目录；
不会覆盖既有结果。默认读取仓库的 `website/tutorials/09_local_image_features.ipynb` 和
`10_local_modeling.ipynb`，也可用 `--notebooks` 提供包含这两个同名文件的目录。

每篇使用调用脚本的同一解释器启动全新内核，并使用独立工作目录。
共享执行器保留原单元顺序、禁止错误后继续执行、核验实际内联 PNG 和必需产物、记录代码 SHA。
若09失败则停止，10不会拿缺失或陈旧特征继续训练。失败 Notebook 与外层 receipt 保留在结果目录。

需要手工执行时，先在内核设置 `TUTORIAL_DATA_ROOT` 和全新的 `TUTORIAL_RUN_ROOT`；
第二篇还需设置 `TUTORIAL_IMAGE_FEATURES`，指向第一篇生成的 `features.csv`。
两篇的 `TUTORIAL_RUN_ROOT` 必须不同。手工运行也应将数据与所有输出保存在 Git 外。

## 09：显式几何处理与55项特征

原始 image／mask 的 affine 不一致时，首先保留原生拒绝信息，不强行复制 affine。
仅对同形状、毫米单位且网格外角点最大漂移不超过0.001 mm的输入，显式调用
`mask.resample_like` 最近邻重采样到 image；随后仍由 `ImageSample` 正常验证。
更大偏差或 ROI 消失会记录失败，不自动注册或居中。逐例 QA 记录重采样动作、
标签变化体素数、ROI前后体素数、原始几何警告和角点漂移。

图像不重采样、不归一化、不裁剪、不截断。固定 `python_golden_v1`、fixed-count 32 bins、
distance 1、direction_mean、GLDM alpha 0、voxel shift 0、threads 1；
32 bins是教学协议，不声称临床 CT preset。提取16项一阶和39项纹理，保留缺失值，拒绝无穷值。
部分输入可能有 shear／sform-qform 冲突，因此不调用要求正交网格的 `shape_basic`。
预览采用源数组 z 索引切面，显示窗仅影响绘图，不用于提取，也不声称完成解剖重定向。

输出有 `features.csv`、`qa.csv`、`source-hashes.json`、协议、两张图和 `receipt.json`。
`failures.csv` 只有表头代表零失败；独立 `failures.json` 的空数组用于无歧义验收。
不能为了满足“CSV至少一条数据”而伪造失败记录。原始输入哈希在执行前后逐项比较。

## 10：固定划分、OOF、保存加载

55列新特征和107列既有特征分别训练一个原生 LR，共用完整 `SplitPlan`。
外层3折／内层2折，seed 20261116，standard scaling，selector=None，
lambda=0.1，tolerance=1e-10，max_iterations=5000，阈值0.5，单线程。
预处理在训练折内拟合，不用 val 调参；不按 val 选择特征视图或挑选划分。

输出训练表观分数、训练三折 OOF 和固定 val 分数，以及 ROC、校准和混淆矩阵。
val 的模型内部 `prediction_type=external` 表示训练外预测，不能解释为真实外部中心验证。
两套特征定义与前处理不同，结果不支持提取后端优劣结论。
去掉同一 val 的标签，保存／加载 `.rad` 后比较完整 PredictionTable，要求完全相等；
这仍是同一批记录的无标签软件演示，不冒充新队列。

## 实际验收与公开边界

本机首次完整执行使用237对文件、188条 train／49条 val；09全部提取成功，
237例显式对齐后标签体素均未变化，最大外角点漂移约0.000029 mm。
两份模型保存／加载后的完整预测结构完全一致，原始输入哈希不变。
该记录是本地数据工作流验收，不是新的 benchmark、临床有效性或独立患者验证。

首次运行曾因通用验收器要求CSV含数据行，把零失败的清单判为失败。
原记录保留，后续用独立空 JSON 数组验收并在新目录重新执行两篇，没有伪造失败行。
PR版本已于2026-11-17在新目录独立执行：09的6个代码单元和10的5个代码单元
按原顺序完成，无执行异常，内核均已结束；两篇分别耗时约143秒和2秒。
原始代码、输入和输出摘要由外层 receipt 绑定，最终命令与结果放入 PR 描述。

源码 Notebook 不含任何保存的输出。真实执行 Notebook、样本图、特征表、标签、
完整个体预测、模型和含本机路径的原始 receipt 均在仓库外，不加入 wheel、sdist 或站点。
公开代码可以随项目许可发布；这不代替数据本身的授权。后续替换自有输入时，
先核对本页契约及[输入与身份要求](input-contracts.md)。
