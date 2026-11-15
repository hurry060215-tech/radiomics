# 自己的 NIfTI＋ROI＋真实临床数据

目标例子：预测治疗前病灶对应的患者是否达到你预先定义的疗效终点。影像、ROI、时间窗、真实临床结局和测试队列均来自你的研究；这里的 `no/yes` 只是列编码示例，程序不会生成它们。

## 1. 准备图像 cohort

```text
my-study/
  cohort.csv
  images/P001_CT.nii.gz
  masks/P001_lesion1.nii.gz
  masks/P001_lesion2.nii.gz
  clinical.csv
```

`cohort.csv`：

```csv
sample_id,patient_id,image,mask,modality,region,lesion_id,center
S001,P001,images/P001_CT.nii.gz,masks/P001_lesion1.nii.gz,CT,tumor,L1,site_A
S002,P001,images/P001_CT.nii.gz,masks/P001_lesion2.nii.gz,CT,tumor,L2,site_A
```

这两行只说明文件结构，样本量不足以训练或分组 CV。真实文件需要覆盖你研究的全部样本；每类训练患者数和内外层折数必须相容。确认影像与 mask 的完整 affine、方向、原点、shape 和单位匹配。多模态输入应事先完成必要配准，程序不自动配准。mask 为整数标签，背景0，所选 ROI 非空。

## 2. 显式提取参数

从仓库根目录运行：

```powershell
python examples/real/nifti_extract.py --cohort my-study/cohort.csv --bin-width 25 --labels 1 --features full --output my-study/extracted
```

`25` 是本命令**显式选择的数值**，不是通用 CT/MR/PET 临床预设。固定 bin count 可以改用 `--bin-count`；两个离散化选项二选一。默认不重采样，Original 范围；`basic` 提取一阶与 shape_basic，`full` 加入已支持的55项强度/纹理特征。shape_basic 的计数、体积/面积、bbox、质心是独立基础形态输出，不包括网格表面积或球形度。

首轮先对少量自己的样本检查 `preview/` 图、`diagnostics.csv` 和 `failures.csv`：ROI 是否覆盖目标结构，方向/切片是否合理，标签是否选对。预览用于 QC，不替代所有三维几何检查。任何失败都需要解释；不能因为提取成功样本数减少而不记录被排除者。

输出包括 `features.csv`、其 `.manifest.json`、`metadata.csv`、失败/诊断记录、来源参数和预览图。查阅实际列名后再选模型特征，不把全部 bbox 坐标或中心标识无审查地加入预测器。

## 3. 准备真实临床和队列

患者级 `clinical.csv` 每人一行，顺序不必与特征表一致：

```csv
patient_id,cohort,response,age_years,sex
P001,train,yes,63,F
P002,test,no,58,M
```

队列必须由完整研究设计确定。同一患者所有病灶共用 cohort。`test` 一经固定不能反复用于调参；`external_A` 和 `external_B` 是不同队列，保留各自身份。回归场景将 `response` 换成真实连续结局，例如 `followup_score`；多分类用显式临床类别。不要根据影像均值生成结局，不把随访后才获得的变量作为治疗前预测器。

在 `my-study/study.json` 写入：

```json
{
  "schema_version": 1,
  "table": "extracted/features.csv",
  "feature_manifest": "extracted/features.csv.manifest.json",
  "metadata": "extracted/metadata.csv",
  "clinical": "clinical.csv",
  "clinical_key": "patient",
  "join_policy": "exact",
  "id_column": "sample_id",
  "patient_column": "patient_id",
  "cohort_column": "cohort",
  "outcome_column": "response",
  "numeric_features": ["original_firstorder_Mean", "original_firstorder_Variance", "age_years"],
  "categorical_features": ["sex"],
  "optional_metadata": {"center": "center", "lesion_id": "lesion_id"},
  "missing_values": [""],
  "nan_as_missing": false
}
```

这里只选择两项影像特征和两项临床变量，使第一次运行易于审阅。影像特征来自表，患者/病灶和中心来自 metadata/manifest，队列和真实结局来自 clinical。按键连接会验证重复、缺失和冲突，不依据行号对齐。

## 4. 训练和看结果

```powershell
python examples/real/train_table.py train --study my-study/study.json --task binary --classes no yes --cv 3 --inner-cv 2 --seed 20261115 --plots --output my-study/model-run-01
```

二分类第二个类别 `yes` 是阳性类别；不得交换后仍按原临床含义解释概率。默认原生 Logistic，患者分组外层3折、内层2折，数值中位数/类别众数、one-hot、Standard scaling 在每次训练折内拟合。当前脚本默认不作单因素/稀疏筛选、不搜索参数，避免把未解释的筛选策略嵌入初次分析；要加筛选时见[训练页面](training.md)。

`cohort-metrics.csv` 中的 `train` 是拟合后表观预测，不能当作泛化性能；`oof-metrics.csv` 与外层分组折分数在训练患者内形成，模型比较使用它们仍有选择不确定性。固定 `test` 和多个外部队列分别报告。患者级指标会先汇总同患者病灶预测，未把同患者多个病灶当作独立患者扩大样本量。

`holdout-*-roc.png`、校准和混淆矩阵只画有真实结局的留出/外部队列。`plot-cohorts.json` 对应文件与真实 cohort，患者不足或类别缺失会报错/缺少相应曲线，不自动补标签。更完整的文件解释见[结果页](results-and-errors.md)。

## 5. 外部预测

对新队列使用同一提取参数和特征定义；保存为新的 `external-study.json`，显式 `outcome_column=null`，队列 `external_B`，保持 patient_id。然后：

```powershell
python examples/real/train_table.py predict --model my-study/model-run-01/best-model.rad --study my-study/external-study.json --output my-study/external-predict-01
```

新表列顺序可以变化，所需名字和类型必须匹配。程序不在新队列重新填补/缩放/筛选，也不要求先知道结局。训练患者伪装成 external 会报错；若确实要查看训练表观预测，使用原训练结果中已经记录的 `train` 输出。
