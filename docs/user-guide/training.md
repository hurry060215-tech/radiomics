# 三任务原生训练、患者分组与无标签预测

## 从任意自有 CSV/Parquet 特征开始

特征不必来自本包。已有 radiomics、数值临床变量、聚合的病理或外部 embedding 都可以进入同一训练入口。特征定义、单位和来源需要一致，不能因为列名相同就认定不同软件的特征等价。

假设已准备 `features.parquet`，每行有 `sample_id/patient_id/cohort`；`clinical.csv` 每患者一行，有真实 `outcome`、年龄和性别。以下 `study.json` 是可直接解析的完整配置；把特征名和结局名换成你的真实列：

```json
{
  "schema_version": 1,
  "table": "features.parquet",
  "clinical": "clinical.csv",
  "clinical_key": "patient",
  "join_policy": "exact",
  "id_column": "sample_id",
  "patient_column": "patient_id",
  "cohort_column": "cohort",
  "outcome_column": "outcome",
  "numeric_features": ["feature_a", "feature_b", "age_years"],
  "categorical_features": ["sex"],
  "optional_metadata": {},
  "missing_values": [""],
  "nan_as_missing": false
}
```

全在一张表时省略 `clinical/clinical_key`；基础特征表没有患者/队列时，用[输入契约](input-contracts.md)里的 metadata/clinical 连接。`numeric_features` 始终显式列出，不使用“除结局外全部列”作为真实研究默认。

## 二分类

例如真实结局列中恰好使用 `stable/progressed`：

```powershell
python examples/real/train_table.py train --study data/study.json --task binary --classes stable progressed --models LR --cv 3 --inner-cv 2 --seed 20261115 --threshold 0.5 --plots --output results/binary-01
```

`probability` 是第二类 `progressed` 的概率。阈值0.5在本例预先指定；脚本不在测试队列搜索最优阈值。若实际类别是字符串 `0/1`，用 `--classes 0 1`；若是 `negative/positive`，必须写实际文本。不可用某个特征阈值临时生成“真实标签”。

默认模型 LR 为带 L2 的 Logistic，默认 Standard scaling；类别众数填补和 one-hot、数值中位数填补均在训练折内拟合。默认不启用筛选。你可以用 `--models LR KNN GaussianNB RandomForest` 作有限模型比较；这会增加训练量，默认比较指标为患者级 AUC。没有参数网格时单候选不会伪造调参结果。模型的精确定义和正则化目标见[原生模型](../models-native.md)。

## 多分类

```powershell
python examples/real/train_table.py train --study data/subtype-study.json --task multiclass --classes subtype_A subtype_B subtype_C --models LR --cv 3 --inner-cv 2 --plots --output results/multiclass-01
```

此处 LR 为多项 softmax。三个编码是类别，不会当成连续的0/1/2做平方损失回归。保存每类概率和固定类别顺序；默认比较 macro OvR AUC，评价另给 macro/micro/weighted 汇总。图中的每条曲线是该类对其余类别的 OvR ROC，不能把所有曲线当作一个二分类总体。DeLong、DCA、H–L 的整体语义仍限二分类。

## 回归

```powershell
python examples/real/train_table.py train --study data/continuous-study.json --task regression --models Ridge --cv 3 --inner-cv 2 --plots --output results/regression-01
```

回归 `outcome_column` 必须指向实际连续量，不传 `--classes`。默认 Ridge、比较 RMSE，同时导出 MAE/RMSE/R²；图为患者真实值对预测值。预测单位与结局单位相同，软件不猜“天/月/分”等物理或临床含义。可以选择原生 Linear、KNN、RandomForest；秩不足 OLS、数值不收敛等将报错。

## 患者分组到底发生在哪里

`patient_id` 可以重复；每次外层和内层划分均以患者作为不可拆分组。填补、编码、缩放、筛选、参数选择只在对应训练患者拟合。固定 `test/external_*` 不参加这些拟合。保存的 `training/split_plan.json` 包括实际患者集合、类别顺序和种子，不能仅靠“第1折”判断两次训练共享划分。

默认 `--fit-unit sample --scoring-unit patient`：用每病灶一行训练，同时评价前按患者汇总；同患者结局必须一致。病理已经聚合到每患者一行时用 `--fit-unit patient`。真正病灶级研究可显式 `--scoring-unit sample`，此时脚本不提供患者图，省略 `--plots`；划分仍保留患者组。

当训练每类患者太少、折内缺类别、外部队列患者重叠、同患者跨 train/test、所有特征被筛空时，程序会报错。它不自动减少折数、合并类别、放宽阈值或改为随机按行划分。需要先修改研究方案和输入，再作为新的运行保存。

## 需要筛选或参数网格时

脚本的保守起点没有隐藏筛选。要使用筛选，可在自己的 Python 脚本调用同一公共 API，将 selector 传入完整训练 pipeline：

```python
import ov_radiomics as rad

loaded = rad.read_study("data/study.json", task="binary")
selector = rad.FeatureSelector(
    task="binary", backend="rust", scale="standard",
    univariate="auto", p_threshold=0.05, p_adjust="bh",
    corr_threshold=0.9, sparse_method="squared_lasso",
    alphas=[0.001, 0.01, 0.1]
)
result = rad.train_models(
    loaded.dataframe, loaded.target, loaded.features,
    categorical=loaded.categorical, backend="rust", task="binary",
    classes=["stable", "progressed"], positive_label="progressed",
    selector=selector, patient_id="patient_id", cv=3, inner_cv=2,
    model_params={"LR": [{"lambda": 0.01}, {"lambda": 0.1}]}
)
result.save("results/selected-binary-01")
```

这不是建议为每个小队列使用高维筛选和大量候选；它说明选择步骤必须进入患者分组 pipeline。多分类需要 task 对应的 OvR L1 Logistic 特征并集，不能照抄 squared_lasso。单因素默认二分类 Mann–Whitney、多分类 Kruskal–Wallis、回归 Pearson；零特征结果应解释，而不是反复降低阈值直到出现期望结果。

## 保存、加载和无标签外部预测

每次训练输出 `best-model.rad` 和每模型的 `training/model-*.rad`。`.rad` 是版本化类型状态，保存训练填补/编码/缩放/筛选、模型、类别/特征顺序、阈值和训练来源；无需 pickle/joblib。

外部数据准备完整 `external-study.json`：

```json
{
  "schema_version": 1,
  "table": "external-features.parquet",
  "clinical": "external-clinical.csv",
  "clinical_key": "patient",
  "join_policy": "exact",
  "id_column": "sample_id",
  "patient_column": "patient_id",
  "cohort_column": "cohort",
  "outcome_column": null,
  "default_cohort": "external_B",
  "numeric_features": ["feature_a", "feature_b", "age_years"],
  "categorical_features": ["sex"],
  "optional_metadata": {},
  "missing_values": [""],
  "nan_as_missing": false
}
```

```powershell
python examples/real/train_table.py predict --model results/binary-01/best-model.rad --study data/external-study.json --output results/external-B-01
```

也可以在训练命令加 `--predict-study data/external-study.json`；脚本保存后重新加载 `.rad`，实际验证加载前后外部预测完全一致。预测模式不接收训练命令的任务/类别覆盖参数，以保存 manifest 为准。

已有 DataFrame 时可直接：

```python
import ov_radiomics as rad
bundle = rad.load_model("results/binary-01/best-model.rad")
external = rad.read_study("data/external-study.json", task=bundle.manifest["task"])
predictions = bundle.predict_study(external.to_dict()["data"])
predictions.dataframe.to_csv("results/external-predictions.csv", index=False)
```

未知 schema、缺少特征、类型不匹配或训练患者与外部重叠明确报错。不因列顺序变化重新拟合，也不因为有相同行数就接受缺失或错误 ID。
