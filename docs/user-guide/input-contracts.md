# 输入契约：先确定每一列的角色

## 一行与一个患者

医学影像的特征表通常一行对应一个病灶/ROI；患者可以有多个病灶或多个时间点。病理 tile 特征一行对应一个 tile，聚合后的一行才可能对应一个切片或患者。

| 角色 | 示例 | 约束 |
|---|---|---|
| 样本 ID | `S001`, `S002` | 在当前输入中唯一、非空；前导零应作为字符串保留 |
| 患者 ID | `P001`, `P001` | 可以重复；同一患者的所有样本必须进入同一研究队列和同一 CV 分组 |
| 队列 | `train`, `test`, `external_A` | 明确指定；固定测试患者不能混入训练；多个外部队列分别保留名称 |
| 数值特征 | `original_firstorder_Mean`, `age_years` | 显式名单；允许明确缺失，不能含无穷；不包含 ID、结局和队列 |
| 类别特征 | `sex`, `scanner_group` | 显式名单；由训练内众数填补和 one-hot 编码 |
| 结局 | `response`, `subtype`, `followup_score` | 研究者提供的真实观测；分类类别顺序或回归任务显式指定 |
| 可选元数据 | `center`, `modality`, `region`, `timepoint`, `lesion_id`, `slide_id` | 描述来源/配对；不会因为存在就自动成为预测变量 |

同一个人换成新的病灶编号，仍然是同一个患者。把一名患者的两次检查分别分配到训练和测试会泄漏。患者级评价还要求该患者的结局一致；如果研究目标本来就是病灶级且同患者内结局不同，需要明确采用 sample 评价，仍保留患者分组划分。

## 原生文件读取配置

`train_table.py` 读取与核心一致的 `StudyTableSpec` JSON。相对路径相对于 **JSON 文件所在目录**解析。输入表可以是 CSV 或平面 Parquet；不需要 pandas 的 Parquet 引擎。

例如 `study.json`：

```json
{
  "schema_version": 1,
  "table": "features.csv",
  "metadata": "metadata.csv",
  "clinical": "clinical.csv",
  "clinical_key": "patient",
  "join_policy": "exact",
  "id_column": "sample_id",
  "patient_column": "patient_id",
  "cohort_column": "cohort",
  "outcome_column": "response",
  "numeric_features": ["original_firstorder_Mean", "age_years"],
  "categorical_features": ["sex"],
  "optional_metadata": {"center": "center", "lesion_id": "lesion_id"},
  "missing_values": [""],
  "nan_as_missing": false
}
```

本教程要求三个 ID/队列角色和两个特征名单都显式填写，避免把另一结局列或辅助字段纳入模型。原生 reader 更宽松的列推断能力不由此脚本启用。没有类别变量时使用 `"categorical_features": []`。没有数值变量时可以用空数值列表，但总特征数必须非零。

`table` 是基础特征表。`metadata` 是按样本 ID 的一对一辅助表；`clinical_key="patient"` 时，`clinical` 中每个患者恰好一行，允许连接到该患者的多个基础样本。患者 ID 必须先能从基础表、metadata 或显式 feature manifest 获得。临床表中的结局/队列可以提供同名角色；两个来源的已知值冲突会报错，不能静默覆盖。

`clinical_key="sample"` 用于真正按样本记录的临床/结局。若辅助表的键名不同，可显式填写 `metadata_id_column` 或 `clinical_id_column`。默认 `exact` 要求相应键集合完全匹配；因此训练与预测应各准备恰当范围的临床表，不能把全院总表直接连进当前子队列。

显式 `join_policy="left"` 保留基础表所有行，并在 `input-report.json` 记录缺失和未使用辅助键。它不能让必需患者/队列变成缺失，也不能授权静默丢弃缺失结局；采用前应审阅连接报告。

## 缺失、类型与 sidecar

CSV 默认只有空字段是缺失。`NA` 若确实代表缺失，需要在 `missing_values` 中声明；它也可能是有效类别，因此不能自动猜。NaN 是否作为缺失由 `nan_as_missing` 明确控制，Inf 总是错误。Parquet null 与空字符串不同；字符串 ID 必须保留为 UTF8，不能把 `001` 先转成数字再恢复。

训练内拟合中位数、众数、编码和缩放；不在合并全队列后手工填补。全缺失列默认错误；未知类别不会悄悄创造新的 one-hot 列。日期、嵌套/list、复数数据需要先定义明确转换，不当作一般数值强制读取。

原生提取/聚合保存的特征表带 `<table>.manifest.json`，记录定义、单位、元数据和文件摘要。使用 `StudyTableSpec` 时可显式加入：

```json
"feature_manifest": "features.csv.manifest.json"
```

不要编辑特征 CSV 后继续沿用旧 SHA sidecar。需要改动时从明确的新特征对象重新导出，或对纯外部表不声明旧 manifest 并另保留来源记录。`rad.read_feature_table` 会重建完整 FeatureTable；`rad.read_study` 则加载用于建模的明确角色和连接，二者用途不同。

## 只有预测时

预测 JSON 使用同样的特征名单、类型和 ID；设置 `"outcome_column": null`，省略临床表中的结局列。若文件确实没有队列列，可以显式给 `"default_cohort": "external_B"`。这个默认值只适用于角色列完全不存在的情况，不能修补已有列中的空队列值。

没有标签就只能预测，不能算 AUC、RMSE 或校准。以后获得真实结局时再通过明确 ID 连接并评价，不将预测值回填为结局。
