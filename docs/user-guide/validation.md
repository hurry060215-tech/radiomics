# 文件驱动教程的实际验证

以下路径为记录时的本机位置，复现时请替换为自己的工作目录。`$WORKDIR` 表示仓库外的构建、缓存和证据目录；源码相对路径以仓库根目录为基准。这里只替换维护文档的路径，不改写历史测量文件。

本页记录 `examples/real/train_table.py` 的软件验收范围。验证运行日期为2026-11-15；使用明确标注的CC0固定合成表格，不将其结果解释为真实临床预测性能。真实教程脚本本身不包含或生成这些合成标签。

验证将已冻结 `golden/python_golden_v1/model_inputs.json` 的三任务记录转换为普通用户目录布局：基础数值特征CSV、按患者键的临床/结局CSV、训练/测试队列、单独无结局外部CSV和完整StudySpec。训练患者均有多个病灶；患者分组由同一公共原生API执行。

输入fixture SHA256：`ce154ab4ba1390661cd5bd3c4828f99aae96becd779620622903e26c99a06fea`。没有修改fixture、参考输出或数值容差。

| 实际执行 | 结果 |
|---|---|
| binary CSV → 默认原生LR → 保存 → 含真实结局的队列/OOF评价 → 留出图 → 无标签外部预测 | exit0 |
| multiclass CSV → 原生softmax → 同样保存/评价/绘图/无标签预测 | exit0 |
| regression CSV → 原生Ridge → 同样保存/评价/绘图/无标签预测 | exit0 |
| 三任务分别启动独立 `predict` 命令，加载 `best-model.rad` | 均exit0；完整PredictionTable与训练命令中保存加载后的外部输出逐结构相等 |
| 同一回归输入经公共FeatureTable原生导出Parquet，再由native read_study读取、训练和绘图 | exit0；保存的完整模型状态与CSV路径完全相等 |
| 二分类固定数据显式按患者取一行、sample_id=patient_id，以fit_unit=patient训练/绘图/保存加载外部预测 | exit0；验证病理聚合表使用的患者拟合层级，未生成或改变结局 |
| 病理页面外部embedding片段，使用固定6tile/2patient验收输入 | 原生按ID对齐、患者聚合和CSV/Parquet导出成功；未改fixture或按行数匹配 |
| training StudySpec缺真实outcome_column | 预期失败，明确“real outcome_column”；没有生成标签 |
| training StudySpec省略numeric_features让脚本猜列 | 预期失败，要求显式numeric_features |
| 将已训练患者改名为external队列后调用predict | 原生Leakage错误，拒绝违反已保存SplitPlan的患者队列身份 |

独立Python3.9.21环境已实际确认没有 sklearn、statsmodels、SciPy、PyRadiomics或SimpleITK；加载的后端报告为Rust。Matplotlib用于渲染。该测试证明教程调用路径与文件角色、保存加载和原生后端工作正常；它不重新验证全部模型数学、所有Parquet逻辑类型、每个平台或临床有效性，这些范围分别由项目Golden/部署证据限定。

本机可审查的输入、完整命令、输出图和receipt位于 `$WORKDIR/real-tutorial-validation-v1/`，`receipt.json` SHA256为 `f35e0c99101f4889b1b0bb19eff3824aada9834fdbc39997f758993d020ae344`。该盘符是本次开发验收的位置，不是使用教程所需的输入路径。你的实际文件应按[文件工作流](file-workflows.md)和[训练页面](training.md)中的相对StudySpec路径准备。

随后两项限定补充验证的输出分别位于该目录的 `patient-unit/` 和 `embedding-doc-example/`，不重写先前receipt。embedding输入来自已冻结的 `pathology_aggregate_inputs.json`；患者层输入仅改变一行的层级布局，仍使用原固定结局。

运行时脚本SHA256为 `5f901df2b6cb69dc687c25632591d402bb8b75fdfb7f21d45089ec64e57b6454`，安装的原生扩展SHA256为 `bda55de68f267fc3bcd710732f621659792741658b5900639c78141c5714cd7c`。这些标识只描述实际执行的软件；后续文档改动不能被误写为重新运行或扩大验证。
