# Onekey 可见工作流与本包原生能力的差距审计

本次以本包提交 **`9315841648f18ae55aacb565f89e8763a0c4998b`** 为基线，检查本机 `onekey_comp/README.md`，comp1传统组学、comp2结构化、comp3融合、comp6解释性、选定comp8统计模块，以及相关sol1/sol2临床组合。共48个非checkpoint Notebook、1302个markdown/code源单元。检查日期：2026-11-15。

**结论：原生计算主流程与Onekey用户常见步骤已有广泛对应，但“原生计算完成”不能扩大为全部旧评分/绘图接口兼容。** 重点差距在原生对象到Rad-score/系数、LASSO路径/CV图、列线图及三任务CV比较图的桥接；学习曲线和探索降维目前走可选Python。图像滤波、ComBat、复杂生存、MIL等仍按用户明确范围暂缓，不因Onekey有菜单而追加实现。

本报告没有执行Onekey、没有检查其已有输出/患者表、没有导入或解密PyArmor/其他字节码、没有复制其代码或说明文字。提取物只含Notebook相对路径、文件/源单元SHA、固定topic标签、import和函数调用符号、关键字名称。**可见静态调用不证明实际运行成功、参数生效、内部数学或性能。** Onekey封装的默认值、算法与缺失处理均不由名称推定。

## 状态与证据怎样理解

- **等效**：本包有同类用户任务入口，按本包明确且冻结的定义工作；不表示数值复现Onekey。
- **部分**：任务主体存在，但范围、后端、输入对象或工作流便利入口不同。
- **缺失但原范围内**：明确承诺而缺失；本报告不从菜单额外制造这种义务，当前条目主要是部分覆盖/桥接。
- **显式暂缓**：原用户已排除，本次不要求占位或实现。
- **仅可选后端**：本包已有Python/reference路径，不能计为原生。
- **不可验证**：仅可见命名调用或菜单，内部定义未审阅；不据此实施同名算法。

所有cell索引均为原Notebook `cells` 数组的**零基索引**，不是执行计数或界面中的第几个代码框。完整逐项证据在[结构化JSON](onekey-workflow-gap-audit.json)：36项用户步骤，每项带来源path/cell/可见函数、源单元SHA、本包文件/入口/行号、状态、影响、优先级和建议。源码/测试的引用是静态审阅，**本轮未重新运行本包测试**；已有验收另见项目报告。

## 最需要避免混写的六件事

### 1. 临床融合不等于列线图已经兼容原生模型

可见来源包括 `comp9-Solutions/sol1. 传统组学-单中心-临床/Step4. 指标汇总.ipynb` cell3/12中的 `sklearn.linear_model.LogisticRegression`，cell19中的 `onekey_algo.custom.components.nomogram.risk_nomogram`，以及comp6列线图Notebook cell5/8的nomogram/risk_nomogram调用。它们说明用户会经过融合和画图两个步骤，不能证明封装采用哪种拟合/评分定义。

本包 `fusion_data/fit_fusion` 已实现真实患者共享SplitPlan、基础OOF、原生Logistic/softmax、显式概率加权和可移植模型。临床变量也能与影像共享pipeline与plan。**基线的 `plot_nomogram` 却读取 `fusion.model.coef_`、`intercept_`；`NativeFusionResult.model` 是 `ModelBundle`，没有这些sklearn属性。** 因此不能把原生融合数值验收当作“原生融合对象可直接画列线图”的验收。

主agent已报告在基线之后新增typed binary Logistic fusion/loaded bundle系数适配，沿用既有points转换、不另fit，并拒绝weighted/multiclass/pipeline。**截至本审计记录，该改动测试待执行，没有在本报告记为通过。** 后续应附实际提交与测试，并保持其限定范围，不把这一个桥接修复泛化为全部筛选图、Rad-score或学习曲线。

### 2. 稀疏筛选、Rad-score和预测概率是不同输出

comp1传统组学任务cell44/46可见 `lasso_cv_coefs/lasso_cv_efficiency`，cell48可见 `sklearn.linear_model.Lasso`；comp6特征重要性及公式cell3可见 `plot_feature_importance`。这些是迁移用户会寻找的展示步骤，不足以单独确认Onekey的评分公式和优化参数。

本包Rust稀疏筛选和 `SparseFit.linear_scores` 已存在；但基线 `NativeFeatureSelector` 公开的是所选特征、单因素表和alpha，没有旧 `linear_score()`、`coefficients_`、`lasso_path_`、`lasso_cv_results_` 这一组接口。`plot_lasso_path/plot_lasso_cv` 依赖旧属性，直接把native selector传入并不兼容。

旧Python `linear_score` 是已缩放、保留特征的LASSO线性组合；它不是概率，也不是任意Logistic的logit。若补入口，应复用既有Rust拟合状态和训练缩放/列顺序，定义清楚输出含义。绘图不能重新在全数据上跑LassoCV，或把候选whole-pipeline任务分数改称旧MSE曲线。

### 3. 学习曲线不能与筛选路径/CV误差图混称

comp1任务cell61、comp2二分类cell43 **import** `onekey_algo.custom.components.comp1.plot_learning_curve`，静态AST没有该函数的调用。这只证明符号出现，不证明任何学习曲线已绘制。

本包 `plot_learning_curve` 实际导入sklearn，使用 `StratifiedKFold`，没有 `groups` 参数。它是可选Python/独立行用途，不能写成原生患者分组学习曲线。原生nestedCV已经成立，不意味着随训练量重拟合全pipeline的学习曲线也完成。若以后扩展，需明确患者训练子集、每次重拟合筛选/缩放、任务指标和重复数；本轮不从菜单自动扩大算法工作。

### 4. 三任务指标完整，不代表每个旧绘图helper支持三任务

native训练结果按任务给出 `auc`、`macro_ovr_auc` 或 `rmse`，评价保留多分类macro/micro/weighted、回归MAE/RMSE/R²。基线 `plot_cv_results` 仍硬编码 `y='auc'` 和0–1纵轴。原生多分类/回归CV表不能直接当binary AUC表画；这属于小型展示桥接问题，不能写成计算未实现或把RMSE缩放进0–1掩盖含义。

可考虑显式metric/task、指标方向和纵轴策略，保留旧binary默认；只读取结果，不重训、不重新挑最佳划分。`plot_metrics(metric=...)` 已允许指定指标，但同样需要使用正确的队列和字段。

### 5. batch、ICC筛选、中心验证要分开

本包batch提取的外层并行/失败/缓存、显式中心留出和分中心评价均已实现；**不等于ComBat**。来源 `comp8-Modules/Combat批次矫正.ipynb` cell2有 `neuroCombat.neuroCombat`，这仍属于用户明确暂缓项目。

本包 `icc(..., kind='ICC(2,1)'或'ICC(3,1)')` 接完整平衡subject×reader矩阵并返回CI；来源comp8 ICC cell3、comp1 ICC cell11中的 `icc_filter_features` 名称不能证明其ICC类型。当前没有统一多reader FeatureTable→按ID对齐→逐特征ICC→阈值筛列的便利工作流。算法支持和自动表格筛选不能混写；小封装可复用同一内核，不增加另一套ICC数学。

`docs/reliability-native.md` 当前还称public adapter验证subject/reader身份与唯一pairs；实际Python `icc` 只接matrix、kind、confidence，没有ID参数，转换为NumericBuffer后没有身份可核对。这句应改成调用者先按明确ID构造矩阵的责任；行列完整不等于已证明subject/reader键唯一。这是文档边界问题，不推翻ICC数值验收。

### 6. 重采样、Original子集和filter是不同范围

本包原生读取完整affine、显式spacing/reference-grid重采样、mask nearest-only、强度/ROI操作都有公开入口和冻结定义。它不自动复现Onekey `ConventionalRadiomics` 的隐藏配置，也不把PyRadiomics的所有family纳入原生。

原生仅明确55项强度/纹理特征加shape_basic。LoG/Wavelet/LBP、复杂预处理/N4/配准等继续按用户暂缓；用户提供已经处理好的影像/特征可接入，但处理来源必须保留。没有“读入filtered表所以原生支持filter”的推论。

## 当前文档是否扩大了支持

`docs/support-matrix.md` 的native/optional/deferred边界与源码总体一致，明确“校准曲线不是拟合recalibrator”，没有把所有绘图、SHAP、ComBat或filter列入原生。`docs/api.md` 开头和后半段分界已明确区分native与legacy；`docs/legacy-python.md` 也明示旧后端。不能忽略这些上下文后反指文档承诺了所有native能力。

仍有孤立阅读风险：`docs/scope.md` 的初始能力表包含LASSO formula/paths、八模型、nomogram和learning curves；`docs/provenance.md` 的旧nomogram说明描述sklearn型对象，`docs/validation.md` 的nomogram/八模型记录明确是历史验证。建议在初始scope/provenance处加醒目的历史/后端提示，或在表中直接注明reference-only与typed-native兼容范围。**历史测试通过不能转移成新的native对象测试通过。**

“概率校准”也应区分曲线与校准器拟合：原生曲线/分箱已有，当前没有声明完整Platt/isotonic训练状态。原计划词语不能由Onekey菜单替代算法定义；若用户需要后者，应先冻结具体数学与患者分组校准协议，而不是在本报告悄悄加入新发布门槛。

## 建议主agent处理的最小事项

| 优先级 | 建议 | 允许的边界 |
|---|---|---|
| P1 | 验证typed binary native融合→nomogram桥接 | 与真实native概率/加载模型对照；不fit，不支持weighted或multiclass列线图 |
| P1 | 明示native selector与旧Rad-score/路径/CV图属性不兼容 | 若只修文档即可；如扩便利接口，只读已有类型化拟合状态，不另造算法 |
| P1 | 三任务CV图支持显式metric或清楚标binary-only | 不改训练、分数、阈值、Golden |
| P2 | 标learning_curve为可选独立行用途；不声称patient groups | 不为迁移菜单扩充原生训练量曲线算法 |
| P2 | 区分typed ICC与多reader表筛列便利流程 | 可先加可执行按ID对齐示例，复用现有ICC |
| P2 | 初始scope/provenance加历史/后端提示 | 当前support matrix/API的清楚边界保留 |
| P3 | mRMR/SHAP等未承诺菜单保留差异说明 | 不要求扩算法/依赖，不当作本轮硬门槛 |

这里的优先级表示用户迁移风险和桥接便利性，不改变原用户完成标准。未执行Onekey而无法确认的内部算法，不成为复现目标；暂缓菜单不变成新义务。本报告只提出具体的小修和边界说明，未实现这些修改。

## 可复查与隐私边界

提取工具：[audit_onekey_notebook_symbols.py](../tools/audit_onekey_notebook_symbols.py)。例如：

```powershell
python tools/audit_onekey_notebook_symbols.py --source-root D:/OnekeyPlatfrom50/onekey_comp --output E:/audit-output/static-symbols.json
```

工具只使用标准库、JSON和AST，不执行源单元，不导入Onekey，不打开 `.pyc/.pyd`，不输出markdown或code正文、不读取表格、图片或模型。checkpoint排除，废弃目录若作为证据则保留路径标识；import-only也单列。别名解析仅静态import级，动态对象类型/分支是否执行均不判定。输出topic只是检索提示，不定义算法。

外置完整符号清单：`E:/radiomics-build/onekey-gap-audit-v1/static-symbols-v3.json`，SHA256 `c194ef26cc3e2175296bea47b152129f9cf6b80b4eb3c6274a49d3c97d919cbf`。结构化报告保留48个来源文件摘要和所引用单元符号，无第三方正文或已有患者输出。README内容只记录SHA和规范topic标签。

本包源码引用固定于基线；主agent后续修复应以新增提交/验收记录附注，而不是改写基线为“当时已经支持”。
