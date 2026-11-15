# API 与 CLI

Python 导入名为 `ov_radiomics`，发行包名为 `ov-radiomics`。

## Python 原生入口

| 任务 | 入口 | 重要约定 |
|---|---|---|
| 读取队列 | `rad.read_cohort(...)` | 样本 ID 唯一，患者可有多个病灶 |
| 提取特征 | `rad.extract(..., backend="rust")` | 明确分箱、几何与 ROI |
| 筛选 | `rad.FeatureSelector(..., backend="rust")` | 所有拟合进入训练划分 |
| 建模 | `rad.train_models(..., split_plan=plan, backend="rust")` | binary / multiclass / regression |
| 评价 | `rad.evaluate(...)` | 默认患者级；保留队列身份 |
| 模型加载 | `rad.load_model(path)` | 类型化 .rad；输入可按名称重排 |
| 病理 | `rad.pathology` | patch、QC、聚合、真实 tile 覆盖图 |
| 容器适配 | `rad.interop` | 按 tile ID 校验，不能仅比较行数 |

训练、筛选和临床统计的兼容默认后端仍可能为 Python；原生教程显式指定 Rust。未知配置会报错。

## 命令行

```bash
radiomics --help
radiomics check --help
radiomics extract --help
radiomics train --help
radiomics predict --help
radiomics evaluate --help
radiomics fuse --help
radiomics pathology --help
```

以同一提交编译的 CLI `--help` 为参数依据。配置文件与输出示例见各教程源码。独立 CLI 运行不依赖 Python。

## 详细参考下载

{download}`API 说明源码 <../docs/api.md>`

{download}`CLI 工作流参数说明 <../docs/cli-workflows-native.md>`

{download}`病理 CLI 说明 <../docs/pathology-cli-native.md>`
