# 快速开始与学习路线

先完成[安装](installation.md)，在完整仓库内运行。所有命令都使用新的输出目录。

## 一分钟检查安装与提取

```bash
python -c "import ov_radiomics as rad; print(rad.backend_info())"
python examples/native/01_extract.py --output artifacts/first-extraction
```

检查 `artifacts/first-extraction/python-api/features.csv`：三行样本的 `original_shape_basic_VoxelCount` 都为24，`original_shape_basic_Volume_mm3` 都为576。`diagnostics.csv`给出失败/质量信息，`config.json`记录实际参数与实现标识。这是合成软件演示，不是临床数据。

## 用Notebook理解每一步

```bash
jupyter lab
```

打开 `website/tutorials/` 下的Notebook。每页包含逐步代码和已保存输出，从头重新运行会创建新目录。不要跳过环境、ID和数据检查单元。

| 目标 | 教程 | 数据性质 |
|---|---|---|
| 理解几何和特征 | [01 基础提取](tutorials/01_extract.ipynb) | 自建小型NIfTI |
| 影像＋临床、共享划分与融合 | [02 二分类](tutorials/02_image_clinical_binary.ipynb) | 固定合成患者/病灶 |
| 换任务，不把类别当连续值 | [03 多分类与回归](tutorials/03_multiclass_regression.ipynb) | 固定合成表格 |
| 多区域、Delta与局部生境 | [04 研究组合](tutorials/04_regions_delta_habitat.ipynb) | 合成配对数据 |
| patch到患者及图像导出 | [05 轻量病理](tutorials/05_patch_patient.ipynb) | 合成RGB图像 |
| embedding按ID绑定与局部概率 | [06 Embedding](tutorials/06_embedding_overlay.ipynb) | 合成embedding与显式坐标 |
| 真实公开文件、许可与ROI预览 | [07 公开CT](tutorials/07_public_ct.ipynb) | IBSI PAT1真实公开CT/GTV |

## 换成自己的文件

批处理和自己的研究优先用 `examples/real/` 的文件入口：

```bash
python examples/real/nifti_extract.py --help
python examples/real/train_table.py train --help
python examples/real/train_table.py predict --help
python examples/real/patch_features.py --help
```

这些入口接受已有文件，不创建临床结局，不根据行数猜测患者或tile身份。先读[输入契约](reference/user-guide/input-contracts.md)，再按[文件工作流](reference/user-guide/file-workflows.md)构建自己的清单和列角色配置。

错误排查见[结果解读与常见错误](reference/user-guide/results-and-errors.md)。保留完整报错、`backend_info()`、配置和脱敏表头，不要直接上传患者影像或临床表到公开issue。
