# 教程：逐步运行，再换成自己的数据

先[安装](../installation.md)，从仓库内启动Jupyter。每页都有实际代码、
中间结果和运行图，按顺序执行单元。重新完整运行会使用新目录，避免覆盖结果。

| 教程 | 输入 | 学习重点 |
|---|---|---|
| 01 | 合成微型NIfTI/ROI | shape、affine、24体素ROI及特征 |
| 02 | 合成影像＋临床 | 重复病灶、患者共享划分、OOF融合 |
| 03 | 合成多分类/连续目标 | 原生任务、评价、保存和外部预测 |
| 04 | 合成多区域/时间点矩阵 | 瘤周、严格键、Delta、局部生境 |
| 05 | 合成普通PNG patch | QC、层级聚合和真实tile特征图 |
| 06 | 合成外部embedding | 打乱行序的ID校验、tile概率图 |
| 07 | 真实公开PAT1 CT/GTV | 固定来源/许可、真实ROI预览、文件提取 |

前六篇是离线软件演示，不是真实临床队列。第七篇显式下载或复用公开数据，
原始CT许可为CC-BY-NC-3.0；没有临床结局，所以不训练诊断模型。

```{toctree}
:maxdepth: 1

01_extract
02_image_clinical_binary
03_multiclass_regression
04_regions_delta_habitat
05_patch_patient
06_embedding_overlay
07_public_ct
```

## 下载Notebook

下载单页仍需完整仓库中的示例辅助函数与固定数据，不能只把Notebook放到空目录。

{download}`01 提取 <01_extract.ipynb>` ·
{download}`02 二分类 <02_image_clinical_binary.ipynb>` ·
{download}`03 多分类/回归 <03_multiclass_regression.ipynb>` ·
{download}`04 研究组合 <04_regions_delta_habitat.ipynb>` ·
{download}`05 病理patch <05_patch_patient.ipynb>` ·
{download}`06 Embedding <06_embedding_overlay.ipynb>` ·
{download}`07 公开CT <07_public_ct.ipynb>`

## 自己的数据

清单、列角色、患者单位、真实坐标和输出解读见
[输入契约](../reference/user-guide/input-contracts.md)、
[文件工作流](../reference/user-guide/file-workflows.md)、
[病理指南](../reference/user-guide/pathology.md)和
[常见错误](../reference/user-guide/results-and-errors.md)。
`examples/real/`入口不生成临床结局或猜测患者身份。
