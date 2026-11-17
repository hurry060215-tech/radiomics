# 教程：按任务选择路线

先完成[安装](../installation.md)，再从完整仓库启动 Jupyter。01–08包含代码、中间检查、保存的实际输出和解释；09–10提供读取本机数据的源码，执行输出只留本机。按顺序执行单元，使用新的输出目录。网页构建不自动下载数据或训练。

## 五条学习路线

<div class="portal-grid">
<a class="route-card" href="radiomics.html"><span class="route-label">影像组学 · 01 → 04</span><h3>从 ROI 到 Original 特征</h3><p>NIfTI / mask → 几何与提取 → 多区域与时间点 → 特征和质量记录。</p><span class="data-badge synthetic">合成入门</span></a>
<a class="route-card" href="modeling.html"><span class="route-label">建模与统计 · 02 → 03</span><h3>从特征到可保存的模型</h3><p>特征与结局 → 患者分组 → 分类／回归 → 评价、.rad 与外部预测。</p><span class="data-badge synthetic">合成入门</span></a>
<a class="route-card" href="pathology.html"><span class="route-label">轻量病理 · 05 → 06</span><h3>从 patch 到层级结果</h3><p>普通图像 / embedding → QC、ID 对齐 → 聚合 → 真实 tile 分数覆盖图。</p><span class="data-badge synthetic">合成入门</span></a>
<a class="route-card" href="real-data.html"><span class="route-label">真实数据 · 07 / 08</span><h3>检查真实文件与数据来源</h3><p>公开 CT 或细胞核测量表 → 许可与 SHA → 提取或固定记录留出 → 可追溯结果。</p><span class="data-badge real">真实公开数据</span></a>
<a class="route-card" href="local-data.html"><span class="route-label">本地数据 · 09 → 10</span><h3>自备影像／ROI 到原生模型</h3><p>参数化本机路径 → 几何质控 → 55项特征 → 固定划分与模型重载。</p><span class="data-badge">代码公开，输出留本机</span></a>
</div>

```{toctree}
:hidden:
:maxdepth: 2

影像组学 <radiomics>
建模与统计 <modeling>
轻量病理 <pathology>
真实数据路线 <real-data>
本地样例路线 <local-data>
```

## 十篇可执行教程

| 教程 | 数据来源 | 执行后检查什么 |
|---|---|---|
| [01 · 基础提取](01_extract.ipynb) | 合成微型 NIfTI / ROI | shape、affine、24 体素 ROI、特征和 QC |
| [02 · 影像与临床二分类](02_image_clinical_binary.ipynb) | 合成影像与临床表 | 重复病灶、共享患者划分、OOF 与融合 |
| [03 · 多分类与回归](03_multiclass_regression.ipynb) | 合成类别与连续目标 | 多任务评价、保存加载和外部预测 |
| [04 · 区域与时间点](04_regions_delta_habitat.ipynb) | 合成多区域 / 时间点矩阵 | 瘤周、键对齐、Delta 与局部生境 |
| [05 · Patch 到患者](05_patch_patient.ipynb) | 合成普通 PNG patch | QC、层级聚合和实际 tile 特征图 |
| [06 · Embedding 与覆盖图](06_embedding_overlay.ipynb) | 合成外部 embedding | 打乱行序的 ID 校验、实际 tile 概率图 |
| [07 · 公开 CT](07_public_ct.ipynb) | 真实 PAT1 CT / GTV，CC-BY-NC-3.0 | 来源、许可、真实 ROI 预览和文件提取 |
| [08 · 公开数值表](08_public_tabular.ipynb) | 真实 UCI 记录与 B/M 标签，CC-BY-4.0 | 固定记录留出、原生 LR、保存加载与无标签预测 |
| [09 · 本地图像与 ROI](09_local_image_features.ipynb) | 用户自备 NIfTI／mask、标签与固定分组 | 显式几何处理、55项特征、QA与输入哈希 |
| [10 · 本地固定划分建模](10_local_modeling.ipynb) | 09生成的55列与既有107列特征 | 共享划分、OOF／val、模型重载和无标签推理 |

01–06 是离线软件演示，不能作为真实临床效能证据。07 没有临床结局，不训练诊断模型。08 的细胞核测量不是本包提取的 radiomics；唯一 source ID 不证明患者独立，留出分数不构成外部临床验证。来源、署名和使用限制见[数据与许可](../datasets.md)。

## 下载与替换数据

下载单页仍需仓库内的示例辅助函数与固定数据。

{download}`01 提取 <01_extract.ipynb>` ·
{download}`02 二分类 <02_image_clinical_binary.ipynb>` ·
{download}`03 多分类/回归 <03_multiclass_regression.ipynb>` ·
{download}`04 研究组合 <04_regions_delta_habitat.ipynb>` ·
{download}`05 病理 patch <05_patch_patient.ipynb>` ·
{download}`06 Embedding <06_embedding_overlay.ipynb>` ·
{download}`07 公开 CT <07_public_ct.ipynb>` ·
{download}`08 公开表格 <08_public_tabular.ipynb>` ·
{download}`09 本地图像 <09_local_image_features.ipynb>` ·
{download}`10 本地建模 <10_local_modeling.ipynb>`

换用自有数据时，按[用户指南](../user-guide/index.md)准备文件、列角色、患者单位和坐标。`examples/real/` 入口读取提供的身份与结局，不生成临床标签。
