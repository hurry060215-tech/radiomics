# 从影像与表格，走到可审查的研究结果

<div class="portal-intro">
<p class="portal-kicker">RADIOMICS · 独立研究软件</p>
<p class="portal-lead">Python 提供交互与绘图，原生核心支撑无 Python 的 CLI。明确选择每一步的后端，围绕输入、患者划分、参数与输出，完成影像组学和轻量病理分析；速度、峰值内存和数值误差分别验证。</p>
<div class="portal-actions"><a class="portal-primary" href="installation.html">安装与首次运行</a><a href="tutorials/index.html">选择学习路线 →</a><a href="user-guide/index.html">使用自己的数据 →</a></div>
</div>

## 从你已有的输入开始

<div class="portal-grid">
<a class="route-card" href="tutorials/radiomics.html"><span class="route-label">01 · 影像组学</span><h3>NIfTI 与 ROI</h3><p>几何检查 → Original 特征 → 多区域、Delta 与基础生境。</p><span class="route-output">输出：特征表、参数与质量记录</span></a>
<a class="route-card" href="tutorials/modeling.html"><span class="route-label">02 · 建模与统计</span><h3>特征表与真实结局</h3><p>列角色与患者划分 → 训练内拟合 → 分类／回归评价与外部预测。</p><span class="route-output">输出：OOF、评价图与 .rad 模型</span></a>
<a class="route-card" href="tutorials/pathology.html"><span class="route-label">03 · 轻量病理</span><h3>普通 patch 与 embedding</h3><p>QC 与 ID 对齐 → tile / slide / patient 聚合 → 实际 tile 分数覆盖图。</p><span class="route-output">输出：层级特征、QC 与坐标图</span></a>
<a class="route-card" href="tutorials/real-data.html"><span class="route-label">04 · 真实公开数据</span><h3>公开 CT 与数值记录</h3><p>核对来源与许可 → 读取真实文件 → 检查特征或固定记录留出结果。</p><span class="route-output">输出：可追溯文件、真实 ROI 预览与结果</span></a>
</div>

## 跟着教程学，带着自己的文件做

第一次使用，先完成[安装检查与快速开始](quickstart.md)，再进入[八篇可逐单元运行的教程](tutorials/index.md)。01–06 使用固定合成数据学习软件行为；07 使用单例公开 CT；08 使用真实 UCI 细胞核测量表。每条路线都说明输入、步骤、输出和替换为自己数据的方法。

已有研究文件时，从[用户指南](user-guide/index.md)准备 cohort、临床表、patch manifest 或 embedding 行 ID。影像与临床路径保留患者分组；病理路径保留 patient→slide→tile 层级和真实坐标。程序不从文件名或特征值猜结局。

## 先确认支持范围，再解释结果

原生能力覆盖明确的 Original 特征子集、三任务建模和普通图像病理工具；自动分割、深度模型训练、通用 WSI 解码和配准不在本轮范围。请查[支持矩阵](reference/support-matrix.md)和[验证说明](validation.md)。Python 原生示例显式指定 `backend="rust"`，CLI 调用同一计算核心。

合成分数只证明流程可运行。单例 CT 没有训练结局；UCI source ID 不能证明患者独立。患者模型概率也不能作为局部 tile 预测图。正确性、研究效能与性能分别报告，详见[验证与开发](development/index.md)。

```{toctree}
:hidden:
:maxdepth: 3

安装 <installation>
教程 <tutorials/index>
用户指南 <user-guide/index>
生态分工 <guide/ecosystem>
English <en/index>
API <reference>
验证与开发 <development/index>
```
