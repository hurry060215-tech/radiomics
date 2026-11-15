# radiomics

独立 Rust 影像组学与轻量病理分析项目。Python提供API、表格和绘图，独立CLI调用同一
计算核心。发行包为 `ov-radiomics`，导入为 `ov_radiomics`，版本保持0.1.0。

| 从哪里开始 | 入口 |
|---|---|
| 安装、平台要求和预编译产物 | [安装指南](website/installation.md) |
| 无数据先跑通 | [快速开始](website/quickstart.md) |
| 逐单元学习，查看实际输出和图 | [八篇Notebook](website/tutorials/index.md) |
| 想看真实公开影像 | [IBSI PAT1 CT/ROI](website/tutorials/07_public_ct.ipynb) |
| 真实公开表格的建模演示 | [UCI WDBC的569条记录](website/tutorials/08_public_tabular.ipynb) |
| 使用自己的NIfTI、ROI、临床表 | [文件驱动研究流程](docs/user-guide/file-workflows.md) |
| 使用patch、mask、embedding | [轻量病理教程](docs/user-guide/pathology.md) |
| 想核对方法与性能 | [技术报告](docs/technical-report.md)、[支持矩阵](docs/support-matrix.md) |

```bash
git clone https://github.com/hurry060215-tech/radiomics.git
cd radiomics
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install .
python -c "import ov_radiomics as rad; print(rad.backend_info())"
python examples/native/01_extract.py --output artifacts/first-extraction
```

源码安装需Rust1.85+及系统链接器；已有Windows/Linux wheel不需要Rust。基础原生流程
不依赖sklearn、SciPy、statsmodels、PyRadiomics、SimpleITK、ov或GPU。wheel不会附带
独立CLI，需要下载对应Actions产物或单独Cargo构建。

核心能力覆盖：标量2D/3D NIfTI、ROI和原生预处理；明确列出的55项Original统计/纹理
及11项基础形态；患者分组的二分类、多分类、回归；原生筛选、模型、统计、评价；
版本化`.rad`保存与无标签外部预测；普通病理图像QC、特征、已有mask统计、严格ID对齐、
切片/患者聚合与真实tile分数覆盖图。具体边界以支持矩阵为准。

教程分开合成演示、真实公开数据和用户已有文件。第08篇使用真实UCI细胞核测量和
B/M结局，按记录ID分组；它不是本包提取的影像组学特征，也不能证明569位独立患者。
合成分数不代表
诊断效果；公开CT为CC-BY-NC-3.0且没有建模结局；真实工作流不会替用户生成结局、
猜患者身份或从DPI推断病理尺度。原始公开CT不入Git。

原生训练/筛选/临床API请显式传 `backend="rust"`。兼容默认可能是可选Python路径；
`auto`只接受原生扩展，不静默回退。测试集不能参与筛选、填补、阈值选择或参数拟合。

本项目参考ov的安装、教程、数据与API文档组织方式，保持独立实现、安装和发布。
性能未全面达到目标：真实CT提取约0.400倍参考速度，端到端约1.225倍加速。
IBSI采用独立轨道，raw Kurtosis已知定义差异不标为通过。完整结果见技术报告。

没有自动发布PyPI/crates.io、修改版本或修改ov源码。开发、文档预览和反馈方法见
[English README](README.md)、[贡献指南](CONTRIBUTING.md)与[安全说明](SECURITY.md)。
