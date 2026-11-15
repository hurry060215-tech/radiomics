# 安装与快速开始

当前预览对应源码提交 `c21cfd9`。尚未声明 PyPI 正式发布。

从仓库根目录安装原生扩展，需要 Rust 工具链和平台链接器：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install .
python -c "import ov_radiomics as rad; print(rad.backend_info())"
```

预编译 wheel 可直接用 `python -m pip install /path/to/package.whl` 安装，无需 Rust。请使用与教程提交一致的构建。

先运行基础提取教程；所有输出使用一个尚不存在的新目录。基础原生教程不需要 sklearn、SciPy、statsmodels、PyRadiomics 或 ov。

```bash
python examples/native/01_extract.py --output artifacts/tutorial-01
```

应得到三个样本的特征表；示例 ROI 各含 24 个体素，物理体积为 576 mm³。实际图像的单位和几何必须单独检查。
