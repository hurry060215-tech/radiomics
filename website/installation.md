# 安装

发行包名为 **`ov-radiomics`**，导入名为 **`ov_radiomics`**。版本仍为0.1.0；仓库构建和 Actions 产物不代表已在 PyPI/crates.io 正式发布。

## 选择安装方式

| 情况 | 安装方式 | 是否需要 Rust |
|---|---|---|
| 已下载对应平台的 wheel | `python -m pip install /path/to/ov_radiomics-....whl` | 否 |
| 从 GitHub 源码安装 | 下文源码步骤 | 是，1.85或更新版本，并有链接器 |
| 只运行独立 CLI | 下载 `native-cli-ubuntu-latest` 或 `native-cli-windows-latest` 产物 | 否，也不需要Python |

预编译产物在[成功的 Actions 工作流](https://github.com/hurry060215-tech/radiomics/actions/workflows/tests.yml)底部的 Artifacts；下载需要登录 GitHub。wheel 的平台标签必须匹配自己的平台。Windows 的 `.exe` 不能直接在 Linux 使用；Linux CLI 解压后先运行 `chmod +x radiomics`。macOS 尚没有本轮的安装验收结论。

## 从源码安装

准备 Python3.9或更新版本、Rust1.85或更新版本。Windows 需要 Visual Studio Build Tools 的 C++ 构建工具和 Windows SDK；Linux 需要系统 C/C++ 链接工具。先检查 `rustc --version` 和 `cargo --version`。

```bash
git clone https://github.com/hurry060215-tech/radiomics.git
cd radiomics
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux shell 激活：

```bash
source .venv/bin/activate
```

然后安装当前检出的源码和 Notebook 环境：

```bash
python -m pip install .
python -m pip install jupyterlab ipykernel
python -c "import ov_radiomics as rad; print(rad.__version__); print(rad.backend_info())"
```

确认 `metrics_backend` 为 `rust`。基础依赖为 NumPy、pandas、matplotlib。原生 NIfTI、特征、建模和普通病理流程不需要 sklearn、SciPy、statsmodels、PyRadiomics、SimpleITK、ov、WSIData、Torch或GPU。

仅在运行明确标为外部/参考后端的旧教程时安装：

```bash
python -m pip install '.[reference,image]'
```

## 冻结源码 tag 的构建前置

当前维护分支已提交 PEP 517 元数据所需的 binding README，可按上面的裸源码步骤
安装。不可变的 `v0.1.0` Git tag／GitHub 源码归档早于这项修复；构建该 tag 时，
先从其根目录运行自带的准备工具，再安装：

```bash
python tools/package_native.py --output artifacts/tag-source-assets
python -m pip install .
```

准备命令只生成该源码版本的本地打包 assets，不修改 tag 或冻结参考。
已准备好的安装 sdist 自带所需 assets；GitHub 的源码 zip／tar 归档不能当成该 sdist。
这里保留历史 tag 的真实前置条件，不改写其文档快照或声称裸 tag 已修复。

## 独立 CLI

Python wheel 不会自动安装名为 `radiomics` 的命令。下载独立 CLI 产物，或从源码构建：

```bash
cargo build --release -p ov-radiomics-cli --locked
```

Linux 程序位于 `target/release/radiomics`，Windows 位于 `target/release/radiomics.exe`。后续 `radiomics` 命令指这个独立程序，需要使用完整路径或加入自己的 PATH。

## 第一遍运行

```bash
python examples/native/01_extract.py --output artifacts/first-extraction
jupyter lab
```

打开 `website/tutorials/01_extract.ipynb` 按顺序运行。基础演示产生三个合成样本，每个ROI有24体素、体积576 mm³。下一步选择[公开真实CT](tutorials/07_public_ct.ipynb)或[自己的文件工作流](reference/user-guide/file-workflows.md)。

结果目录默认不能覆盖，第二遍请换目录。扩展不可用时，先检查 Jupyter 内核是不是安装 wheel 的同一个 Python，不能切换到参考后端掩盖原生安装失败。

```{toctree}
:hidden:
:maxdepth: 1

快速开始 <quickstart>
```
