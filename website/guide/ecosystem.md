# 生态分工：影像工具、组学接口与独立核心

[English](../en/ecosystem.md) · [安装](../installation.md)

独立包负责明确范围内的特征提取和建模。OmicVerse 的可选适配层提供命名空间入口，
保留独立安装、计算核心和许可证边界。用户可以直接使用独立包或 CLI。

| 入口 | 主要职责 | 维护与依赖边界 |
|---|---|---|
| `ov.radiology` | DICOM 几何、测量、阅片 | 由 OmicVerse 维护的影像工具 |
| `ov.radiomics` | 对独立组学包的可选、延迟访问 | 适配代码位于 OmicVerse 仓库 |
| `ov_radiomics` | 明确的原生特征子集、具名表格、建模和评价 | 独立发行的 MIT 包 |
| `radiomics` CLI | 文件驱动提取、训练和预测 | 调用共享 Rust 核心，不需要 Python |

```{mermaid}
flowchart TD
    A[ov.radiology：几何、测量、阅片] -->|显式提供影像与 mask| C[独立 ov_radiomics 包]
    B[ov.radiomics：可选延迟适配] -->|独立安装边界| C
    C --> D[共享 Rust 核心]
    E[独立 radiomics CLI] --> D
```

图中是职责与调用关系，不为组学接口新增 DICOM 转换器，也不把 CT 体素包装成
AnnData 表达矩阵。外部已处理数组、mask 和 embedding 只按已有输入契约进入，
保留真实 ID、几何和处理来源。

适配层只在访问属性时加载独立包；`import omicverse` 不应新增影像后端或原生扩展
的即时导入。只有兼容 wheel 和相应 OmicVerse extra 实际发布后，才能把该入口写成
公开安装承诺。独立包不会反向依赖 OmicVerse。

独立包使用 MIT；OmicVerse 使用 GPL-3.0。GPL 项目可引用 MIT 包；不能把 GPL 实现
复制回独立包后按 MIT 发布。详情见[贡献政策](../contributing.md)。

原生、可选 Python 和明确暂缓的范围见[能力边界](../reference/native-boundaries.md)。
共享核心不意味着所有旧 Python 结果对象都支持原生绘图或序列化。

## 文档版本与后续工作

`latest/` 跟随维护分支。`v0.1.0/` 从实际发行 tag 构建并保留，tag 和部署尚未发生时
明确记为待发布。页脚的提交号说明本次页面由哪个源码版本生成。

Read the Docs、conda-forge 和 Docker 发行均为后续事项，不属于本轮发布承诺。
