# API 与 CLI 参考

这里按调用对象查函数、参数、数据约束与错误。页面由仓库维护的 Markdown 生成，网站与源码共用说明。Python 负责转换与绘图，原生数值操作复用 Rust 核心；原生示例显式指定 `backend="rust"`。

第一次组织研究文件，请先读[用户指南](user-guide/index.md)。算法验收与性能报告集中在[验证与开发](development/index.md)。

```{toctree}
:maxdepth: 1
:caption: 入口与数据

reference/api
reference/python-native
reference/table-io-native
```

```{toctree}
:maxdepth: 1
:caption: 影像与特征

reference/image-native
reference/features-native
reference/native-extraction
reference/research-native
```

```{toctree}
:maxdepth: 1
:caption: 建模与统计

reference/preprocessing-native
reference/learning-native
reference/models-native
reference/evaluation-native
reference/clinical-native
reference/baseline-native
reference/reliability-native
```

```{toctree}
:maxdepth: 1
:caption: 轻量病理与互操作

reference/pathology-pixels
reference/pathology-aggregate
reference/interop
```

```{toctree}
:maxdepth: 1
:caption: 独立 CLI

reference/cli-workflows-native
reference/pathology-cli-native
```

CLI 是单独构建产物，wheel 不会自动附带命令。运行 `radiomics --help` 核对当前程序；患者集合、列角色、类别顺序和预测层级必须明确，相同 fold 编号不能代替划分集合验证。
