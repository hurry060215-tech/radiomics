# API与CLI参考

以下页面从仓库内维护的Markdown生成，网页与源码共用说明。数组/表格转换和绘图
留在Python，数值操作调用同一Rust核心。所有原生示例显式指定 `backend="rust"`。

```{toctree}
:maxdepth: 1
:caption: Python与数据

reference/api
reference/image-native
reference/features-native
reference/native-extraction
reference/table-io-native
reference/preprocessing-native
reference/learning-native
reference/models-native
reference/evaluation-native
reference/clinical-native
reference/reliability-native
reference/research-native
reference/pathology-pixels
reference/pathology-aggregate
reference/interop
```

```{toctree}
:maxdepth: 1
:caption: CLI与验证报告

reference/cli-workflows-native
reference/pathology-cli-native
reference/support-matrix
reference/technical-report
reference/radiomics-benchmark-v2-results
reference/learning-pathology-benchmark-v2-results
reference/deployment-benchmark-v2-results
```

CLI是单独构建产物，wheel不会自动附带命令。运行 `radiomics --help` 核对当前程序，
配置中输入列、类别顺序、患者单位和实际划分都必须明确；相同fold编号不能代替患者集合验证。
