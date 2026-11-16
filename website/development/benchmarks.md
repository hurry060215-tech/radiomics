# Benchmark v2：三套独立报告

正确性门槛与性能目标分开验收。影像组学、建模统计、病理不合成一个总加速比；逐项报告参考版本、数据、资源、原始观察和失败。小矩阵加速不能外推为真实 CT 的通用性能。

```{toctree}
:maxdepth: 1

Radiomics 测量 <../reference/radiomics-benchmark-v2-results>
Machine Learning / Statistics 与 Pathology 测量 <../reference/learning-pathology-benchmark-v2-results>
L5 部署与启动成本 <../reference/deployment-benchmark-v2-results>
Python/Rust数组边界诊断 <../reference/native-performance-diagnosis>
Sample构造有限优化及未观察到的内存收益 <../reference/sample-constructor-optimization>
```

Radiomics 按算子、提取、端到端和吞吐分别评估；L5 单列冷进程、产物与安装 payload 成本。模型收敛、预测一致性、ID/坐标正确性与泄漏控制在各自套件验收。报告保留未达目标和监控边界，不以单一最快结果代替完整数据。

{download}`预声明 Benchmark v2 规范 <../../docs/benchmark-v2.md>`
