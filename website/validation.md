# 验证与适用范围

## 正确性与性能

发布硬门槛覆盖适用的正确性、数值稳定性、患者泄漏控制、保存加载和独立部署。性能目标单独报告；正确但较慢的实现必须披露并优化。

Radiomics、Machine Learning & Statistics、Pathology 使用三个独立套件，不能汇总成一个加速比。

## 双轨验证

Python 兼容轨道固定公开 PyRadiomics 3.0.1。IBSI 轨道使用独立参考值和明确配置。当前 digital phantom 范围包含 55 项一阶/纹理特征，其中 54 项声明符合 IBSI；原始峰度保留已知定义差异。该结论不能推广到所有 2D、CT 重采样或预处理配置。

## 研究边界

01–06使用合成数据；07为一例真实公开CT且没有训练结局；08为真实UCI既有细胞核测量及B/M标签，source记录组不证明患者独立，固定留出不构成外部临床验证。所有模型分数均需按各自数据与划分解释。病理实例计数限定patch层级，重叠patch不直接累加全切片细胞数；患者概率不广播为局部tile预测。

## 原始报告

{download}`Benchmark v2 规范 <../docs/benchmark-v2.md>`

{download}`独立验收报告 <../docs/acceptance-v2-report.md>`

{download}`功能范围说明 <../docs/scope.md>`
