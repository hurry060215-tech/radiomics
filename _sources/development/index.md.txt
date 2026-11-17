# 验证与开发

这里集中说明软件支持范围、算法验证、独立性能实验与贡献流程。首次运行和数据准备请使用[教程](../tutorials/index.md)或[用户指南](../user-guide/index.md)。

| 要检查的问题 | 入口 |
|---|---|
| 当前哪些功能原生支持，哪些暂缓？ | [支持矩阵](../reference/support-matrix.md) |
| Python 兼容与 IBSI 定义如何分别验证？ | [正确性与研究边界](../validation.md)、[方法与技术报告](methods.md) |
| 真实 CT 的性能是否达到目标？ | [三套独立基准](benchmarks.md)，保留慢项和原始测量范围 |
| 如何修改项目与严格构建文档？ | [开发与贡献](../contributing.md) |

```{toctree}
:maxdepth: 2

验证与适用范围 <../validation>
原生支持矩阵 <../reference/support-matrix>
方法与技术报告 <methods>
独立基准报告 <benchmarks>
开发与贡献 <../contributing>
源码版本与包发布 <../reference/release>
```
