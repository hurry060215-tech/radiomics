# 小型标量精度审计：先比较准确性，再决定实现归属

本审计不修改算法、Golden、默认定义、数值容差或依赖。它只检验同一实际float64输入、同一定义下，现有Rust与公开PyRadiomics3.0.1哪个更接近独立精确参考。**相互更接近不等于更准确，使用Rust也不自动更准确。** 速度、峰值内存、数学精度、可部署性需要分别量化，不能由语言名称推定。

## 首轮冻结范围

只审Mean、总体Variance与Energy三项，不把结果扩展到全部影像特征、模型或临床效度。使用4个固定2×4小矩阵：普通值、正负大数取消、1e12偏置叠加实际可表示的微小变化、常量7。所有体素选入label1；无预处理、插值、滤波或ROI变化，Energy shift=0。两条路径显式binCount8，避免大正负值的固定bin width产生巨大灰度范围。

原生路径是公开 `ImageSample.extract(..., families=['firstorder'])`，配置 `python_golden_v1`；它计算支持的一阶族，本审计只读取三项。Python路径是公开 `radiomics.firstorder.RadiomicsFirstOrder`，只启用三项，公开3.0.1固定环境，不调用Onekey。直接一阶class不执行高层自动裁剪/重采样；两个后端读取相同ROI和原始数值。

每个输入以 `float.hex()` 保存，worker在调用之前再次逐位核对。参考用 `Fraction.from_float` 精确还原已经输入的二进制浮点数：

- Mean：`sum(x)/n`。
- 总体Variance：`sum((x-exact_mean)^2)/n`，分母n，不是样本方差n−1。
- Energy：`sum(x*x)`，强度偏移为0。

分数的整数分子/分母是精确参考；Decimal80只作可读展示。不把打印的0.1当作理论精确1/10，不以显示小数位数或格式化round判断精度。

manifest已经在任何Rust/PyRadiomics计算之前由工具的 `freeze` 命令写入：`D:/radiomics-numeric-accuracy-audit/manifest.json`，SHA256 **`a7ae74c4ab250f2532042041e96f15477baa0171e3202a4fc10d5ceb6bc9581a`**。它包含输入f64 hex、精确参考、正确舍入的f64参考、公式、配置和工具源码SHA。源码与本协议先提交，再运行两个worker，随后仅追加结果报告，不重录参考。

## 指标及边界

输出绝对误差、相对误差（精确参考为0时明确undefined）、与正确舍入参考之间的整数ULP距离，以及绝对误差除以参考处一个ULP的比值。这两个ULP量在跨指数范围时不相同；ULP距离0也不意味着结果与精确有理数完全相等。

拒绝/未定义/非有限状态单列，不能跳过后只比较成功项目。每项比较使用精确误差，不把Python当数学真值，不用更接近Python的结果冒充更精确。常量方差0允许两者都精确；小误差差异也不自动构成医学或工程上的实质收益。

当前兼容profile必须继续遵守被冻结的Python行为和容差。若未来发现可量化的稳定求和/矩算法优势，是否增加另一个显式stable profile需单独定义、冻结及评估性能/内存；本次不实现高精度后端，不默改兼容profile，也不因发现误差而临时放宽门槛。

## 复查命令

```powershell
python tools/audit_scalar_accuracy.py freeze --output D:/new-scalar-audit
python tools/audit_scalar_accuracy.py run --manifest D:/new-scalar-audit/manifest.json --manifest-sha256 <freeze打印的摘要> --native-python <安装原生包的python> --python301 <已固定公开PyRadiomics3.0.1环境的python>
```

输出目录和JSON使用独占创建；不覆盖已有参考/worker/report。每个worker记录解释器、NumPy、实际原生扩展或PyRadiomics源码SHA与版本。脚本只启动已有环境，不安装或修改它们。实际比较结果在冻结之后单独补充。
