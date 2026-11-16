# 小型标量精度审计：先比较准确性，再决定实现归属

本审计不修改算法、Golden、默认定义、数值容差或依赖。它只检验同一实际float64输入、同一定义下，现有Rust与公开PyRadiomics3.0.1哪个更接近独立精确参考。**相互更接近不等于更准确，使用Rust也不自动更准确。** 速度、峰值内存、数学精度、可部署性需要分别量化，不能由语言名称推定。

## 首轮冻结范围

只审Mean、总体Variance与Energy三项，不把结果扩展到全部影像特征、模型或临床效度。使用4个固定2×4小矩阵：普通值、正负大数取消、1e12偏置叠加实际可表示的微小变化、常量7。所有体素选入label1；无预处理、插值、滤波或ROI变化，Energy shift=0。两条路径显式binCount8，避免大正负值的固定bin width产生巨大灰度范围。

原生路径是公开 `ImageSample.extract(..., families=['firstorder'])`，配置 `python_golden_v1`；它计算支持的一阶族，本审计只读取三项。Python路径是公开 `radiomics.firstorder.RadiomicsFirstOrder`，只启用三项，公开3.0.1固定环境，不调用外部封装库。直接一阶class不执行高层自动裁剪/重采样；两个后端读取相同ROI和原始数值。

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

## 实测：当前Rust不能被笼统称为“更准”

首次执行在协议/源码提交 **`5c34371e92549d8c0ef6c92095ef93605fc7e588`** 之后，两个后端串行运行约1.95秒；这是验收命令耗时，不是性能benchmark。四个case双方均返回ok，没有拒绝或静默跳过。12个case×feature比较中，Rust绝对误差较小2项、Python较小1项、9项相同；这种有目的构造的小集合不能计算“语言精度胜率”或推广到临床数据。

下表误差为紧凑显示而舍入，排名依据JSON中精确分数；ULP为距正确舍入f64参考的整数距离。

| 用例 | 特征 | Rust绝对误差 | PyRadiomics3.0.1绝对误差 | Rust/Python ULP距离 |
|---|---|---:|---:|---:|
| 普通 | Mean | 2.428613e−17 | 2.428613e−17 | 0 / 0 |
| 普通 | Variance | 3.034899e−17 | 3.034899e−17 | 0 / 0 |
| 普通 | Energy | 6.938894e−16 | 6.938894e−16 | 0 / 0 |
| 正负取消 | Mean | 0.375 | 0.5 | 9007199254740992 / 4602678819172646912 |
| 正负取消 | Variance | 2.683081e15 | 6.324118e15 | 0 / 1 |
| 正负取消 | Energy | 2.146465e16 | 2.146465e16 | 0 / 0 |
| 大offset微小变化 | Mean | 4.577637e−5 | 4.577637e−5 | 0 / 0 |
| 大offset微小变化 | Variance | 2.095476e−9 | 2.095476e−9，略小1ULP | 158329674399744 / 158329674399743 |
| 大offset微小变化 | Energy | 7.718449e8 | 7.718449e8 | 1 / 1 |
| 常量7 | Mean | 0 | 0 | 0 / 0 |
| 常量7 | Variance | 0 | 0 | 0 / 0 |
| 常量7 | Energy | 0 | 0 | 0 / 0 |

最值得关注的不是“Rust赢了两项”：

- 取消用例的精确Mean为 **0.5**，Rust为 **0.125**，Python为 **0**；相对误差分别75%与100%，两者都严重失真。Rust虽然更近，也不是令人满意的准确结果。
- 大offset方差的精确值为 **9.289942681789398193359375e−8**。Rust为9.499490261077881e−8，Python为9.499490261077880e−8；两者相对误差都约 **2.255639%**。Python小1ULP不足以消除共同的偏差。
- 普通三项双方均为正确舍入的f64结果，但相对精确有理数仍有非零误差；“0 ULP距离”不是无限精度。
- 大offset均值双方已经正确舍入；精确均值位于相邻f64之间，仅输出一个f64无法把它变成完全精确。绝对误差大小必须结合数值尺度解释。

这与浮点求和/中心化顺序敏感相符，但本次没有做编译器、向量化或每条算术指令的因果消融。Python参考的Variance可见实现是 `numpy.nanstd(...)**2`，Rust一阶用中心矩；观察到的一ULP差异不能推广为某条路径普遍更好。

### 更新：取消输入的兼容性缺口已修复，数学差异仍单列

本次只将一阶 Mean 的朴素累加改为固定二叉 pairwise 求和；没有修改
输入、特征名、配置、Golden、atol=1e-10 或 rtol=1e-8。对上面的同一
取消输入，当前兼容 Mean 为 **0.0**，与冻结的公开 3.0.1 值 **0.0**
相符。原严格 xfail 已移除，原有兼容断言成为正常通过项。

独立数学对照仍对原输入直接计算 `math.fsum(values) / len(values)`，
得到 **0.5**。因此本次关闭的是该反例的兼容性缺口；兼容值与数学值
之间仍有 **0.5** 的绝对误差。pairwise 不等于精确求和，本例也不支持
声称数学精度问题已经修复或默认 profile 在所有输入上等同公开参考。
两个结果在回归输出中分别报告，不能把兼容值 0.0 解释为数学真值。

上述历史审计的原生值 0.125 与公开参考值 0.0 仍保留为当时的真实
观测。本次修复晚于 Benchmark v2 的 **854 条正式观测**，不重跑、不
重录或替换它们；历史 L2/L3 速度仍只描述其已记录的构建版本，本次
未建立新的性能结论。既有 Golden 和标量容差均保持不变。

## 补充：稳定算法原型，不是新后端

经主agent明确追加任务，新增 `tools/audit_scalar_stable_prototype.py`，仍使用**同一份已冻结输入和有理数参考**。原型源码在任何原型计算前提交为 **`9ba93029ca3dc973665c2dc3508f0f583ac91486`**，补充manifest SHA256 **`a8cdb0aeb98305f8591220669f3e2df677ae2c2c39a0c79db7cb840e19215598`**。

原型只演示三种标准库f64计算策略：Mean用 `math.fsum(x)/n`；Variance先取首个输入为anchor，计算f64的 `x-anchor`，再fsum得到小量中心及平方残差，仍除以n；Energy用 `math.fsum(x*x)`。平移、减法、残差与乘法仍为f64，**Energy乘法舍入未消除**，也不保证所有动态范围上的稳定性或精确性。

| 同一冻结用例 | 原型实际结果 | 与精确参考关系 |
|---|---|---|
| 取消Mean | 0.5 | 完全精确，修正现有两后端的大误差 |
| 大offset Variance | 9.289942681789398e−8 | 完全精确，消除本例约2.255639%的偏差 |
| 大offset Mean | 1000000000000.0005 | 与原路径同样正确舍入，仍有不可避免的单f64表示误差 |
| 大offset Energy | 8.000000000000007e24 | 本例达正确舍入，绝对误差仍约3.018969e8，非精确平方和 |
| 取消Energy | 4e32 | 与原路径相同，绝对误差仍约2.146465e16；fsum没有修复逐项平方舍入 |
| 普通三项、常量三项 | 与原路径相同 | 普通正确舍入但非精确；常量三项完全精确 |

原型表明**同样的Python语言也能通过算法选择提高这两个用例的精度**；不能把改进归功于Rust。要在Rust保留哪些部分，应分别依据真实吞吐、峰值内存、数值误差或部署价值的证据。此原型未计时、未测内存，没有被接入核心、绑定或CLI，也没有新增 `stable` profile。语言、算法、定义、工作量和验收目标分别决策。

## 原始结果、版本与不变性

原报告：`D:/radiomics-numeric-accuracy-audit/report.json`，SHA256 **`79e90b6cfe086573f7c325d20a6e5e532da0ad6e341ca9d946840031dcbce17c`**。补充执行前后该SHA逐字节一致；原输入manifest也保持不变。

补充报告：`D:/radiomics-numeric-accuracy-audit/stable-prototype/report.json`，SHA256 **`a8c399aefb7f94fdbf961507360562463b34c475583c40b539933b8360379823`**。两份报告分开，不用原型结果替换原生或Python实测。

实际解释器均为Python3.9.21。原生安装环境NumPy2.0.2，仅承担数组转换；Rust计算的扩展SHA256为 `bda55de68f267fc3bcd710732f621659792741658b5900639c78141c5714cd7c`。公开参考环境为原有 `E:/radiomics-build/golden/env-301`，PyRadiomics **v3.0.1**、NumPy **1.23.0**、SimpleITK **2.4.1**；firstorder源码SHA256为 `80d01b13750c4180f4ae2f8dbefab0ae95856ab6836ccd97fa13dfa5cbee54d9`。没有安装或修改参考依赖。

主审计工具实际源码SHA256为 `66aaaa6d7875cb6a1b1461b6e9839fff9942325b4217498a8a7dacbc117febe3`；每个worker JSON有其输出f64 hex、输入位核对、身份与源码摘要。审计范围没有延伸到RMS、Skewness、Kurtosis、纹理、模型、统计推断或实际医学队列；不据此泛化“Rust一定更准”或“所有Python实现都有同样问题”。
