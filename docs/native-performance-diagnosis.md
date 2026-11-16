# 原生数组边界性能诊断：固定 PAT1 CT

本次诊断定位了 Python 数组入口的实质开销：在同一次完整数组调用的嵌套时钟中，**Python JSON 编码、解码与 Python `deepcopy` 合计份额的中位数为52.59%**。它们既不是特征算法，也不是必要的 NIfTI I/O。当前 `ImageSample` 从已经验证过的 image/mask 再构造一次，就需要约141 ms。优先优化大缓冲区跨边界的表示和对象构造，比删除微小的特征表验证更有依据。

这不意味着全部开销都来自边界。相同预载 CT 在纯 Rust 中调用原 `extract_families`，仍需约104.7 ms，包含其原有验证、离散化、矩阵和标量计算。本诊断没有证明 Rust 必然快于成熟的 C 扩展，也没有单独测量 PyRadiomics 的等价纯 C 内核。

本页是独立诊断，**不替换、重录或重新解释原854条正式 Radiomics Benchmark v2 观测**。正式结果仍为 CT L2 原生394.163 ms、公开PyRadiomics3.0.1参考157.666 ms，速度比0.400×；L3原生183.807 ms、参考225.082 ms，速度比1.225×。L2/L3输入与处理边界不同，不得相减后声称得到了某个算法或I/O阶段。

## 输入、构建与执行范围

- 原始 CT/ROI：固定 IBSI PAT1，原始文件SHA分别为 `909129fd...d7bbcb`、`0e9243e4...94ae78`，许可CC-BY-NC-3.0。原始文件、大数组和约2 MB的请求JSON只保留在Git之外。
- 预载输入：与正式CT L2完全相同的tight bbox，shape `[100,99,26]`、125256个ROI体素，完整更新后的RAS affine；原始整数值HU的无损f64表示。
- 参数：`python_golden_v1`、fixed width25、distance1、direction_mean、GLDM alpha0、voxel shift0；没有clip、normalize、resample或resegment；全部55项统计/纹理＋11项shape_basic。
- 每个数值输出路径都用原冻结 `bbox_crop_reference.json` 的 `features` 和 `basic_shape` 两节检查66项，仍用原 `atol=1e-10,rtol=1e-8`。没有修改参考或容差。
- Python：不可变 `snapshot-native-v2` 对应环境，CPython3.9.21，扩展SHA `7a7109a732678ba1c498a2af249c4db867d7a906d58d2ccd458a89a461b29878`，构建来源 `48ecb310`。
- Rust：原核心和原gateway源码，Rust1.100.0/LLVM23.1.5、MSVC x86_64、release opt3/thin LTO/codegen-units1。依赖锁原始字节与v2精确一致；运行时implementation SHA严格为 `a39c66ee16dcf04aefd79f2b3996e1be5666b2903b0f40e991aaf5097d54c2a8`。
- 主机：Windows build22631、Intel Core Ultra9 285K、24逻辑CPU、33,663,320,064 RAM bytes。NumPy实际线程库限制为1，OMP/MKL/OpenBLAS/NumExpr/Rayon声明为1；没有改变或安装公开301参考环境。
- 每阶段1次预热、7次实际观测；每轮轮转阶段执行顺序。共98个Python观测、63个Rust观测、7条完整数组调用的实际request trace。预热也另存原始记录。
- 诊断运行前Root协调其他任务停止Sphinx/浏览器/编译；正式v3全部完成后发出all-clear。没有删去慢样本。

测量代码快照为 `719612f`。后续提交只添加证据、说明及独立执行器的错误输出保护，不改变本次已记录的观测。全部原始计时、嵌套trace、源文件及构建身份见[诊断数据](native-performance-diagnosis-data.json)。它不包含CT像素或大型请求payload。完整本地目录为 `D:/radiomics-native-boundary-diagnosis/baseline-v3/`。

## 真实调用链和重复工作

原正式L2的一个ROI依次执行：

1. `rad.image.from_array`：NumPy转Python list，`validate_image`发送整个image JSON，Rust验证后把整个image序列化回Python。
2. `rad.mask.from_array`：检查有限整数标签、转uint32 list，`validate_mask`发送整个mask JSON，Rust把整个mask返回。
3. `ImageSample(...)`：先对两个payload执行Python深复制，再通过 `validate_image_sample`重发和回传整个image/mask。
4. `sample.extract`：`array_features`再次发送同一大缓冲区，Rust反序列化、调用原特征引擎，返回66个值。
5. 小型 `validate_feature_table` 和DataFrame导出。

所以每个image/mask缓冲区在同一调用中各被JSON发送3次、回传2次。五个请求的UTF-8请求与响应合计 **10,201,863 bytes**；单独的 `array_features` 请求为2,035,442 bytes，66值响应仅3362 bytes。

`encode_json` 已有有限值快路径。本次有限CT没有以递归 `clean_json` 作为常规编码路径，不能把它虚构成主因。PyO3使用拥有的String并在native计算期间释放GIL；本诊断中的预编码调用仍包含字符串转换、Rust JSON反序列化、核心计算及响应编码，不能把它叫成“纯内核”。

## 同一次完整调用内的实际时钟

下表来自7次**嵌套跟踪**，毫秒值分别报告各项中位数。每行是实际操作，不能把各项中位数相加当作某一次精确总和。

| gateway操作 | Python编码ms | native调用ms | Python解码ms | 请求总ms |
|---|---:|---:|---:|---:|
| validate_image | 22.536 | 14.333 | 16.329 | 53.108 |
| validate_mask | 6.420 | 8.448 | 8.164 | 22.931 |
| validate_image_sample | 24.775 | 23.418 | 24.370 | 72.756 |
| array_features | 32.106 | 121.730 | 0.064 | 154.018 |
| validate_feature_table | 0.122 | 0.080 | 0.043 | 0.242 |

按每次调用自己的完整wall时间计算占比、再取七次占比的中位数：JSON编码22.43%、JSON解码12.70%、Python深复制17.23%，三者**合计占比的中位数52.59%**。合计份额使用同一次调用内的测量，未通过“两个不同路径耗时相减”产生。

全部native请求合计占比中位数43.69%，其中既有特征计算，也有大型请求的JSON解析、构造验证和大对象响应序列化，不能都归到特征矩阵。其余约3.8%包含NumPy/list转换、分派、对象/表格组织和跟踪开销。instrumented L2中位数385.036 ms，未跟踪的L2中位数381.585 ms；跟踪本身的扰动没有被视为零。

## 独立Python阶段诊断

下表的各行是**独立调用**，会重叠，禁止相加或相减来构造阶段成本。p95采用明确记录的线性插值经验分位数；n=7不足以给出稳定尾部性能保证。

| 阶段 | median ms | IQR ms | p95 ms |
|---|---:|---:|---:|
| 仅编码完整array请求 | 25.549 | 0.334 | 28.165 |
| 预编码请求调用PyO3 gateway | 121.709 | 0.935 | 122.942 |
| 解码66值响应 | 0.021 | 0.001 | 0.042 |
| 解码完整image响应 | 16.083 | 0.654 | 16.788 |
| NumPy image转list | 1.838 | 0.732 | 2.303 |
| NumPy mask转uint32/list | 1.047 | 0.029 | 1.115 |
| Python深复制image＋mask | 67.035 | 0.705 | 68.309 |
| 完整image.from_array | 55.850 | 1.630 | 58.209 |
| 完整mask.from_array | 26.789 | 0.512 | 28.149 |
| 已验证volume再构造ImageSample | 140.970 | 4.038 | 147.526 |
| 已构造Sample再extract | 146.729 | 1.759 | 150.560 |
| 完整未跟踪L2公共调用 | 381.585 | 4.745 | 391.147 |
| 完整跟踪L2公共调用 | 385.036 | 1.493 | 389.124 |
| 完整L3读取/裁剪/提取/保存 | 184.281 | 0.732 | 184.971 |

最后一行L3内部另有同一次调用的明确Python阶段时钟：读取cohort中位数0.483 ms、`rad.extract`182.161 ms、保存1.607 ms。`extract`内的内容摘要、NIfTI读取、准备、质量记录、计算和边界响应尚未在不可变库内部逐项打点，不能把它进一步凭空拆分。

## 纯Rust同输入probe

probe直接包含被审查的 `radiomics-python/src/gateway.rs`，并直接调用同一 `features::extract_families`。它先把完整NIfTI读出、用原ROI bbox裁剪，断言裁剪后的**values、labels、shape、完整affine**与Python预载请求精确一致。它不是另一个小矩阵或不同特征集合。

| 阶段 | median ms | IQR ms | p95 ms |
|---|---:|---:|---:|
| typed JSON反序列化 | 5.583 | 0.210 | 5.717 |
| image/mask Vec克隆 | 0.682 | 0.047 | 0.712 |
| image/mask JSON序列化 | 5.609 | 0.046 | 5.670 |
| 同一预载buffer核心提取 | 104.707 | 0.468 | 105.492 |
| 原gateway处理预编码请求 | 121.179 | 0.522 | 122.623 |
| 原始NIfTI image/mask读取 | 49.632 | 0.293 | 49.911 |
| 已读原始图像对裁剪 | 0.904 | 0.024 | 0.922 |
| 同一次读NIfTI→bbox裁剪→原核心 | 156.715 | 0.560 | 157.597 |

另保存了 `image_mask_validate_geometry` 微测，但不将其近零值解释为全部验证成本：它只覆盖公开geometry/length检查，没有私有 `features::validate_pair` 的ROI存在/有限值扫描；不变输入循环也可能被编译器优化。那三个完整扫描仍包含在104.7 ms核心调用内，尚未独立测出。

预编码PyO3调用121.709 ms与纯Rust gateway121.179 ms在此工作量上接近。这支持优先减少重复的Python对象/JSON往返；它不等价于严格测出了PyO3开销为两者差值。纯Rust的临时结果销毁在其时钟内，Python外层时钟不含最后一个临时返回对象的销毁，也是两类微测不能直接作精确差值的原因。

## 有限修复方案与需要保留的语义

第一优先级是一个**内部数值缓冲区入口**：以 `Vec<f64>`/`Vec<u32>`传入图像和mask，小型geometry/config/family继续使用既有类型化定义，仍调用原 `extract_families`。可以保留abi3-py39而不引入NumPy Rust crate；这仍会分配/复制，不能称作zero-copy。

只替换 `array_features` 的大JSON请求，尚不能解决前面三次构造验证往返和约66 ms的Python深复制。因此候选应分开验收：先验证单次特征调用，再检查从 `from_array→mask→ImageSample→extract→DataFrame` 的完整原L2入口。内部可以使用拥有的连续buffer、小型验证返回或只读状态复用，避免验证后再把全部体素编码回Python。不要仅隐藏到未计时的预处理阶段来展示加速。

必须保留未知单位、完整affine、mask整数/范围、空ROI、非有限强度状态、独立形态/强度mask、family选择、定义/特征顺序和错误语义。Python对象的所有权和别名行为不能因省深复制而改变。候选应通过原固定2D/3D矩阵/几何/计数/Golden和此CT66；没有理由降低容差或删掉输入校验。微小的FeatureTable验证不是本次优先优化对象。

第二优先级才是对剩余约105 ms核心进行真实内部profiling，区分离散化、GLCM/GLRLM/GLSZM/GLDM/NGTDM矩阵和标量，再决定数据结构与重复扫描优化。当前没有CPU采样、allocator计数或精确矩阵阶段数据，不能断言是哪一个纹理内核主导，也不应立即大规模重写。

## 小规模实际内存诊断

时间诊断后另作一次固定内存诊断，没有重测或替换正式性能套件：三条路径各3个fresh process，共9条观测，约4.6秒一次完成。代码快照 `bc550cd`，全部66特征仍与原冻结参考匹配。完整原始绝对RSS、阶段快照、实际PID与构建身份见[内存诊断数据](native-memory-diagnosis-data.json)，外置目录为 `D:/radiomics-native-boundary-diagnosis/memory-v1/`。

Python worker直接读冻结cropped `.npy`，Rust worker直接读同一cropped typed JSON；协调者事先逐元素核对两种载体的values/labels、完整affine和bin配置。**所有worker都未先读完整NIfTI源卷**，避免全图解码setup峰值覆盖L2阶段。每个worker在自己的进程内调用Windows `GetProcessMemoryInfo`，报告当前WorkingSetSize和PeakWorkingSetSize，最后在写完输出后补footer。

六个Python worker的venv launcher PID均与实际worker PID不同；工具分别记录两者。实际RSS始终来自worker自身API，其阶段PID和footer PID一致，未把Popen返回的launcher当成算法worker。三个Rust worker的launcher/worker PID相同。

| 路径 | 3次绝对峰值MiB（全部保留） | 峰值中位数MiB | 预载后当前RSS中位数MiB | 预载结束时已达HWM中位数MiB | 随后新HWM增量中位数MiB |
|---|---|---:|---:|---:|---:|
| immutable完整L2公共数组调用 | 103.5625 / 103.2227 / 103.7070 | 103.5625 | 55.5820 | 55.5820 | 47.9805 |
| 同Python启动环境的预编码gateway低层调用 | 77.6484 / 78.5195 / 78.9062 | 78.5195 | 57.6914 | 59.6328 | 18.8984 |
| 独立Rust typed buffer预载后核心 | 11.8047 / 11.9023 / 11.7930 | 11.8047 | 6.5586 | 8.9922 | 2.8125 |

第一、二行都有相同Python/NumPy/pandas/原生包导入和cropped数组启动成本，导入后的当前RSS中位数分别53.121、53.195 MiB；第二行另外保留约2 MB请求文本，因此它的预载setup稍大。第二行仍调用同一Rust特征核心，省略完整公共volume/sample/FeatureTable构造，**不是Python参考算法**，也不是已经交付的等价快速公共API。

完整L2的实际阶段当前RSS中位数为：预载55.582 MiB、image构造66.828、mask构造69.516、sample构造81.582、最后对象保留81.723 MiB；累计峰值分别升至79.840、79.840、95.969、103.562 MiB。它说明当前常驻对象与临时往返都有可观察开销。四个各257400元素的独立Python list，仅浅容器大小就共9,249,888 bytes（8.821 MiB），未包含盒装标量、临时JSON、allocator和核心缓冲区；这是实际保留下界，未用逐元素相加重复计算共享小整数。

第三行的独立Rust启动当前RSS约3.355 MiB，typed JSON解析时当前RSS8.500 MiB/HWM8.992 MiB，释放输入文本后的预载RSS6.559 MiB，核心返回后当前RSS8.125 MiB/HWM11.805 MiB。输入的Rust数值buffer为f64/u32，而Python `.npy`为f64/u8，载体大小也已单列记录。不能把不同启动环境的11.8/103.6之比宣传成Python-facing公共API节省了相同倍数。

这些数据支持“重复对象/表示有具体内存成本、纯Rust独立核心可以在较小进程中运行”的有限结论。**它们没有建立Rust相对PyRadiomics同公共入口的内存优势**：本轮未运行后者内存对照。对用户选择高层后端，仍需保持结果一致并取得等价接口的速度或内存收益证据。

PeakWorkingSetSize是截至末尾footer的生命周期HWM，不是可重置的特征阶段峰值；预载解析setup仍计入其中。正的新HWM只证明后续阶段超过此前峰值；`peak-loaded RSS`和`peak-prior HWM`都不是精确分配量。未作Linux实际运行、allocator测量、OS cache purge、预算强制或大数据最大内存声明。Root协调短quiet窗口；另worker报告边界处有上一条0.2257秒stdlib标量检查，无原生worker/大扫描；本节不作延迟结论，九条观测全部保留。

复现使用新目录，不会改动原854条结果：

```powershell
python tools/profile_native_memory.py `
  --fixtures E:/radiomics-build/benchmark-v2/radiomics-inputs-a37-ct-v1/public_ct_pat1 `
  --request D:/radiomics-native-boundary-diagnosis/baseline-v3/array-request.json `
  --build-manifest E:/radiomics-build/snapshot-native-v2/build-manifest.json `
  --rust-probe target/release/examples/profile_native_boundary.exe `
  --native-python E:/radiomics-build/ml-path-native-v2/Scripts/python.exe `
  --output D:/fresh-memory-diagnostic-directory --repeats 3 --acknowledge-license
```

## 是否仍有必要保留Rust

本基线诊断之后，已实现并单独检查了[Sample构造临时深复制优化](sample-constructor-optimization.md)。
同二进制对照中完整公共L2中位耗时下降约17.38%，实际进程峰值内存没有下降。
这只关闭了临时复制的一小部分速度问题；下文的更大缓冲区边界方案、正式性能目标、
相对Python的内存收益和已知标量数值反例仍不能据此宣布解决。

按“仅在速度、峰值内存或精度有明确收益时采用Rust，优先结果一致”的选择标准，本轮还不能证明应该把所有高层流程都换成Rust。独立CLI、统一实现和保存加载是工程能力，不能单独替代用户要求的数值或资源收益。当前数组入口的重复数据表示是具体可修复问题，也不等价于证明修复后的Rust一定胜出。

不能用“Rust”名称代替速度证据。PyRadiomics关键纹理由成熟C扩展执行；要声称语言或内核优势，还需要同输入、同计算边界、同资源的对应内核测量。104.7 ms纯Rust调用不能直接与157.7 ms的Python公共API相比后宣称内核加速。本轮定位了本包内部时间/内存开销，但未建立相对PyRadiomics等价公共API的内存或更高精度收益，也未将原1.5×目标宣布通过。可以逐模块显式混用已经验证的高层后端，维持严格auto规则和特征定义/列顺序/数值一致；无需为了语言统一把更合适的路径强行替换。

候选修复若有收益，应先在新的不可变构建上保留所有原病例/原次数重新验证，再更新正式性能报告；本次854条观测始终保留。没有对应证据的阶段继续标为未证明，不宣传语言速度、内存或精度的普遍优势。

## 尝试、失败与字节身份

`baseline-v1`在计时前严格拒绝了诊断适配器的55/66列不一致：正确做法是合并冻结参考已有的55项 `features` 和11项 `basic_shape`，没有修改预期。

`baseline-v2`保存了全部阶段原始计时，但最终核心身份检查失败，且后来获知可能与文档/浏览器活动重叠，因此**不作主结果**。原因是新worktree把Cargo.lock检出为CRLF：40,291 bytes、SHA `ff310d53...c77d`；严格CRLF→LF得到Git blob的38,771 bytes、SHA `de33d6df...552bb`，与原v2锁文件逐字节相同，依赖版本及核心源码均未变化。build.rs对Rust源码规范化换行，但对Cargo.lock直接纳入摘要，因此换行即可令implementation ID变成 `dddda1cc...452c`。

仅将独立worktree的锁文件工作区恢复为原LF字节并重建后，probe严格恢复为原 `a39c66ee...c2a8`。锁文件没有索引/依赖内容变更，没有改Root，更没有通过接受错误ID让测量通过。新runner在开始计时前就检查原锁字节。完整修复前后摘要、轮次和失败目录均保存在诊断数据中。

## 复现

需要完整仓库、已冻结的外置prepared CT目录、固定v2 wheel环境和匹配工具链。先检查Cargo.lock工作区原始SHA是否与构建manifest相同；若只因CRLF不同，恢复原Git blob LF，确认Git diff仍为空。不要改依赖版本。

```powershell
cargo build --release --locked -p ov-radiomics-core --example profile_native_boundary
E:/radiomics-build/ml-path-native-v2/Scripts/python.exe tools/profile_native_boundary.py `
  --fixtures E:/radiomics-build/benchmark-v2/radiomics-inputs-a37-ct-v1/public_ct_pat1 `
  --build-manifest E:/radiomics-build/snapshot-native-v2/build-manifest.json `
  --rust-probe target/release/examples/profile_native_boundary.exe `
  --output D:/fresh-external-diagnostic-directory --acknowledge-license
```

若设置CARGO_TARGET_DIR，probe路径需要对应调整。输出必须是新目录；脚本不修改参考环境、Golden、正式raw或既有诊断结果。所有原始观测和失败保留；这仍是暖OS缓存、单线程、单个公开CT的诊断，未建立其他数据、机器、冷启动、并行扩展或最大内存的性能结论。
