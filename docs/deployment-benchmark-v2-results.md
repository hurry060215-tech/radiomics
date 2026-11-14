# 独立 L5 部署成本证据：snapshot-native-v3

本报告只评价两个预声明启动端点、固定构建产物体积，以及一次离线 wheel payload 安装。它不测影像特征、模型或病理算法，不与三个计算套件合并总分，也不计算 CLI/Python 端点之间的“语言加速比”。

## 固定对象和方法

- 源码 `45140bff46b3ef1ae7eff9a170acb3a1116308b1`，核心实现 `a39c66ee16dcf04aefd79f2b3996e1be5666b2903b0f40e991aaf5097d54c2a8`，包版本 `0.1.0`。
- 不可变 wheel SHA256 `8767c22d3344d0009669de7c5297c0db58b5f9144c506be04722fe70ab4ea9b3`；CLI SHA256 `2197ea4af3cc3078f660a26f5a1401c8c9b185ace3d1d31984c7ba74fbb471f4`。
- 先审阅并批准 [deployment-plan.json](../benchmarks/v2/deployment-plan.json) 精确 SHA `d807983caaee9bb95f6046327c014bb9a2b86c6e421e77e96b66e938c0c6bfa6`，之后只执行一次固定试验。
- Windows、CPython3.9.21；新建 `system-site-packages=false` venv。创建环境为不计时准备，不修改既有 Golden/测量环境或系统安装。
- CLI `--version` 和 Python `-I` import＋`backend_info()` 各运行7个独立进程，无预热；round0 CLI→Python，round1反向，固定交替，全部14次保留。时间为父进程启动到 stdout/stderr 传输与退出完成；字节核验、日志解码、文件计数和扫描在计时外。
- 主 agent 和验收工作包确认短窗口没有重叠构建、安装、算法预检或其他套件计时。OS/user 后台进程和文件/bytecode cache 未独占控制。

## 结果

| 端点 | n | Median ms | IQR ms | 经验 p95 ms | Min–max ms | 输出核验 |
|---|---:|---:|---:|---:|---:|---|
| CLI `radiomics --version` | 7 | 3.611 | 0.567 | 48.817 | 3.132–67.938 | 7/7 exit0、版本正确 |
| Python fresh interpreter import＋`backend_info` | 7 | 45.052 | 3.447 | 51.142 | 44.125–52.270 | 7/7 rust backend、版本和实现摘要正确 |

CLI 首个67.938ms慢样本完整保留；没有重跑或剔除。该样本的 OS 调度/文件缓存等具体原因未被定位，不能仅从结果指定成因。七次经验 p95 对尾部波动很粗，不能作为精确尾延迟保证。“cold”仅表示新进程，不表示清空磁盘/OS/bytecode cache。两个端点做的工作不同，因此不生成两者耗时比或语言优劣结论。

## 产物和安装文件体积

| 对象 | 实测 |
|---|---:|
| wheel 压缩文件 | 3,723,631 bytes |
| wheel ZIP regular files | 31 |
| ZIP 未压缩文件总逻辑体积 | 9,880,697 bytes |
| 独立 CLI 文件 | 9,058,304 bytes |
| 已安装项目 package＋dist-info | 59 regular files；10,074,196 logical bytes |
| 新 venv 安装前 | 1,305 regular files；16,994,242 logical bytes |
| payload 安装后与启动观测后 | 1,364 regular files；27,068,438 logical bytes |

安装后25个 Python/native 源文件逐字节匹配固定 wheel。文件数包括安装器生成的 metadata/bytecode，和 ZIP entry 数不是同一口径。这里的磁盘指标为 regular-file **逻辑字节之和**，不是 NTFS cluster allocation、共享系统 DLL、base Python 或全部科学依赖的实际物理占用。

## 一次离线 payload 安装

新环境中只执行一次：

```text
python -I -m pip --disable-pip-version-check install --no-index --no-deps --no-cache-dir FIXED_WHEEL
```

记录耗时 **0.762398s，n=1**，日志完整保留，不报告 median/IQR 或重复选择最快值。该步骤只安装项目 wheel payload，没有解析/下载/安装 NumPy、Pandas、Matplotlib 或可选后端；venv/base Python 创建也不计入。惰性导入使 `import`/`backend_info` 能在这个受限环境中工作，但完整分析仍需要声明的基础依赖。此数字**不能称作完整安装成本**。

## 证据与复现

可核查 [独立报告](../benchmarks/results/deployment-native-45140bff/report.json)、[14条原始观测](../benchmarks/results/deployment-native-45140bff/raw-observations.jsonl)、[单次安装记录](../benchmarks/results/deployment-native-45140bff/payload-install.json) 与 [artifact index](../benchmarks/results/deployment-native-45140bff/artifact-index.json)。外部完整目录为 `E:/radiomics-build/benchmark-v2/deployment-v3-1`，包含 fresh venv 和原始日志；归档不包含可执行构建产物或整个环境。

```powershell
python tools/benchmark_deployment.py `
  --plan benchmarks/v2/deployment-plan.json `
  --approved-plan-sha256 d807983caaee9bb95f6046327c014bb9a2b86c6e421e77e96b66e938c0c6bfa6 `
  --build-manifest E:/radiomics-build/snapshot-native-v3/build-manifest.json `
  --base-python D:/OnekeyPlatfrom50/onekey_envs/python.exe `
  --output /fresh/deployment-evidence
```

重新执行应使用新目录，视为新的独立试验，不能替换本次14条。该受限 L5 证据不自动建立 Linux 启动成本、完整依赖安装成本、算法性能或所有部署门槛；版本、source/receipt、构建大小与工作量限定必须一同保留。
