# 网站导航调整：来源、设计与验收

<!-- PR-2: replace recorded machine paths with reproducible work-directory placeholders. -->

本次只调整网站信息架构与展示，不修改算法、Golden、八篇已执行 Notebook 或执行收据。

## 上游实读范围

2026-11-15 通过 GitHub 主源读取 `omicverse/omicverse-tutorials`，`git ls-remote HEAD` 核实 revision 为 `a34982d466431112f23db4dfcf79ae10cbb414a5`。以下链接均固定到该 revision：

- [docs/index.md](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/index.md)：首页简介与安装、教程、API 等入口。
- [docs/Tutorial.md](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/Tutorial.md)：按领域组织的教程总览。
- [docs/conf.py](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/conf.py)：主题、导航和 Notebook execution off 配置。
- [docs/tutorials/index.md](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/tutorials/index.md)：模块入口层。
- [bulk 模块](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/Tutorials-bulk/index.md)、[single 模块](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/Tutorials-single/index.md)、[spatial 模块](https://github.com/omicverse/omicverse-tutorials/blob/a34982d466431112f23db4dfcf79ae10cbb414a5/docs/Tutorials-space/index.md)：按上游、预处理、下游与推荐工作流展开。

借鉴的是“少量入口 → 任务模块 → 按步骤执行”的组织方式。没有复制其文案、图像、品牌、配色或性能结论。本项目继续使用现有 PyData 主题，不引入新构建依赖；上游存在的 warning 降级或宽松检查没有引入本项目。

## 本站结构与选择原因

顶部导航固定为五项：**安装、教程、用户指南、API、验证与开发**。快速开始收进安装，数据与许可收进真实数据路线，避免顶层长入口进入 More。

```text
安装
  快速开始
教程
  影像组学：01 → 04
  建模与统计：02 → 03
  轻量病理：05 → 06
  真实数据：07 / 08；数据与许可
用户指南
  输入契约 → 文件工作流 → 训练与外部预测
  病理与 embedding；结果排错；真实表格；实际验证范围
API
  入口与数据；影像与特征；建模与统计；病理与互操作；CLI
验证与开发
  验证与适用范围；支持矩阵；方法与技术报告；独立基准；贡献
```

教程按输入组织，每条路线分别列出输入、操作检查和输出，并指向自有文件工作流。01–06 在路线页保留“合成入门”标识；07/08 标为真实公开数据，分别说明 NC 许可/无结局及 source ID 不证明患者独立。每篇 Notebook 只属于一个路线 toctree，使侧栏、面包屑与前后页顺序一致；八篇下载入口继续保留。

用户指南使用独立 toctree，指向现有维护文档的生成页，不再将多个长指南塞入顶栏。API 页按操作对象查参数；算法验收、技术报告与三套独立 benchmark 移入验证与开发，避免新用户在安装或数据准备时穿过研究报告。

样式仅新增作用域明确的输入路线卡片、真实/合成标识、响应式两列布局和键盘焦点。移动端沿用主题折叠菜单，窄屏卡片改为单列。

## 构建与页面检查

在独立 worktree 执行以下命令，未修改构建工具：

<!-- PR-2: restore the source-generation and Sphinx commands with an explicit documentation interpreter. -->

首次构建生成 33 份维护文档引用页，57 个 Sphinx 页面全部通过，退出码 0，无 warning。Notebook execution 保持 off，未重新执行、训练或下载数据。增补 API 索引与样式后再次用 `-a -W --keep-going` 完整写出页面，最终退出码 0，无 warning；日志保留在 `E:/radiomics-build/website-navigation-review/sphinx-final.log`。

实际 Chrome headless 截图保留在 `E:/radiomics-build/website-navigation-review/`：

| 文件 | 窗口尺寸 | 目视检查 |
|---|---|---|
| `home-desktop.png` | 1440×1100 | 五项顶栏全部可见，无 More；四张输入卡片 |
| `tutorials-desktop.png` | 1440×1100 | 四条路线、合成/真实标识、八篇目录与分组侧栏 |
| `guide-desktop.png` | 1440×1100 | 独立用户指南侧栏、操作顺序和输入/产物对照 |
| `real-data-desktop.png` | 1440×1100 | CT 与表格分开说明；许可、身份与结局边界可见 |
| `home-narrow.png` | 520×1100 | 折叠菜单、卡片单列、正文与按钮换行，无横向裁切 |

窄屏检查采用 520 像素窗口；本机 headless Chrome 的 420 像素外窗仍报告 500 像素内部 viewport，早期 `home-mobile.png` 截图因此出现取景裁切，不作为 420 像素响应式通过证据。520 像素结果没有该问题；这里不宣称完成真实手机浏览器验收。截图在主 agent 确认性能测量独占窗口结束后补齐，没有重跑任何算法 benchmark。

额外使用标准库 HTMLParser 检查构建后全部页面的 1,801 个站内链接，缺失文件目标为 0。`git diff --check` 通过，`git diff --exit-code 9315841 -- 'website/tutorials/*.ipynb' 'website/*execution*.json'` 退出 0。

生成的 `website/reference/`、HTML、截图和浏览器 profile 不纳入代码提交。原始 Notebook 与执行收据保持原字节；本次不添加运行时或文档依赖。
