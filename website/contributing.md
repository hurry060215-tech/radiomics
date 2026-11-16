# 开发与贡献

从[贡献指南](https://github.com/hurry060215-tech/radiomics/blob/main/CONTRIBUTING.md)开始。代码、用户教程、算法定义与冻结参考均需要可复现证据。

本项目参考[OmicVerse主仓库](https://github.com/omicverse/omicverse)与[教程仓库](https://github.com/omicverse/omicverse-tutorials)的用户入口组织：安装、按任务选择教程、逐单元Notebook、独立数据来源页、API参考和开发指南。这里的实现、示例和验证属于独立项目，不继承ov的论文或性能结论。

## 本地预览

在完整源码仓库执行：

```bash
python -m pip install -r website/requirements.txt
python tools/build_website_sources.py
python -m sphinx -W --keep-going -b html website website/_build/html
python -m http.server 8000 --directory website/_build/html
```

打开 `http://localhost:8000`。Sphinx默认使用已保存Notebook输出，不在构建时下载真实数据或重新训练。修改教程代码后应独立运行执行器，保存收据，再构建文档。

PR只构建HTML预览；main推送自动发布[GitHub Pages](https://hurry060215-tech.github.io/radiomics/)。
维护者也可手动触发`docs.yml`发布允许分支的审阅版本，部署来源由Actions记录。
`github-pages`环境限制可发布分支，其他分支需先经明确配置；手动发布文档不合并PR、不发布软件包。
