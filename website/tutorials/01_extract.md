# NIfTI 与 ROI 基础提取

本章使用仓库中的固定合成数据。请在仓库根目录、与此源码版本对应的安装环境中运行。

## 分析约定

检查 affine、单位和 ROI；分箱参数必须显式指定。

## 运行

```bash
python examples/native/01_extract.py --output artifacts/01_extract
```

脚本拒绝覆盖已存在的输出目录。阅读执行结果和生成文件，确认 ID、队列及预测层级后再替换自己的数据。

## 完整可执行示例

```{literalinclude} ../../examples/native/01_extract.py
:language: python
```

{download}`下载脚本 <../../examples/native/01_extract.py>`

## 验证状态

本页已接入版本化示例源码；网站预览的构建通过不代表本章已在本次文档任务中重新执行。独立运行记录与结果图将在教程验收时补齐。

{download}`下载 Notebook <01_extract.ipynb>`
