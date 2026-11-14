# 影像与临床二分类

本章使用仓库中的固定合成数据。请在仓库根目录、与此源码版本对应的安装环境中运行。

## 分析约定

所有拟合步骤只使用训练患者；外部队列保留独立身份。

## 运行

```bash
python examples/native/02_image_clinical_binary.py --output artifacts/02_image_clinical_binary
```

脚本拒绝覆盖已存在的输出目录。阅读执行结果和生成文件，确认 ID、队列及预测层级后再替换自己的数据。

## 完整可执行示例

```{literalinclude} ../../examples/native/02_image_clinical_binary.py
:language: python
```

{download}`下载脚本 <../../examples/native/02_image_clinical_binary.py>`

## 验证状态

本页已接入版本化示例源码；网站预览的构建通过不代表本章已在本次文档任务中重新执行。独立运行记录与结果图将在教程验收时补齐。

{download}`下载 Notebook <02_image_clinical_binary.ipynb>`
