# 真实文件输入示例

这些脚本读取用户已有文件或明确许可的公开文件，不生成临床结局、训练/测试标签、患者身份、WSI 坐标或物理尺度。计算调用同一 Rust core；NumPy/Matplotlib 仅用于转换和显示。请先安装当前编译 wheel 及基础 Python 依赖，使用新输出目录。完整分析教程见 [`docs/user-guide`](../../docs/user-guide/)。

## 1. 自有 NIfTI＋ROI → 特征和真实 ROI 预览

cohort CSV 必需列为 `sample_id,patient_id,image,mask`，路径相对 cohort 文件；可选 `roi_label,modality,region,lesion_id,timepoint,center`。同一患者可以有多个不同 sample ID。ROI 标签列若提供会覆盖全局 `--labels`。

```csv
sample_id,patient_id,image,mask,roi_label,modality,region
lesion-A,真实患者键,images/case-A.nii.gz,masks/case-A.nii.gz,1,CT,tumor
```

```bash
python examples/real/nifti_extract.py --cohort /your/cohort.csv --bin-width 25 --labels 1 --features basic --output /fresh/extraction
# 需要完整明确子集时显式选择full；bin设置由研究方案决定，不能照抄成通用CT/MR/PET默认。
python examples/real/nifti_extract.py --cohort /your/cohort.csv --bin-count 32 --labels 1 --features full --output /fresh/full-extraction
```

`basic` 为 Original first-order16＋`shape_basic`11；`full` 为 Original明确55＋`shape_basic`11。固定 width/count 二选一且必须显式给出。默认无重采样、截断、强度归一化或配准修复；`--preview-window -500 400` 只改变显示，不改变提取输入。`--n-jobs` 控制外层并行，`--cache` 为显式内容缓存。

输出：

- `features.csv`：Rust canonical FeatureTable，第一列 `sample_id`，其他列为数值；`features.csv.manifest.json` 保留顺序、定义、单位、样本和参数。
- `metadata.csv`、`features-metadata.csv`：真实提供的 sample/patient/模态/区域等身份，不混入特征矩阵。
- `diagnostics.csv`、`failures.csv`：失败不丢弃；有失败 exit2，空结果没有伪造零向量。
- `extraction/`：完整公开 API 结果；`provenance.json`：输入内容摘要、有效参数和实现摘要。
- `preview/sample-*.png`：成功输入中实际 ROI 最大轴向截面的图像/轮廓；展示的是 voxel index，不自动冒称已按 RAS 重定向的临床阅片视图，完整 affine/单位保留在 provenance。

只有实际临床/结局表可进入建模。单个公开患者或无结局文件不能通过随意填写0/1变成训练数据。可把 `features.csv`、其 manifest 和真实 clinical/outcome 表交给 [`train_table.py`](train_table.py)，按显式 `StudyTableSpec`/患者分组方案处理。

## 2. 公开 PAT1 数据：外部缓存、固定 SHA 和许可

```bash
python examples/real/download_public_ct.py --output /external/public-pat1 --acknowledge-license
# 已下载本机cache时可先复用，通过固定SHA后复制，不重复网络下载。
python examples/real/download_public_ct.py --output /external/pat1-copy --from-cache /existing/source --acknowledge-license
python examples/real/nifti_extract.py --cohort /external/public-pat1/cohort.csv --bin-width 25 --labels 1 --preview-window -500 400 --output /fresh/pat1-features
```

下载器只依赖随脚本分发的 [`public_ct_source.json`](public_ct_source.json)：固定公开 URL、文件大小/SHA、来源和署名事实，不依赖安装包中被排除的 numerical Golden。它输出原始 `image.nii.gz`、`mask.nii.gz`、`SOURCE_LICENSE.md`、`SOURCE_README.md`、`NOTICE.md`、`cohort.csv` 和 `source-manifest.json`。`PAT1` 是来源数据集的标识，不是新增临床结局；cohort 无训练 label/group。

数据为 **CC-BY-NC-3.0**，原始贡献 Philippe Lambin/CancerData，NIfTI转换 Alex Zwanenburg，DOI `10.17195/candat.2016.08.1`。软件 MIT 不覆盖该数据许可。原图、准备数组和派生医学预览保留相应来源/许可；原始缓存拒绝写入任意 Git worktree。下载/复用前后检查固定 bytes和SHA，不覆盖已存在的不匹配文件。必须先查看许可并传入 `--acknowledge-license`，否则在创建目录/网络访问前退出；无交互询问，该标志也不授予额外使用权。

## 3. 已有普通 patch 文件 → QC/颜色 → 聚合 → 真 tile 特征覆盖图

生产输入 CSV 的身份由原始元数据提供：

```csv
path,patient_id,slide_id,tile_id,x0,y0,width0,height0,mpp_x_um,mpp_y_um
patches/A.png,真实患者键,真实切片键,原tile键,4096,2048,512,512,,
```

```bash
python examples/real/patch_features.py --manifest /your/patches.csv --coordinate-space level0 --identity-scope patient --output /fresh/patch-features
```

必需列 `path,patient_id,slide_id,tile_id`；`level0` 另必需 `x0,y0,width0,height0`。坐标是原始 source/level0 左上角像素边界，不从 CSV行号、tile数或文件名猜位置。实际 patch 解码像素尺寸与 level0 extent 分开。MPP可空，不从20×、DPI或像素尺寸推断。边缘 padding 必须由 `valid_x,valid_y,valid_width,valid_height` 声明，否则有效区为提供文件的全部像素；不能在缺元数据时识别上游未知 padding。

普通输入支持 PNG/JPEG RGB8、单页 TIFF；本颜色/QC入口要求 RGB8，灰度纹理请另用明确的灰度 API。`--qc-config` 传原生 QC 配置，默认原始分数、无隐式排除阈值；`--color-config` 传原生固定边界 JSON。默认边界完整写入 provenance，输出7个明确标量：R/G/B均值、Lab-L均值、Hue resultant length、组织比例、Laplacian方差。Hue角度不作为线性聚合特征。

输出 `tiles.json` 是带已算特征的 canonical TileTable，`tiles-geometry.json` 是对应无特征版本，供用户外部 embedding 按完整显式 tile ID 集接入；不能把等行数当对齐，也不能隐式覆盖已有 feature matrix。`tile-features.csv`、`slide-features.csv`、`patient-features.csv` 均有 native manifest 和 `*-features-metadata.csv`；完整几何另保留 `tiles-metadata.csv`。默认先 tile→slide六统计，再切片等权取患者 mean，列名例如 `patient_mean(slide_tile_mean(rgb_R_mean))`。无结局、无训练动作。

`--overlay-feature rgb_R_mean` 只回填实际测量的 tile 特征，绝不广播患者概率。`--canvases` 可传 `slide_id -> Canvas` JSON；未提供时使用已有 tile 坐标计算的覆盖范围，明确不是未知整张 WSI 尺寸。`--overlay-pixel-size` 为显示采样的 source-pixel step，不是 MPP。输出分块NPZ、真实coverage/透明未覆盖及 overview PNG；过大预览明确跳过并建议提高step，不静默分配全WSI。

只有独立普通原图而没有真患者/WSI身份时，可显式作图像处理：

```csv
path,patient_id,slide_id,tile_id,mpp_x_um,mpp_y_um
/existing/ihc.png,operator-source-group,operator-source-image,source-file-ihc,,
```

```bash
python examples/real/patch_features.py --manifest /your/source-image.csv --coordinate-space image-local --identity-scope source-image --output /fresh/image-only
```

该模式每 slide/group 只允许一个完整源图，坐标为该文件已知局部像素网格，不是原 WSI 放置坐标；不能混入冲突的原始坐标列。输出 `group-features.csv` 而非患者特征声明。操作者 source-image 组ID不证明真实患者或原组织切片身份。整个源图为单个tile时，特征覆盖图自然均匀，表示单个tile标量，不是局部细胞预测。`preview/` 另展示实际原图与 Rust QC tissue threshold mask；这不是自动分割或组织学诊断。

## 本机实际文件验证

[`validation.json`](validation.json) 记录运行命令、源 SHA、实际输出与适用边界；原数据/医学预览在外部目录，没有作为测试数据或源码资产提交。

- 冻结公开 PAT1：实际原始NIfTI204×201×60＋GTV-1，0失败，basic27列；ROI125256、mean−46.88272018905282、体素支持域体积358681.451871528mm³；真实最大ROI轴向z=23预览已检查。无结局/队列生成。
- 本机现成 `scikit-image0.24.0` `data/ihc.png`：实际512×512 RGB8 IHC源图，0失败；真实组织QC比例0.6367225646972656，R均值177.25387954711914，MPP未知。上游说明为 CMMI IHC原图、**No known copyright restrictions**；不把该说明改写成未经确认的CC0，不提交原PNG，不把source组ID称为真实患者。见[上游文档](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.immunohistochemistry)。
- 契约测试验证缺少原始坐标拒绝、图像局部模式不拼造布局、ID保留前导零、未知MPP为空、Git内缓存拒绝和不匹配元数据不覆盖。

本页处理真实输入的文件/计算/显示能力，不建立临床模型有效性或数据使用许可之外的权利。带结局建模请使用用户真实表和患者分组；这些脚本不补造缺失研究数据。
