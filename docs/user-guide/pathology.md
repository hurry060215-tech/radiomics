# 自有 patch、外部 embedding 与患者建模

本轮处理普通图像/已解码数据和已有 mask/embedding。它不负责通用 WSI/KFB/厂商文件解码、不训练深度骨干、不自动分割，也不从文件名推断患者或切片坐标。

## 路径 A：已经有 patch 文件及真实坐标

准备 `my-pathology/patches.csv`，每行一个已经存在的 PNG/JPEG RGB8 patch；单页 TIFF 的部分像素类型另有原生支持。示例：

```csv
path,patient_id,slide_id,tile_id,x0,y0,width0,height0,mpp_x_um,mpp_y_um
patches/A_001.png,P001,SLIDE_A,A_001,1000,2000,512,512,0.5,0.5
patches/A_002.png,P001,SLIDE_A,A_002,1256,2000,512,512,0.5,0.5
patches/B_001.png,P001,SLIDE_B,B_001,0,0,512,512,0.5,0.5
```

`path` 相对于 manifest 所在目录。patient→slide→tile 都是明确身份；slide_id/tile_id应在输入中保持唯一，不同患者不能共享同一slide_id。`x0/y0` 是 **level-0 左上角像素边缘坐标**，x向右、y向下；width0/height0 是在这个坐标系中的覆盖范围，可能不同于解码 patch 的宽高。不能从“第20个patch”推算任意位置，也不能把中心坐标直接填入左上角字段。

MPP 可留空；没有可信采集尺度时只使用像素单位，不能由20x或DPI猜测。已经 padding 的输入可声明 `valid_x/valid_y/valid_width/valid_height`，这些是解码patch中的有效像素矩形；padding不进入有效覆盖率分母。若不清楚padding来源，先核实再填写。

```powershell
python examples/real/patch_features.py --manifest my-pathology/patches.csv --output my-pathology/features-run-01 --overlay-feature rgb_R_mean --overlay-pixel-size 4
```

脚本读取已有 patch，输出QC原始分数及排除原因、颜色特征、tile/slide/patient特征表、metadata与manifest、坐标覆盖图。它不会删除原patch。QC阈值可用 `--qc-config qc.json`，例如：

```json
{
  "max_white_fraction": 0.8,
  "min_tissue_fraction": 0.1
}
```

这两个数是显式示例阈值，需要在自己的染色/组织类型上检查，不是所有病理的推荐默认。颜色固定bin配置可由 `--color-config` 提供，定义见[病理像素API](../pathology-pixels.md)。组织颜色、模糊与黑边分数是QC信息，不是真实肿瘤结局。

常用输出：

| 文件 | 层级与用途 |
|---|---|
| `tiles.json` | 带实际颜色特征、QC、身份和坐标的完整 TileTable |
| `tiles-geometry.json` | 同一身份/坐标/QC但未附特征，给外部 embedding 接入，避免覆盖已有特征 |
| `tile-features.csv` | 每tile一行，sample_id=tile_id |
| `slide-features.csv` | 每slide一行，sample_id=slide_id |
| `patient-features.csv` | 每patient一行，sample_id=patient_id；默认tile→slide，再切片等权到患者 |
| `*-features-metadata.csv`、`*.manifest.json` | 对应表的身份/来源/单位与摘要；如 `patient-features-metadata.csv` |

平均时先得到每切片统计，再对患者切片等权，避免patch更多的切片自动获得更大患者权重。没有显式面积权重，不把patch数当组织面积。所有patch都被QC排除的切片保留空状态，先审阅排除/聚合报告，不以全零患者向量继续建模。

## 从真实患者结局训练

建立真正每患者一行的 `clinical.csv`，含 `patient_id/cohort/response`，可再加真实临床变量。这里不将slide诊断从tile颜色临时生成。`patient-study.json` 完整例子：

```json
{
  "schema_version": 1,
  "table": "features-run-01/patient-features.csv",
  "feature_manifest": "features-run-01/patient-features.csv.manifest.json",
  "metadata": "features-run-01/patient-features-metadata.csv",
  "clinical": "clinical.csv",
  "clinical_key": "patient",
  "join_policy": "exact",
  "id_column": "sample_id",
  "patient_column": "patient_id",
  "cohort_column": "cohort",
  "outcome_column": "response",
  "numeric_features": [
    "patient_mean(slide_tile_mean(rgb_R_mean))",
    "patient_mean(slide_tile_mean(rgb_G_mean))",
    "patient_mean(slide_tile_mean(rgb_B_mean))"
  ],
  "categorical_features": [],
  "optional_metadata": {},
  "missing_values": [""],
  "nan_as_missing": false
}
```

```powershell
python examples/real/train_table.py train --study my-pathology/patient-study.json --task binary --classes no yes --fit-unit patient --scoring-unit patient --plots --output my-pathology/patient-model-01
```

这些三项颜色统计只是容易审查的起点，不代表经过临床验证的诊断模型。外部患者仍用 `outcome_column=null` 新StudySpec和[同一加载预测命令](training.md)。若使用slide/tile行训练，患者ID仍必须保留，不能随机切分同患者的patch。

## 普通图片没有WSI全局坐标时

如果只有一张独立普通图片，其像素网格可作为该图片本身的坐标系。显式选择 `--coordinate-space image-local`，每slide只允许单文件。若也没有可证明的临床患者关联，使用 `--identity-scope source-image`，输出是 `group-features.csv` 和对应metadata，表示操作者定义的源图像组，不冒称临床患者，也不能据此进行患者级临床结论。

对于需要从完整普通图像切块的使用者，公共API提供真实切块，无需伪造patch布局：

```python
import ov_radiomics as rad
patches = rad.pathology.tile_image(
    "ordinary-slide.png", patient_id="P001", slide_id="SLIDE_A",
    patch_size=(256, 256), stride=(128, 128), edge="pad",
    origin0=(0.0, 0.0), pixel_size0=(1.0, 1.0),
    mpp_x_um=None, mpp_y_um=None
)
for patch in rad.pathology.qc_patches(patches, config={"max_white_fraction": 0.8}):
    print(patch.record["tile_id"], patch.record["x0"], patch.record["y0"], patch.qc)
```

这里明确声明普通图片自己的网格；它不是读取金字塔WSI。上面的循环只示范流式处理和真实坐标，已有patch文件的完整导出路径用本页前面的manifest脚本。

## 路径 B：你已有 embedding

外部数值矩阵需要明确行ID。准备：

```text
embedding-study/
  embeddings.npy          # dense real tile×feature，allow_pickle=False可读
  embedding-tile-ids.json  # 与矩阵行一一对应的字符串ID数组
  embedding-names.json     # 与矩阵列对应的唯一特征名数组
  tiles-geometry.json     # 上述明确的TileTable几何/QC，features为空
```

行ID可以与几何顺序不同；ID集合必须准确相同，重复/缺失/多余都会拒绝。以下代码实际调用原生ID对齐和聚合，不依靠“行数相等”：

```python
import json
from pathlib import Path
import numpy as np
import ov_radiomics as rad

base = Path("embedding-study")
geometry = json.loads((base / "tiles-geometry.json").read_text(encoding="utf-8"))
ids = json.loads((base / "embedding-tile-ids.json").read_text(encoding="utf-8"))
names = json.loads((base / "embedding-names.json").read_text(encoding="utf-8"))
matrix = np.load(base / "embeddings.npy", allow_pickle=False)
table = rad.interop.from_arrays(matrix, tile_ids=ids, tiles=geometry,
                               feature_names=names, source="external/provided embeddings.npy")
aggregated = rad.pathology.aggregate(table, patient_stats=["mean"])
out = base / "aggregated-01"
out.mkdir(exist_ok=False)
aggregated.slide_features.save(out / "slide-features.parquet")
aggregated.patient_features.save(out / "patient-features.csv")
(out / "aggregation-report.json").write_text(
    json.dumps(aggregated.report, ensure_ascii=False, indent=2), encoding="utf-8"
)
```

然后按上面的patient-study模式连接真实结局，显式填写实际embedding聚合列名。不要把embedding维数、行序或缓存文件存在当作ID/历史有效性证明。该接入声明为 external/provided，不声称本包验证了上游深度模型或生成历史。

真实AnnData可用 `rad.interop.from_anndata(adata, tiles=geometry, tile_id_column="tile_id")`，保留真实obs ID/var顺序；大稀疏/backed矩阵需先自行评估是否适合明确dense分配。WSIData路径还要从 `wsi.shapes[tile_key]` 的真实矩形、`wsi.attrs["tile_spec"][tile_key]` 与 `wsi.tables[feature_key].X`/行ID证明关联。生成前snapshot、历史记录与level0转换的完整约束见[interop](../interop.md)。现在才给旧cache创建snapshot不能追认生成时几何。

## 覆盖图必须接收真实局部分数

patch脚本默认覆盖的是实际tile颜色特征，如 `rgb_R_mean`，不是肿瘤概率。真实tile-level模型分数也可以按tile ID接入并栅格化。重叠区域平均并保存coverage count，未覆盖区域NaN/透明；大图按块输出。患者模型概率只能在患者表解释，不能复制到每个tile后称为局部预测图。

已有instance/semantic mask统计可用 `rad.pathology.object_features`；semantic需要明确连通性后分组件。重叠patch中同一实例可能重复，没有全局实例ID就只报告patch-local计数。MPP缺失时不输出假定物理面积。
