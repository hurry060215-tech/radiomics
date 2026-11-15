"""Static Notebook-source inventory: hashes, topic IDs, imports and call symbols only.

Does not execute notebooks/import their packages, inspect outputs, open bytecode,
retain source text, export tables or reproduce third-party documentation/code.
Cell indices are zero-based positions in the original .ipynb cells array.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


TOPICS = {
    "image_geometry_preparation": r"resampl|重采样|spacing|N4|配准|register|SimpleITK",
    "image_filtering": r"wavelet|\bLoG\b|\bLBP\b|滤波|filtering",
    "extraction": r"radiomic|特征抽取|特征提取|extract.*feature",
    "scaling_missing_encoding": r"normalize|standard|scaler|标准化|归一化|one.hot|缺失|imput",
    "selection_univariate_correlation": r"相关性|单因素|univaria|correlation|pearson|spearman",
    "selection_sparse": r"lasso|L1|稀疏|coef_path",
    "icc": r"ICC|组内相关",
    "group_validation": r"cross.val|Stratified|KFold|交叉验证|train_test_split",
    "external_center_validation": r"多中心|外部验证|external|center|center-specific",
    "models": r"Logistic|SVM|XGBoost|LightGBM|RandomForest|GaussianNB|KNN|模型训练",
    "fusion": r"fusion|融合|ensemble|stacking",
    "score_formula": r"Rad.?score|预测公式|公式|coef_|intercept_",
    "nomogram": r"nomogram|列线图",
    "learning_curve": r"learning.curve|学习曲线",
    "metrics_comparison": r"ROC|AUC|Delong|DeLong|DCA|Hosmer|Brier|混淆|校准|calibration",
    "clinical_stats": r"统计|ttest|chi2|fisher|Mann|Kruskal|Wald|OR值",
    "shap_explanation": r"SHAP|Grad.CAM|Transformer",
    "survival": r"Cox|Kaplan|生存|KM曲线",
    "habitat_pathology": r"生境|病理|patholog|habitat|embedding",
    "deferred_batch_correction": r"ComBat|批次矫正|批次校正",
}


def topics(text):
    return sorted(key for key, pattern in TOPICS.items() if re.search(pattern, text, re.I))


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return parent + "." + node.attr if parent else None
    return None


def symbols(source, aliases):
    # Ignore IPython command lines for static syntax parsing; never evaluate them.
    text = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!", "?")))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return dict(parse_status="syntax_not_supported", imports=[], calls=[], definitions=[], call_keyword_names=[])
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split(".")[0]] = name.name if name.asname else name.name.split(".")[0]
                imports.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                full = (node.module + "." if node.module else "") + name.name
                aliases[name.asname or name.name] = full
                imports.append(full)
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name:
                head, separator, tail = name.partition(".")
                calls.add(aliases.get(head, head) + (separator + tail if separator else ""))
    definitions = sorted({node.name for node in ast.walk(tree)
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))})
    keyword_names = sorted({node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg})
    return dict(parse_status="parsed_static", imports=sorted(set(imports)), calls=sorted(calls),
                definitions=definitions, call_keyword_names=keyword_names)


def selected(path):
    if ".ipynb_checkpoints" in path.parts:
        return False
    directory = path.parts[0]
    if directory.startswith(("comp1-", "comp2-", "comp3-", "comp6-")):
        return True
    if directory.startswith("comp8-"):
        return path.stem in {"ICC特征筛选", "统计分析", "单_多因素回归", "指标汇总", "交叉验证AUC", "Metrics", "Combat批次矫正"}
    if directory.startswith("comp9-") and len(path.parts) > 1:
        return path.parts[1].startswith(("sol1. ", "sol2. "))
    return False


def inventory(root):
    root = root.resolve()
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8-sig")
    result = dict(schema="radiomics.onekey_static_symbols.v1",
                  scope="Only visible README and selected Notebook markdown/code source. No execution, output inspection, bytecode or source replication.",
                  index_convention="zero-based original cells array index",
                  readme=dict(path="README.md", sha256=hashlib.sha256(readme.read_bytes()).hexdigest(), topic_ids=topics(text)),
                  notebooks=[])
    for path in sorted(root.rglob("*.ipynb")):
        relative = path.relative_to(root)
        if not selected(relative):
            continue
        raw = path.read_bytes()
        notebook = json.loads(raw.decode("utf-8-sig"))
        record = dict(path=relative.as_posix(), sha256=hashlib.sha256(raw).hexdigest(), cells=[])
        aliases = {}
        for index, cell in enumerate(notebook.get("cells", [])):
            kind = cell.get("cell_type")
            if kind not in ("markdown", "code"):
                continue
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            entry = dict(index=index, cell_type=kind, source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(), topic_ids=topics(source))
            if kind == "code":
                entry.update(symbols(source, aliases))
            record["cells"].append(entry)
        result["notebooks"].append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Inventoried", len(result["notebooks"]), "notebooks; output fields contain no source/output text:", args.output)


if __name__ == "__main__":
    main()
