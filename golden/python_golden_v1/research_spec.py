"""Small independent specifications for keyed feature joins and paired delta.

Rows are normalized metadata plus a name->optional-float mapping, not a second
production FeatureTable implementation. Missing numeric values use None.
"""
import math


METADATA = ["sample_id", "patient_id", "slide_id", "lesion_id", "modality", "region", "timepoint", "center"]


def keyed(rows, fields):
    if not fields or len(fields) != len(set(fields)) or any(field not in METADATA for field in fields):
        raise ValueError("explicit unique metadata keys are required")
    result = {}
    samples = set()
    for row in rows:
        if not row.get("sample_id") or row["sample_id"] in samples:
            raise ValueError("sample IDs must be nonempty and unique within each source")
        samples.add(row["sample_id"])
        key = tuple(row.get(field) for field in fields)
        if any(not isinstance(value, str) or not value.strip() for value in key):
            raise ValueError("join/pair key fields must be present nonempty strings")
        if key in result:
            raise ValueError("duplicate explicit key")
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)) for value in row["values"].values()):
            raise ValueError("observed numeric values must be finite")
        result[key] = row
    return result


def concat(sources, key_fields, varying_fields, already_registered):
    """Output follows the first source row order and declared source/feature order."""
    if not sources or not already_registered:
        raise ValueError("caller must explicitly declare required image registration completed")
    prefixes = [source["prefix"] for source in sources]
    if len(set(prefixes)) != len(prefixes) or any(not isinstance(value, str) or not value.strip() for value in prefixes):
        raise ValueError("source prefixes must be nonempty and unique")
    if len(set(varying_fields)) != len(varying_fields) or any(field not in ["modality", "region"] for field in varying_fields):
        raise ValueError("only explicit modality/region differences may be allowed")
    maps = [keyed(source["rows"], key_fields) for source in sources]
    if any(set(mapping) != set(maps[0]) for mapping in maps[1:]):
        raise ValueError("source key sets differ; no implicit inner join")
    names = []
    for source in sources:
        feature_names = source["feature_names"]
        if not feature_names or len(set(feature_names)) != len(feature_names) or any(not name for name in feature_names):
            raise ValueError("feature names must be nonempty and unique")
        if any(set(row["values"]) != set(feature_names) for row in source["rows"]):
            raise ValueError("row feature definitions differ")
        names.extend(source["prefix"] + "::" + name for name in feature_names)
    if len(set(names)) != len(names):
        raise ValueError("prefixed feature names collide")
    output = []
    for anchor in sources[0]["rows"]:
        key = tuple(anchor[field] for field in key_fields)
        aligned = [mapping[key] for mapping in maps]
        metadata = {field: anchor.get(field) for field in METADATA}
        for field in METADATA:
            if field == "sample_id":
                continue  # Source-specific IDs are retained in provenance.
            values = [row.get(field) for row in aligned]
            if any(value != values[0] for value in values[1:]):
                if field not in varying_fields:
                    raise ValueError("conflicting source metadata: " + field)
                metadata[field] = None
        output.append(dict(metadata=metadata, values=[row["values"][name] for source, row in zip(sources, aligned) for name in source["feature_names"]],
                           source_sample_ids={prefix: row["sample_id"] for prefix, row in zip(prefixes, aligned)},
                           source_metadata={prefix: {field: row.get(field) for field in varying_fields} for prefix, row in zip(prefixes, aligned)}))
    return dict(feature_names=names, key_fields=key_fields, varying_fields=varying_fields, rows=output, registration="caller_declared_already_registered; no registration algorithm executed")


def delta(rows, feature_names, baseline, followup, zero_denominator="error"):
    if not baseline or not followup or baseline == followup or zero_denominator not in ["error", "missing"]:
        raise ValueError("distinct explicit timepoints and known zero-denominator policy required")
    fields = ["patient_id", "lesion_id", "modality", "region"]
    selected = {}
    for timepoint in [baseline, followup]:
        selected[timepoint] = keyed([row for row in rows if row.get("timepoint") == timepoint], fields)
    if any(row.get("timepoint") not in [baseline, followup] for row in rows):
        raise ValueError("unexpected timepoint; explicitly select the requested pair first")
    if not selected[baseline] or set(selected[baseline]) != set(selected[followup]):
        raise ValueError("unpaired baseline/followup keys")
    if not feature_names or len(feature_names) != len(set(feature_names)) or any(set(row["values"]) != set(feature_names) for row in rows):
        raise ValueError("paired feature definitions differ")
    result = []
    for key in sorted(selected[baseline]):
        before = selected[baseline][key]; after = selected[followup][key]
        if before.get("center") != after.get("center") or before.get("slide_id") != after.get("slide_id"):
            raise ValueError("paired metadata conflict outside declared timepoint")
        values = []
        for name in feature_names:
            a, b = before["values"][name], after["values"][name]
            if a is None or b is None:
                values.extend([None, None, None])  # Missing propagation precedes zero-denominator checks.
                continue
            difference = b - a
            if a == 0:
                if zero_denominator == "error":
                    raise ValueError("zero baseline denominator")
                ratios = [None, None]
            else:
                ratios = [b / a, difference / a]
            if any(value is not None and not math.isfinite(value) for value in [difference] + ratios):
                raise ValueError("delta numeric overflow")
            values.extend([difference] + ratios)
        result.append(dict(key=list(key), baseline_sample_id=before["sample_id"], followup_sample_id=after["sample_id"], values=values))
    return dict(key_fields=fields, feature_names=[name + "::" + method for name in feature_names for method in ["difference", "ratio", "relative_change"]], rows=result,
                baseline=baseline, followup=followup, zero_denominator=zero_denominator)
