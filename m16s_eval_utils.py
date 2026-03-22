"""Shared helpers for Micro16S evaluation scripts."""

import json


def resolve_region_indices_json_path(dataset_split_dir):
    marker = "/seqs/split_"
    if marker not in dataset_split_dir:
        raise ValueError(
            f"DATASET_SPLIT_DIR must contain '{marker}' so region_indices.json can be auto-resolved. "
            f"Got: {dataset_split_dir}"
        )
    return dataset_split_dir.split(marker)[0] + "/seqs/encoded/region_indices.json"


def load_region_index_mappings(region_indices_json_path):
    with open(region_indices_json_path, "r") as f:
        payload = json.load(f)
    raw = payload.get("region_indices", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"region_indices.json is missing a non-empty 'region_indices' mapping: {region_indices_json_path}"
        )

    idx_to_id = {}
    id_to_idx = {}
    for idx_raw, region_id_raw in raw.items():
        idx = int(idx_raw)
        region_id = str(region_id_raw)
        idx_to_id[idx] = region_id
        if region_id in id_to_idx and id_to_idx[region_id] != idx:
            raise ValueError(
                f"Duplicate region_id '{region_id}' in {region_indices_json_path} "
                f"for indices {id_to_idx[region_id]} and {idx}."
            )
        id_to_idx[region_id] = idx
    return idx_to_id, id_to_idx


def normalize_region_selection(region_values, idx_to_id, id_to_idx, setting_name, region_indices_json_path):
    if region_values is None:
        return None

    normalized = []
    unknown = []
    seen = set()
    for value in region_values:
        idx = None
        if isinstance(value, int):
            idx = value
        elif isinstance(value, str):
            if value in id_to_idx:
                idx = id_to_idx[value]
            elif value.strip().isdigit():
                idx = int(value.strip())
            else:
                unknown.append(value)
                continue
        else:
            unknown.append(value)
            continue

        if idx not in idx_to_id:
            unknown.append(value)
            continue
        if idx not in seen:
            normalized.append(idx)
            seen.add(idx)

    if unknown:
        raise ValueError(
            f"{setting_name} contains unknown region IDs/indices: {unknown}. "
            f"Use values from {region_indices_json_path}."
        )
    return tuple(normalized)


def normalize_single_region_value(region_value, idx_to_id, id_to_idx, setting_name, region_indices_json_path):
    if region_value is None:
        return None
    normalized = normalize_region_selection(
        (region_value,),
        idx_to_id,
        id_to_idx,
        setting_name,
        region_indices_json_path,
    )
    return normalized[0]


def build_region_export_fields(region_idx, idx_to_id):
    idx = int(region_idx)
    if idx not in idx_to_id:
        raise ValueError(f"Unknown region_idx={idx}; not present in REGION_IDX_TO_ID_MAPPING.")
    return {"region_idx": idx, "region_id": idx_to_id[idx]}
