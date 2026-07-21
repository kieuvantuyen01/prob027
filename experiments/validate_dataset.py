#!/usr/bin/env python3
"""Validate a benchmark directory created by generate_instances.py."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_instances import target_from_clicks, target_sha256, validate_matrix


def validate_instance(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        instance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: cannot read JSON: {exc}"]

    required = ("name", "N", "c", "target", "generation")
    missing = [key for key in required if key not in instance]
    if missing:
        return None, [f"{path}: missing fields {missing}"]

    n, c, target = instance["N"], instance["c"], instance["target"]
    try:
        if not isinstance(n, int) or n < 1:
            raise ValueError("N must be a positive integer")
        if not isinstance(c, int) or c < 2:
            raise ValueError("c must be an integer >= 2")
        validate_matrix(target, n, c, "target")
    except ValueError as exc:
        errors.append(f"{path}: {exc}")
        return instance, errors

    digest = target_sha256(n, c, target)
    stored_digest = instance["generation"].get("target_sha256")
    if stored_digest != digest:
        errors.append(f"{path}: target_sha256 mismatch")

    guaranteed = instance.get("guaranteed_solvable", False)
    witness = instance.get("witness_clicks")
    if guaranteed and witness is None:
        errors.append(f"{path}: guaranteed_solvable requires witness_clicks")
    if witness is not None:
        try:
            validate_matrix(witness, n, c, "witness_clicks")
            if target_from_clicks(witness, c) != target:
                errors.append(f"{path}: witness does not produce target")
            witness_total = sum(sum(row) for row in witness)
            if instance.get("witness_total_clicks") != witness_total:
                errors.append(f"{path}: witness_total_clicks mismatch")
            if instance.get("known_upper_bound") != witness_total:
                errors.append(f"{path}: known_upper_bound mismatch")
        except ValueError as exc:
            errors.append(f"{path}: {exc}")

    if "known_optimum" in instance and instance["known_optimum"] is not None:
        optimum = instance["known_optimum"]
        upper_bound = instance.get("known_upper_bound")
        if not isinstance(optimum, int) or optimum < 0:
            errors.append(f"{path}: known_optimum must be a non-negative integer")
        if upper_bound is not None and optimum > upper_bound:
            errors.append(f"{path}: known_optimum exceeds known_upper_bound")

    return instance, errors


def validate_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        return {}, [f"missing manifest: {manifest_path}"]

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    manifest_paths = {row["path"] for row in manifest_rows}
    if len(manifest_paths) != len(manifest_rows):
        errors.append("manifest contains duplicate paths")

    digests: dict[str, Path] = {}
    counts_by_split: Counter[str] = Counter()
    counts_by_family: Counter[str] = Counter()
    checked_paths: set[str] = set()

    for row in manifest_rows:
        relative_path = row["path"]
        instance_path = dataset_dir / relative_path
        if not instance_path.is_file():
            errors.append(f"manifest file is missing: {relative_path}")
            continue
        instance, instance_errors = validate_instance(instance_path)
        errors.extend(instance_errors)
        checked_paths.add(relative_path)
        if instance is None:
            continue

        digest = instance["generation"].get("target_sha256")
        if digest in digests:
            errors.append(
                f"duplicate target: {instance_path} and {digests[digest]}"
            )
        else:
            digests[digest] = instance_path

        for field in ("name", "split"):
            expected = instance["name"] if field == "name" else instance["generation"]["split"]
            if row[field] != str(expected):
                errors.append(f"{relative_path}: manifest {field} mismatch")
        if row["target_sha256"] != digest:
            errors.append(f"{relative_path}: manifest target_sha256 mismatch")

        counts_by_split[instance["generation"]["split"]] += 1
        counts_by_family[instance["generation"]["method"]] += 1

    instance_files = {
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.glob("*/*/*/*.json")
    }
    extra_files = instance_files - manifest_paths
    if extra_files:
        errors.append(f"instance files absent from manifest: {sorted(extra_files)}")
    unvisited = manifest_paths - checked_paths
    if unvisited:
        errors.append(f"manifest paths not checked: {sorted(unvisited)}")

    summary = {
        "instances": len(checked_paths),
        "by_split": dict(sorted(counts_by_split.items())),
        "by_family": dict(sorted(counts_by_family.items())),
        "unique_targets": len(digests),
    }
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an Alien Tiles dataset")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    summary, errors = validate_dataset(args.dataset_dir)
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Validation PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
