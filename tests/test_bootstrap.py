"""Regression tests for the project skeleton.

These lock invariants that are cheap to break and expensive to notice: a config key that
gets renamed, an output directory that disappears, and the dev/sealed IDP layout split
that makes the extraction numbers meaningful.

    pytest -q
"""

from __future__ import annotations

import pytest

from src.utils.config import REPO_ROOT, load_all, load_config, output_dir, source_path

CONFIGS = load_all()


def test_all_configs_parse():
    assert set(CONFIGS) == {"paths", "data", "training", "idp", "experiments"}
    assert all(isinstance(value, dict) and value for value in CONFIGS.values())


def test_expected_dirs_exist():
    for relative in ("configs", "src", "scripts", "tests", "data/manifests", "results"):
        assert (REPO_ROOT / relative).is_dir(), f"missing {relative}"


def test_output_dir_resolves_and_creates():
    assert output_dir("results").is_dir()
    assert output_dir("manifests").is_dir()


def test_source_paths_are_absolute():
    # Source data is read-only and lives outside the repo; a relative path here would
    # silently resolve against the caller's cwd.
    for key in ("odir_label_csv", "odir_images_train", "merged_root"):
        assert source_path(key).is_absolute()


def test_idp_dev_and_sealed_layouts_are_disjoint():
    layouts = CONFIGS["idp"]["layouts"]
    assert set(layouts["dev"]).isdisjoint(layouts["sealed"])
    assert layouts["dev"] and layouts["sealed"]


def test_every_layout_has_a_style():
    layouts = CONFIGS["idp"]["layouts"]
    styles = CONFIGS["idp"]["layout_styles"]
    for name in layouts["dev"] + layouts["sealed"]:
        assert name in styles, f"layout {name} has no style"


def test_degradation_params_cover_every_level():
    degradation = CONFIGS["idp"]["degradation"]
    assert len(degradation["levels"]) == len(degradation["names"])
    for level in degradation["levels"]:
        assert level in degradation["params"], f"level {level} has no parameters"


def test_train_corruption_rate_differs_from_test_rates():
    # If the training corruption rate is one of the test rates, the robustness study has
    # no out-of-distribution condition left to measure.
    corruption = CONFIGS["idp"]["diagnosis_corruption"]
    assert corruption["train_rate"] not in corruption["test_rates"]


def test_expected_class_counts_sum_to_total():
    expected = CONFIGS["data"]["expected"]
    assert sum(expected["images_per_class"].values()) == expected["images"]


def test_selection_fold_is_within_range():
    splits = CONFIGS["data"]["splits"]
    assert 0 <= splits["selection_fold"] < splits["n_folds"]


def test_planned_comparisons_reference_real_experiments():
    stats = CONFIGS["experiments"]["statistics"]
    for comparison in stats["planned_comparisons"]:
        experiment_id = comparison.split(":")[0].strip()
        assert experiment_id in CONFIGS["experiments"], f"unknown experiment {experiment_id}"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")
