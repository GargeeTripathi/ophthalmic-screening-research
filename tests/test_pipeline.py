"""Tests for the reproducibility orchestrator.

These are cheap and they catch the failure mode that would otherwise only show up during a
full 70-minute rebuild: a stage pointing at a script that no longer exists, or declaring an
output path that nothing writes.
"""

from __future__ import annotations

import pytest

from scripts.run_all import STAGES, is_cached
from src.utils.config import REPO_ROOT, load_config


def test_every_stage_points_at_a_real_script():
    for stage, script, _, _ in STAGES:
        assert (REPO_ROOT / "scripts" / script).is_file(), f"{stage} -> missing {script}"


def test_stage_ids_are_unique():
    ids = [s[0] for s in STAGES]
    assert len(ids) == len(set(ids))


def test_every_stage_has_a_description():
    for stage, _, description, _ in STAGES:
        assert description.strip(), f"{stage} has no description"


def test_data_stages_declare_outputs():
    # The final three regenerate every run by design; everything before them must be
    # cacheable, or a re-run would repeat hours of work.
    for stage, _, _, outputs in STAGES[:-3]:
        assert outputs, f"{stage} declares no outputs, so it can never be skipped"


def test_declared_outputs_are_repo_relative():
    for stage, _, _, outputs in STAGES:
        for path in outputs:
            assert not path.startswith(("/", "\\")) and ":" not in path, \
                f"{stage} declares a non-relative output: {path}"


def test_is_cached_is_false_for_a_missing_output():
    assert not is_cached(["data/manifests/definitely_not_a_real_file.csv"])


def test_is_cached_is_false_for_an_empty_declaration():
    # An empty list means "always run", which must never read as cached.
    assert not is_cached([])


@pytest.mark.parametrize("experiment", ["E0b", "E1", "E1b", "E5", "E11", "E12", "E13"])
def test_experiments_in_the_pipeline_are_registered(experiment):
    # Anything the pipeline runs must be declared in the registry, so results can never
    # come from an experiment nobody wrote down.
    assert experiment in load_config("experiments")
