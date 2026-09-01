"""Tests for near-duplicate detection.

These use synthetic vectors rather than real embeddings, so they run in milliseconds and
test the logic rather than the model. The pairing and grouping code is where an off-by-one
would silently either miss duplicates or delete good data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dedup import (
    detector_scores,
    find_candidate_pairs,
    group_duplicates,
    sample_for_review,
)


def unit_vectors(rows: list[list[float]]) -> np.ndarray:
    array = np.array(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


# --- pair finding ---------------------------------------------------------------------

def test_within_set_finds_the_similar_pair_only():
    # 0 and 1 are near-identical; 2 is orthogonal to both.
    vectors = unit_vectors([[1, 0, 0], [0.999, 0.045, 0], [0, 0, 1]])
    pairs = find_candidate_pairs(vectors, threshold=0.9, verbose=False)
    assert len(pairs) == 1
    assert set(pairs.iloc[0][["index_a", "index_b"]]) == {0, 1}


def test_within_set_excludes_self_matches():
    vectors = unit_vectors([[1, 0], [0, 1]])
    pairs = find_candidate_pairs(vectors, threshold=0.5, verbose=False)
    assert ((pairs["index_a"] != pairs["index_b"]).all())


def test_within_set_reports_each_pair_once():
    vectors = unit_vectors([[1, 0], [1, 0.01], [1, 0.02]])
    pairs = find_candidate_pairs(vectors, threshold=0.9, verbose=False)
    unordered = {frozenset((a, b)) for a, b in zip(pairs["index_a"], pairs["index_b"])}
    assert len(unordered) == len(pairs)


def test_block_size_does_not_change_the_result():
    # The blocked loop carries a row offset; a mistake there shifts indices silently.
    rng = np.random.default_rng(0)
    vectors = unit_vectors(rng.normal(size=(40, 8)).tolist())
    full = find_candidate_pairs(vectors, threshold=0.5, block=1000, verbose=False)
    blocked = find_candidate_pairs(vectors, threshold=0.5, block=7, verbose=False)
    assert len(full) == len(blocked)
    as_set = lambda frame: {  # noqa: E731
        (int(a), int(b)) for a, b in zip(frame["index_a"], frame["index_b"])
    }
    assert as_set(full) == as_set(blocked)


def test_cross_set_search_keeps_both_directions_of_index():
    left = unit_vectors([[1, 0], [0, 1]])
    right = unit_vectors([[1, 0.01], [5, 5]])
    pairs = find_candidate_pairs(left, right, threshold=0.99, verbose=False)
    assert len(pairs) == 1
    assert int(pairs.iloc[0]["index_a"]) == 0
    assert int(pairs.iloc[0]["index_b"]) == 0


def test_results_are_sorted_by_descending_similarity():
    rng = np.random.default_rng(1)
    vectors = unit_vectors(rng.normal(size=(30, 6)).tolist())
    pairs = find_candidate_pairs(vectors, threshold=0.0, verbose=False)
    assert pairs["similarity"].is_monotonic_decreasing


# --- grouping -------------------------------------------------------------------------

def test_grouping_is_transitive():
    # A~B and B~C means all three are the same eye, so one group of three.
    pairs = pd.DataFrame({"index_a": [0, 1], "index_b": [1, 2]})
    groups = group_duplicates(pairs, n_items=5)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_grouping_keeps_separate_clusters_apart():
    pairs = pd.DataFrame({"index_a": [0, 3], "index_b": [1, 4]})
    groups = group_duplicates(pairs, n_items=6)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [3, 4]]


def test_singletons_are_not_groups():
    groups = group_duplicates(pd.DataFrame({"index_a": [], "index_b": []}), n_items=4)
    assert groups == []


# --- detector scoring -----------------------------------------------------------------

def test_detector_scores_are_arithmetically_right():
    labelled = pd.DataFrame({
        "similarity": [0.99, 0.98, 0.95, 0.91],
        "is_duplicate": [1, 0, 1, 0],
    })
    scores = detector_scores(labelled, threshold=0.97)
    assert scores["true_positive"] == 1      # 0.99 predicted dup, is dup
    assert scores["false_positive"] == 1     # 0.98 predicted dup, is not
    assert scores["false_negative"] == 1     # 0.95 not predicted, is dup
    assert scores["precision"] == 0.5
    assert scores["recall_within_pool"] == 0.5


def test_detector_scores_handle_no_predictions():
    labelled = pd.DataFrame({"similarity": [0.5], "is_duplicate": [1]})
    scores = detector_scores(labelled, threshold=0.99)
    assert scores["precision"] == 0.0
    assert scores["f1"] == 0.0


# --- review sampling ------------------------------------------------------------------

def test_review_sample_spans_multiple_bands():
    # Sampling only the top matches would estimate precision at the easy end and say
    # nothing about where the threshold belongs.
    rng = np.random.default_rng(2)
    candidates = pd.DataFrame({
        "index_a": range(400),
        "index_b": range(400, 800),
        "similarity": rng.uniform(0.90, 1.0, size=400),
    })
    sample = sample_for_review(candidates, n=50)
    assert sample["band"].nunique() >= 3
    assert (sample["is_duplicate"] == "").all()


def test_review_sample_is_deterministic():
    rng = np.random.default_rng(3)
    candidates = pd.DataFrame({
        "index_a": range(200),
        "index_b": range(200, 400),
        "similarity": rng.uniform(0.90, 1.0, size=200),
    })
    first = sample_for_review(candidates, n=30, seed=7)
    second = sample_for_review(candidates, n=30, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_review_sample_survives_an_empty_pool():
    empty = pd.DataFrame({"index_a": [], "index_b": [], "similarity": []})
    assert len(sample_for_review(empty)) == 0
