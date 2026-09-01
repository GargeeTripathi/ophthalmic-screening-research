"""Near-duplicate detection over image embeddings.

Two questions, both of which matter to the study:

  WITHIN ODIR    Near-duplicates inside the training set. If two copies of the same eye
                 land in different folds, the test score measures memorisation. This is
                 leakage the patient-disjoint split alone does not catch, because a
                 duplicate can carry a different patient id.

  ODIR vs MERGED The merged benchmark is partly built from ODIR. Any of its images that
                 also appear in ODIR cannot serve as cross-source test data.

The similarity threshold is deliberately NOT hard-coded as truth. `sample_for_review`
produces a stratified sample of candidate pairs for hand-labelling, and the resulting
labels are what fix the operating point. A detector whose threshold was never calibrated
is a guess with a decimal point on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Deliberately low: we want the review sample to span the whole decision region, including
# pairs that are clearly not duplicates, so precision can actually be estimated.
CANDIDATE_THRESHOLD = 0.90


def _pairs_from_block(similarity: np.ndarray, row_offset: int, threshold: float,
                      same_set: bool) -> list[tuple[int, int, float]]:
    if same_set:
        # Keep only the upper triangle so each pair appears once and self-matches drop out.
        rows, cols = np.where(similarity >= threshold)
        keep = (rows + row_offset) < cols
        rows, cols = rows[keep], cols[keep]
    else:
        rows, cols = np.where(similarity >= threshold)
    return [(int(r + row_offset), int(c), float(similarity[r, c]))
            for r, c in zip(rows, cols)]


def find_candidate_pairs(vectors_a: np.ndarray, vectors_b: np.ndarray | None = None,
                         threshold: float = CANDIDATE_THRESHOLD,
                         block: int = 512, verbose: bool = True) -> pd.DataFrame:
    """Cosine-similar pairs above `threshold`.

    Vectors are already L2-normalised, so a dot product is the cosine similarity.
    Computed in row blocks: the full 9k x 9k matrix would be several hundred megabytes.

    With `vectors_b` omitted, searches within one set. Otherwise searches across two.
    """
    same_set = vectors_b is None
    right = vectors_a if same_set else vectors_b

    found: list[tuple[int, int, float]] = []
    for start in range(0, len(vectors_a), block):
        stop = min(start + block, len(vectors_a))
        similarity = vectors_a[start:stop] @ right.T
        found.extend(_pairs_from_block(similarity, start, threshold, same_set))
        if verbose and (start // block) % 5 == 0:
            print(f"    compared {stop}/{len(vectors_a)}   candidates so far: {len(found)}")

    frame = pd.DataFrame(found, columns=["index_a", "index_b", "similarity"])
    return frame.sort_values("similarity", ascending=False).reset_index(drop=True)


def sample_for_review(candidates: pd.DataFrame, n: int = 100,
                      seed: int = 42) -> pd.DataFrame:
    """Stratified sample across similarity bands, for hand-labelling.

    Sampling only the top matches would estimate precision at the easy end and tell us
    nothing about where the threshold should sit. Bands spread the sample across the
    region where the decision is actually difficult.
    """
    bands = [(0.90, 0.94), (0.94, 0.96), (0.96, 0.98), (0.98, 0.995), (0.995, 1.01)]
    rng = np.random.default_rng(seed)
    per_band = max(1, n // len(bands))

    chunks = []
    for low, high in bands:
        band = candidates[
            (candidates["similarity"] >= low) & (candidates["similarity"] < high)
        ]
        if band.empty:
            continue
        take = min(per_band, len(band))
        picked = band.iloc[rng.choice(len(band), size=take, replace=False)].copy()
        picked["band"] = f"{low:.3f}-{high:.3f}"
        chunks.append(picked)

    if not chunks:
        return candidates.head(0).assign(band=[], is_duplicate=[])
    sample = pd.concat(chunks).sort_values("similarity", ascending=False)
    # Blank column for the human to fill in: 1 = same eye, 0 = different eyes.
    sample["is_duplicate"] = ""
    return sample.reset_index(drop=True)


def group_duplicates(pairs: pd.DataFrame, n_items: int) -> list[list[int]]:
    """Connected components over confirmed duplicate pairs (union-find).

    Duplicates are transitive in practice - if A matches B and B matches C, all three are
    the same eye - so groups, not pairs, are what gets removed from a split.
    """
    parent = list(range(n_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["index_a"], pairs["index_b"]):
        root_a, root_b = find(int(a)), find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a

    groups: dict[int, list[int]] = {}
    for item in range(n_items):
        groups.setdefault(find(item), []).append(item)
    return [members for members in groups.values() if len(members) > 1]


def detector_scores(labelled: pd.DataFrame, threshold: float) -> dict:
    """Precision / recall / F1 of the detector at `threshold`, from hand labels.

    `labelled` needs a `similarity` column and an `is_duplicate` column of 0/1.
    Recall here is recall *within the reviewed candidate pool*, not over the whole
    dataset - pairs below the candidate threshold were never shown to the reviewer.
    That limitation belongs in the report.
    """
    truth = labelled["is_duplicate"].astype(int)
    predicted = (labelled["similarity"] >= threshold).astype(int)

    true_positive = int(((predicted == 1) & (truth == 1)).sum())
    false_positive = int(((predicted == 1) & (truth == 0)).sum())
    false_negative = int(((predicted == 0) & (truth == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": round(threshold, 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall_within_pool": round(recall, 4),
        "f1": round(f1, 4),
    }
