from __future__ import annotations

import pandas as pd

from retina_fl.manifest import _build_from_original_per_eye, assign_clients, assign_patient_splits, audit_manifest, classify_eye


def test_eye_label_policy_is_explicit():
    assert classify_eye("normal fundus") == ("normal", "accepted")
    assert classify_eye("cataract") == ("cataract", "accepted")
    assert classify_eye("glaucoma") == ("glaucoma", "accepted")
    assert classify_eye("suspected glaucoma") == (None, "ambiguous_glaucoma_keyword")
    assert classify_eye("normal fundus, cataract") == (None, "multiple_target_labels")


def test_patient_grouping_prevents_split_and_client_leakage():
    rows = []
    labels = ["normal", "glaucoma", "cataract"]
    for patient in range(60):
        for eye in ("left", "right"):
            label = labels[patient % len(labels)]
            rows.append({
                "image_id": f"{patient}_{eye}.jpg",
                "patient_id": str(patient),
                "image_path": f"/not-required/{patient}_{eye}.jpg",
                "label": label,
                "label_index": labels.index(label),
                "target_normal": int(label == "normal"),
                "target_glaucoma": int(label == "glaucoma"),
                "target_cataract": int(label == "cataract"),
            })
    frame, _ = assign_patient_splits(pd.DataFrame(rows), [0.64, 0.16, 0.20], 2025)
    frame = assign_clients(frame, 4, 2025)
    report = audit_manifest(frame, require_files=False)
    assert report["ok"]
    assert frame.groupby("patient_id")["split"].nunique().max() == 1
    assert frame[frame["split"] == "train"].groupby("patient_id")["client_id"].nunique().max() == 1


def test_original_per_eye_multilabel_semantics_are_preserved(tmp_path):
    (tmp_path / "17_left.jpg").touch()
    source = pd.DataFrame([{"ID": "17_left.jpg", "normal": 0, "glaucoma": 1, "cataract": 1}])
    rows, exclusions = _build_from_original_per_eye(source, tmp_path, exclude_multilabel=False)
    assert not exclusions
    assert rows[0]["patient_id"] == "17"
    assert rows[0]["label"] == "glaucoma+cataract"
    assert [rows[0]["target_normal"], rows[0]["target_glaucoma"], rows[0]["target_cataract"]] == [0, 1, 1]

