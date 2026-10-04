"""Fast checks that need neither the ODIR images nor a trained checkpoint."""
import inspect

import numpy as np
import pytest
import torch

from src.data.preprocessing import INPUT_SIZE, to_tensor
from src.data.splits import fold_sets, leakage_report, load_split
from src.models.convnext_tiny import ConvNeXtScreener
from src.utils.config import load_config


@pytest.fixture(scope="module")
def net():
    return ConvNeXtScreener(4, pretrained=False).eval()


def test_config_names_convnext_as_the_single_model():
    config = load_config()
    assert config["model"]["name"] == "convnext_tiny"
    assert config["model"]["n_classes"] == len(config["data"]["labels"]) == 4


def test_split_is_patient_disjoint():
    frame = load_split(require_images=False)
    report = leakage_report(frame)
    assert report["patients_in_more_than_one_fold"] == 0
    assert report["duplicate_image_ids"] == 0
    assert report["images"] == 5056 and report["patients"] == 2909


@pytest.mark.parametrize("k", range(5))
def test_every_fold_assignment_keeps_patients_apart(k):
    train, val, test, val_fold = fold_sets(load_split(require_images=False), k)
    assert val_fold == (k + 1) % 5
    assert not set(train.patient_id) & set(test.patient_id)
    assert not set(val.patient_id) & set(test.patient_id)
    assert len(train) + len(val) + len(test) == 5056


def test_output_shape_and_finiteness(net):
    with torch.no_grad():
        out = net(torch.randn(3, 3, INPUT_SIZE, INPUT_SIZE))
    assert tuple(out.shape) == (3, 4)
    assert torch.isfinite(out).all()


def test_stem_is_frozen_and_stays_in_eval(net):
    net.train()
    assert not net.stem.training
    assert all(not p.requires_grad for p in net.stem.parameters())
    assert all(p.requires_grad for p in net.fc.parameters())
    net.eval()


def test_head_is_layernorm_then_linear(net):
    assert isinstance(net.norm, torch.nn.LayerNorm) and net.norm.normalized_shape == (768,)
    assert isinstance(net.fc, torch.nn.Linear) and net.fc.in_features == 768 and net.fc.out_features == 4


def test_feature_map_shapes(net):
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    stem = net.stem_features(x)
    assert tuple(stem.shape) == (1, 384, 14, 14)
    assert tuple(net.last_stage(stem).shape) == (1, 768, 7, 7)


def test_preprocessing_normalises_with_imagenet_statistics():
    grey = np.full((INPUT_SIZE, INPUT_SIZE, 3), 255, dtype=np.uint8)
    x = to_tensor(grey)
    assert tuple(x.shape) == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert torch.allclose(x[0, :, 0, 0], (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225]), atol=1e-5)


def test_gradcam_produces_a_normalised_map(net):
    from src.explainability.gradcam import grad_cam
    cam, index, logits = grad_cam(net, torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
    assert cam.shape == (INPUT_SIZE, INPUT_SIZE)
    assert 0 <= cam.min() and cam.max() <= 1
    assert index == int(logits.argmax())


def test_predict_image_does_not_accept_a_document():
    from src.app.pipeline import ScreeningPipeline
    assert list(inspect.signature(ScreeningPipeline.predict_image).parameters) == ["self", "image_path"]
