"""Tests for weight persistence — save/load roundtrip for torch brain backends.

Verifies:
- Torch NN weight save/load roundtrip
- Torch Transformer weight save/load roundtrip
- Trainer state (baseline, global_step) preservation
- Load from nonexistent directory → graceful warning
- Weights survive across a BrainManager pair (save from one, load into fresh)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from config import default_config
from main import BrainManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain_mgr(brain_type: str = "torch_nn", seed: int = 42) -> BrainManager:
    """Create a BrainManager with the specified brain type."""
    cfg = default_config()
    cfg.brain.default = brain_type
    return BrainManager(cfg, seed)


def _init_torch_nn(mgr: BrainManager) -> None:
    """Force-initialize the torch NN registry and trainers."""
    mgr._ensure_torch_nn()


def _init_torch_tf(mgr: BrainManager) -> None:
    """Force-initialize the torch transformer registry and trainers."""
    mgr._ensure_torch_tf()


# ---------------------------------------------------------------------------
# Torch NN weight persistence
# ---------------------------------------------------------------------------

class TestTorchNNWeightPersistence:
    """Save and reload torch NN weights."""

    def test_roundtrip_weights_match(self, tmp_path: Path):
        mgr = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr)
        import torch
        originals: dict[str, dict[str, np.ndarray]] = {}
        for role in mgr._torch_nn_registry.roles():
            model = mgr._torch_nn_registry.get(role)
            originals[role] = {
                k: v.detach().cpu().numpy().copy()
                for k, v in model.state_dict().items()
            }

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for role in originals:
            model2 = mgr2._torch_nn_registry.get(role)
            loaded = model2.state_dict()
            for key in originals[role]:
                assert key in loaded, f"Missing key {key} in loaded torch_nn state"
                np.testing.assert_allclose(
                    loaded[key].detach().cpu().numpy(),
                    originals[role][key],
                    err_msg=f"Torch NN {role} {key} mismatch",
                )

    def test_weight_files_created(self, tmp_path: Path):
        mgr = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr)
        mgr.save_weights(tmp_path / "weights")
        npz_files = list((tmp_path / "weights").glob("torch_nn_*.npz"))
        assert len(npz_files) >= 4, f"Expected ≥4 torch_nn weight files, got {len(npz_files)}"


# ---------------------------------------------------------------------------
# Torch Transformer weight persistence
# ---------------------------------------------------------------------------

class TestTorchTransformerWeightPersistence:
    """Save and reload torch transformer weights."""

    def test_roundtrip_weights_match(self, tmp_path: Path):
        mgr = _make_brain_mgr("torch_transformer")
        _init_torch_tf(mgr)

        originals: dict[str, dict[str, np.ndarray]] = {}
        for role in mgr._torch_tf_registry.roles():
            model = mgr._torch_tf_registry.get(role)
            originals[role] = {
                k: v.detach().cpu().numpy().copy()
                for k, v in model.state_dict().items()
            }

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("torch_transformer")
        _init_torch_tf(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for role in originals:
            model2 = mgr2._torch_tf_registry.get(role)
            loaded = model2.state_dict()
            for key in originals[role]:
                assert key in loaded, f"Missing key {key} in loaded torch_transformer state"
                np.testing.assert_allclose(
                    loaded[key].detach().cpu().numpy(),
                    originals[role][key],
                    err_msg=f"Torch Transformer {role} {key} mismatch",
                )

    def test_weight_files_created(self, tmp_path: Path):
        mgr = _make_brain_mgr("torch_transformer")
        _init_torch_tf(mgr)
        mgr.save_weights(tmp_path / "weights")
        npz_files = list((tmp_path / "weights").glob("torch_transformer_*.npz"))
        assert len(npz_files) >= 4, (
            f"Expected ≥4 torch_transformer weight files, got {len(npz_files)}"
        )


# ---------------------------------------------------------------------------
# Trainer state persistence
# ---------------------------------------------------------------------------

class TestTrainerStatePersistence:
    """Trainer baseline & global_step survive save/load roundtrip."""

    def test_torch_nn_trainer_state(self, tmp_path: Path):
        mgr = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr)

        for trainer in mgr._torch_nn_trainers.values():
            trainer.baseline = 0.789
            trainer.global_step = 1234

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for trainer in mgr2._torch_nn_trainers.values():
            assert trainer.baseline == 0.789
            assert trainer.global_step == 1234

    def test_torch_transformer_trainer_state(self, tmp_path: Path):
        mgr = _make_brain_mgr("torch_transformer")
        _init_torch_tf(mgr)

        for trainer in mgr._torch_tf_trainers.values():
            trainer.baseline = 0.987
            trainer.global_step = 4321

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("torch_transformer")
        _init_torch_tf(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for trainer in mgr2._torch_tf_trainers.values():
            assert trainer.baseline == 0.987
            assert trainer.global_step == 4321


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestWeightPersistenceEdgeCases:
    """Edge cases: missing dir, empty dir, partial saves."""

    def test_load_nonexistent_dir_no_crash(self, tmp_path: Path, capsys):
        """Loading from a nonexistent directory prints a warning, doesn't crash."""
        mgr = _make_brain_mgr("torch_nn")
        mgr.load_weights(tmp_path / "does_not_exist")

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "warning" in captured.out.lower()

    def test_load_empty_dir_no_crash(self, tmp_path: Path, capsys):
        """Loading from an empty directory prints a message, doesn't crash."""
        empty = tmp_path / "empty_weights"
        empty.mkdir()
        mgr = _make_brain_mgr("torch_nn")
        mgr.load_weights(empty)

        captured = capsys.readouterr()
        assert "no weight files" in captured.out.lower()

    def test_save_creates_directory(self, tmp_path: Path):
        """Saving to a non-existent nested path creates the directory."""
        mgr = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr)

        nested = tmp_path / "a" / "b" / "c"
        mgr.save_weights(nested)
        assert nested.is_dir()
        assert len(list(nested.glob("torch_nn_*.npz"))) >= 4

    def test_uninit_registries_no_files(self, tmp_path: Path):
        """Saving a fresh BrainManager with no initialized registries produces no files."""
        mgr = _make_brain_mgr("rule_based")
        mgr.save_weights(tmp_path / "weights")

        weight_files = list((tmp_path / "weights").glob("*.npz"))
        assert len(weight_files) == 0

    def test_overwrite_existing_weights(self, tmp_path: Path):
        """Saving twice to the same path overwrites cleanly."""
        mgr = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr)
        import torch

        weight_dir = tmp_path / "weights"
        mgr.save_weights(weight_dir)

        # Mutate and save again
        role = list(mgr._torch_nn_registry.roles())[0]
        model = mgr._torch_nn_registry.get(role)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(-42.0)
        mgr.save_weights(weight_dir)

        # Load and verify latest save
        mgr2 = _make_brain_mgr("torch_nn")
        _init_torch_nn(mgr2)
        mgr2.load_weights(weight_dir)
        model2 = mgr2._torch_nn_registry.get(role)
        for p in model2.parameters():
            np.testing.assert_array_equal(p.detach().cpu().numpy(), -42.0)
