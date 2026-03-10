"""Tests for weight persistence — save/load roundtrip for all brain backends.

Verifies:
- NN weight save/load roundtrip (weights match after reload)
- Transformer weight save/load roundtrip
- Torch NN weight save/load roundtrip (if torch available)
- Torch Transformer weight save/load roundtrip (if torch available)
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

def _make_brain_mgr(brain_type: str = "nn", seed: int = 42) -> BrainManager:
    """Create a BrainManager with the specified brain type."""
    cfg = default_config()
    cfg.brain.default = brain_type
    return BrainManager(cfg, seed)


def _init_nn(mgr: BrainManager) -> None:
    """Force-initialize the NN registry and trainers."""
    mgr._ensure_nn()


def _init_tf(mgr: BrainManager) -> None:
    """Force-initialize the transformer registry and trainers."""
    mgr._ensure_tf()


def _init_torch_nn(mgr: BrainManager) -> None:
    """Force-initialize the torch NN registry and trainers."""
    mgr._ensure_torch_nn()


def _init_torch_tf(mgr: BrainManager) -> None:
    """Force-initialize the torch transformer registry and trainers."""
    mgr._ensure_torch_tf()


# ---------------------------------------------------------------------------
# NN weight persistence
# ---------------------------------------------------------------------------

class TestNNWeightPersistence:
    """Save and reload NumPy MLP weights — arrays must match exactly."""

    def test_roundtrip_weights_match(self, tmp_path: Path):
        mgr = _make_brain_mgr("nn")
        _init_nn(mgr)

        # Snapshot original weights
        originals: dict[str, dict[str, np.ndarray]] = {}
        for role in mgr._nn_registry.roles():
            w = mgr._nn_registry.get(role)
            originals[role] = {
                "W1": w.W1.copy(), "b1": w.b1.copy(),
                "W2": w.W2.copy(), "b2": w.b2.copy(),
                "W_out": w.W_out.copy(), "b_out": w.b_out.copy(),
            }

        # Save
        mgr.save_weights(tmp_path / "weights")

        # Load into a fresh manager
        mgr2 = _make_brain_mgr("nn")
        _init_nn(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        # Verify
        for role in originals:
            w2 = mgr2._nn_registry.get(role)
            for name, arr in originals[role].items():
                loaded = getattr(w2, name)
                np.testing.assert_array_equal(
                    loaded, arr,
                    err_msg=f"NN {role} {name} mismatch after roundtrip",
                )

    def test_weight_files_created(self, tmp_path: Path):
        mgr = _make_brain_mgr("nn")
        _init_nn(mgr)
        mgr.save_weights(tmp_path / "weights")

        npz_files = list((tmp_path / "weights").glob("nn_*.npz"))
        assert len(npz_files) >= 4, f"Expected ≥4 NN weight files, got {len(npz_files)}"

    def test_modified_weights_persist(self, tmp_path: Path):
        """Mutate weights, save, reload — mutations survive."""
        mgr = _make_brain_mgr("nn")
        _init_nn(mgr)

        # Mutate a weight
        role = list(mgr._nn_registry.roles())[0]
        w = mgr._nn_registry.get(role)
        w.W1[:] = 99.0

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("nn")
        _init_nn(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        w2 = mgr2._nn_registry.get(role)
        np.testing.assert_array_equal(w2.W1, 99.0)


# ---------------------------------------------------------------------------
# Transformer weight persistence
# ---------------------------------------------------------------------------

class TestTransformerWeightPersistence:
    """Save and reload NumPy Transformer weights — arrays must match."""

    def test_roundtrip_weights_match(self, tmp_path: Path):
        mgr = _make_brain_mgr("transformer")
        _init_tf(mgr)

        # Snapshot original params
        originals: dict[str, list[np.ndarray]] = {}
        for role in mgr._tf_registry.roles():
            ws = mgr._tf_registry.get(role)
            originals[role] = [p.copy() for p in ws.all_parameters()]

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("transformer")
        _init_tf(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for role in originals:
            ws2 = mgr2._tf_registry.get(role)
            params2 = ws2.all_parameters()
            for i, (orig, loaded) in enumerate(zip(originals[role], params2)):
                np.testing.assert_array_equal(
                    loaded, orig,
                    err_msg=f"Transformer {role} param[{i}] mismatch",
                )

    def test_weight_files_created(self, tmp_path: Path):
        mgr = _make_brain_mgr("transformer")
        _init_tf(mgr)
        mgr.save_weights(tmp_path / "weights")

        npz_files = list((tmp_path / "weights").glob("transformer_*.npz"))
        assert len(npz_files) >= 4, f"Expected ≥4 transformer weight files, got {len(npz_files)}"


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

    def test_nn_trainer_state(self, tmp_path: Path):
        mgr = _make_brain_mgr("nn")
        _init_nn(mgr)

        # Mutate trainer state
        for role, trainer in mgr._nn_trainers.items():
            trainer.baseline = 0.123
            trainer.global_step = 999

        mgr.save_weights(tmp_path / "weights")

        # Verify JSON sidecar files exist
        json_files = list((tmp_path / "weights").glob("trainer_nn_*.json"))
        assert len(json_files) >= 4

        # Verify JSON content
        sample = json_files[0]
        with open(sample) as f:
            data = json.load(f)
        assert data["baseline"] == 0.123
        assert data["global_step"] == 999

        # Load into fresh manager and verify state restored
        mgr2 = _make_brain_mgr("nn")
        _init_nn(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for role, trainer in mgr2._nn_trainers.items():
            assert trainer.baseline == 0.123, f"Trainer {role} baseline not restored"
            assert trainer.global_step == 999, f"Trainer {role} global_step not restored"

    def test_transformer_trainer_state(self, tmp_path: Path):
        mgr = _make_brain_mgr("transformer")
        _init_tf(mgr)

        for role, trainer in mgr._tf_trainers.items():
            trainer.baseline = 0.456
            trainer.global_step = 2000

        mgr.save_weights(tmp_path / "weights")

        mgr2 = _make_brain_mgr("transformer")
        _init_tf(mgr2)
        mgr2.load_weights(tmp_path / "weights")

        for role, trainer in mgr2._tf_trainers.items():
            assert trainer.baseline == 0.456
            assert trainer.global_step == 2000

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
        mgr = _make_brain_mgr("nn")
        mgr.load_weights(tmp_path / "does_not_exist")

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "warning" in captured.out.lower()

    def test_load_empty_dir_no_crash(self, tmp_path: Path, capsys):
        """Loading from an empty directory prints a message, doesn't crash."""
        empty = tmp_path / "empty_weights"
        empty.mkdir()
        mgr = _make_brain_mgr("nn")
        mgr.load_weights(empty)

        captured = capsys.readouterr()
        assert "no weight files" in captured.out.lower()

    def test_save_creates_directory(self, tmp_path: Path):
        """Saving to a non-existent nested path creates the directory."""
        mgr = _make_brain_mgr("nn")
        _init_nn(mgr)

        nested = tmp_path / "a" / "b" / "c"
        mgr.save_weights(nested)
        assert nested.is_dir()
        assert len(list(nested.glob("nn_*.npz"))) >= 4

    def test_uninit_registries_no_files(self, tmp_path: Path):
        """Saving a fresh BrainManager with no initialized registries produces no files."""
        mgr = _make_brain_mgr("rule_based")
        mgr.save_weights(tmp_path / "weights")

        weight_files = list((tmp_path / "weights").glob("*.npz"))
        assert len(weight_files) == 0

    def test_overwrite_existing_weights(self, tmp_path: Path):
        """Saving twice to the same path overwrites cleanly."""
        mgr = _make_brain_mgr("nn")
        _init_nn(mgr)

        weight_dir = tmp_path / "weights"
        mgr.save_weights(weight_dir)

        # Mutate and save again
        role = list(mgr._nn_registry.roles())[0]
        w = mgr._nn_registry.get(role)
        w.W1[:] = -42.0
        mgr.save_weights(weight_dir)

        # Load and verify latest save
        mgr2 = _make_brain_mgr("nn")
        _init_nn(mgr2)
        mgr2.load_weights(weight_dir)
        w2 = mgr2._nn_registry.get(role)
        np.testing.assert_array_equal(w2.W1, -42.0)
