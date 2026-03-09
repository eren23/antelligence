# Roadmap — GPU Training & Beyond

## Phase 1: Colab Imitation Learning (immediate next step)

```bash
# On Colab (T4 GPU, free tier sufficient)
!pip install numpy pyyaml
# Upload: brains/, agents/, world/, config.py, train_imitation.py
!python train_imitation.py --brain nn --demo-ticks 20000 --epochs 200 --demo-ants 500
!python train_imitation.py --brain transformer --demo-ticks 20000 --epochs 200 --demo-ants 500
```

- 20k ticks x 500 ants = ~10M demo samples (vs ~1M locally)
- 200 epochs with larger batches (256) — GPU handles matrix ops
- Expected: NN matches rule-based; transformer approaches it

## Phase 2: Colab PPO Fine-Tuning (requires Phase 1 weights)

```bash
!python train_ppo.py --brain nn --load-imitation weights/imitation/ --ticks 500000 --lr 1e-4
```

- 500k ticks (10x local) — enough for PPO to converge
- Progressive patch removal: train 100k ticks, disable `auto_pickup`, train 100k more, etc.
- Expected: NN exceeds rule-based after patch removal

## Phase 3: Analytical Transformer Backprop (longer-term)

- Replace zeroth-order gradient with analytical backprop through attention layers
- Requires implementing attention backward pass (Q/K/V gradients, softmax Jacobian)
- Would make transformer training viable without GPU — currently zeroth-order is the bottleneck

## Phase 4: Multi-Seed Validation

- Run `compare_brains.py` with 10 seeds x 50k ticks per seed
- Statistical comparison: mean +/- std of food collected
- Publish results table in README

## Colab Notebook Structure (future `notebooks/train_colab.ipynb`)

1. Clone repo / upload files
2. Collect demonstrations (rule-based, 20k ticks)
3. Train imitation (NN + transformer, 200 epochs)
4. Validate imitation (headless 5k tick run)
5. PPO fine-tune (NN, 500k ticks)
6. Compare all brains (3 seeds x 5k ticks)
7. Download trained weights
