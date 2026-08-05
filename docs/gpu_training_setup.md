# GPU training setup — bracket-experts MoE

## What to upload

Everything needed is 3 files, ~638 MB total:

```
data/processed/nn_bracket_moe/dataset.parquet   (638 MB, 3,948,473 games)
data/processed/nn_bracket_moe/manifest.json     (vocab_size, bracket ranges, coverage)
data/processed/nn_bracket_moe/vocab.json         (token -> id map, not read by training but needed later for inference)
```

plus the single self-contained training script:

```
scripts/train_bracket_moe_gpu.py
```

No other file in the repo is needed — the script has no imports from the rest of this project.

## Colab

1. Zip the `nn_bracket_moe` folder locally, upload the zip to Google Drive.
2. In Colab: `Runtime > Change runtime type > T4/A100 GPU`.
3. Mount Drive, unzip, then:
   ```
   !pip install pyarrow -q   # usually already present in Colab
   !python train_bracket_moe_gpu.py --data-dir /content/drive/MyDrive/nn_bracket_moe --out-dir /content/drive/MyDrive/checkpoints --epochs 20
   ```
   Writing checkpoints straight to Drive means a session disconnect doesn't lose progress — resume with `--resume /content/drive/MyDrive/checkpoints/bracket_moe_gpu_last.pt`.

## Rented GPU box (RunPod / Lambda / vast.ai)

1. Spin up a single GPU instance (a T4 or 3090 is enough for this model size — it's small: embed_dim=24, hidden_dim=48, 8 GRUs).
2. `scp` the 3 data files + the script over.
3. `pip install torch pyarrow numpy`
4. `python train_bracket_moe_gpu.py --data-dir ./nn_bracket_moe --out-dir ./checkpoints --epochs 20`

## Before committing the full budget: calibrate first

I don't have a real throughput number for this exact model on a real GPU yet — only CPU timings from earlier smoke tests, which don't transfer. Rather than project an epoch time and be wrong again, run this first:

```
python train_bracket_moe_gpu.py --data-dir ./nn_bracket_moe --out-dir ./checkpoints --epochs 1
```

Watch the very first printed epoch line — it reports wall-clock seconds for that epoch directly (`(NNN.Ns)` in the log). Multiply by however many epochs you actually want (20 is the default and a reasonable starting point), add the instance's hourly rate, and that gives a real cost estimate instead of a guessed one. If epoch 1 alone would blow past the $100 budget, stop and either drop `--epochs`, use a smaller `--hidden-dim`, or switch to a cheaper GPU tier before continuing.

## Resuming / checkpoints

Every epoch overwrites `<model-tag>_last.pt` (safe to resume from after any interruption); the best validation-loss epoch is separately kept at `<model-tag>_best.pt` and is what gets used for the final test-set report at the end of the run.

## Flags worth knowing

- `--epochs` (default 20), `--batch-size` (default 512), `--embed-dim` (24), `--hidden-dim` (48) — the main cost/capacity knobs.
- `--weight-power` (default 0.5) — sqrt-softened inverse-frequency class weighting; this was the setting that fixed shrinkage bias in the single-model version, kept as the default here too.
- `--expert-loss-weight` (default 1.0) — relative weight of the 8 experts' masked bracket losses vs. the combiner's loss.
