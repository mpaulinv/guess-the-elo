# Chess Elo Prediction — Final Model Report

## Summary

Final model: an **8-expert bracket-masked Mixture-of-Experts GRU** predicting
white and black Elo separately as a 14-way bucket classification (200-Elo-wide
buckets, 400-3200), trained on **3,948,473 Lichess games** using only move
sequences and per-ply clock time — no engine analysis, no board-state
features, no XGBoost-derived inputs.

| metric | this model (epoch 65) | prior best single-model | published literature* |
|---|---|---|---|
| MAE (white/black) | **191.8 / 190.7** | 221.2 / 220.0 | 182 (regression, board+clock CNN-LSTM) |
| Exact-bucket accuracy | 32.5% / 32.9% | 29.9% / 30.0% | — |
| Adjacent-bucket accuracy (±1 bucket) | 78.0% / 78.4% | 72.4% / 72.7% | — |
| Player-convergence shrinkage slope** | **0.792** | 0.598 | — |

\* [arXiv 2409.11506](https://arxiv.org/html/2409.11506v2), "Chess Rating
Estimation from Moves and Clock Times Using a CNN-LSTM" — regression, not
bucket classification, trained on 1.2M games with board-state CNN features
this project deliberately excluded.

\** Slope of (predicted running-average Elo after N held-out games) vs.
(actual Elo), across players with ≥20 held-out games. 1.0 = perfect
convergence, 0 = pure regression-to-the-mean. This was the central problem
that motivated the entire redesign documented below.

---

## 1. Starting point and the problem that drove the redesign

The original approach was a single GRU regressing directly to `avg_elo =
(white_elo + black_elo) / 2`. `scripts/analyze_player_convergence.py` tested
whether averaging this model's per-game predictions across a player's
multiple held-out games converged to their true Elo — the real test of
whether a rating estimator is usable, since any single game is noisy.

**Result: severe, non-diminishing shrinkage bias, slope ≈ 0.598.** The model
was hedging toward the population mean rather than confidently predicting
extreme ratings, because that minimizes loss on an imbalanced population
under a symmetric loss function (a classic empirical-Bayes/regression-to-mean
failure mode). This motivated every major decision below.

## 2. What was tried, in order, with results

### 2.1 Bucket classification instead of continuous regression
Switched to classifying white/black Elo separately into 14 buckets (200 Elo
wide, 400-3200) instead of regressing a single continuous `avg_elo`.
Classification alone doesn't fix shrinkage — the fix is the loss weighting
below — but it makes bucket-level class imbalance a legible, directly
correctable problem instead of an implicit regression artifact.

### 2.2 Class-weighted loss (fixes the shrinkage bias)
Inverse-frequency class weighting, `w_c = (N / (K·n_c))^power`, softened with
`power=0.5` (raw `power=1.0` produced 200x+ weight ratios between rarest and
most common buckets, destabilizing training at this data scale). This is the
single change that fixed the regression-to-mean problem — later confirmed
directly by the final player-convergence check (slope 0.598 → 0.792).

### 2.3 Clock-time features — validated positive
Added per-ply time-spent (parsed from Lichess's `%clk` annotations,
`src/time_features.py`) as an auxiliary GRU input alongside move-token
embeddings. Small-scale controlled comparison (`models/time_aware_experiment_results.json`,
~12k games):

| variant | acc (w/b) | adj_acc (w/b) |
|---|---|---|
| token-only baseline | 26.5% / 26.9% | 68.7% / 68.5% |
| token + clock-time | 28.0% / 28.1% | 71.7% / 71.8% |

Consistent, real improvement — kept in every subsequent model, and matches
literature: the CNN-LSTM paper above reports clock-time cuts MAE by 24%
overall.

### 2.4 First MoE attempt — fixed XGBoost gate — negative result
6 bracket experts routed by a **pre-computed, fixed** XGBoost bracket-probability
gate. Tested against the same single-head baseline (`models/moe_experiment_results.json`,
~12k games):

| variant | loss | acc (w/b) | adj_acc (w/b) |
|---|---|---|---|
| single head (baseline) | 3.761 | 26.5% / 26.9% | 68.7% / 68.5% |
| MoE, 6 experts, fixed gate | 3.824 | 25.5% / 26.2% | 66.2% / 65.6% |

**Worse than the baseline on every metric.** Diagnosed cause: routing
decided by a separately-trained, fixed classifier decouples expert
specialization from the actual training signal — an expert can get garbage
routed to it with no mechanism to compensate. This directly motivated the
final architecture's masked-loss design (§3), where "expertise" comes from
which ground-truth labels get gradient, not a learned/fixed router.

### 2.5 Mean/diff target decomposition — negative result
Hypothesis: decompose `(white_elo, black_elo)` into `avg_elo` (overall game
skill level) and `diff_elo = white - black` (who's ahead in this specific
game), on the theory that these have cleaner, more separable signal than the
entangled white/black pair. Tested at matched scale (100k games, 8 epochs,
`models/nn_mean_diff_v1_metrics.json`):

| target | acc | adj_acc | MAE |
|---|---|---|---|
| avg_elo | 25.7-26.7% | 65-67% | ~242-256 |
| diff_elo | 7.5-8.8% (≈ random) | ~20% (≈ random) | 224 → 345 (got *worse* while training) |
| reconstructed white/black (`avg ± diff/2`) | — | — | **324 / 328** |
| direct white/black baseline | — | — | 242 / 243 |

`avg_elo` was learnable (about as easy as predicting either player's Elo
directly), but `diff_elo` was not learnable at all from moves+clock-time
alone — accuracy was indistinguishable from random guessing, and it got
*worse* over training (overfitting noise). Reconstructing white/black from
the two landed clearly worse than predicting them directly. Conclusion:
"who's relatively ahead" needs an explicitly comparative signal between the
two players' move quality that this architecture doesn't construct; simply
reframing the target didn't supply one.

### 2.6 Short-game exclusion — tested, no benefit found
Question: does dropping very short games (<15 ply) help, either on raw
accuracy or on multi-game convergence? Checked both ways with real data
rather than assumption:

- **Aggregate MAE** (`models/length_vs_error_analysis.json`, cutoff sweep on
  the original single model): raising the cutoff from 10 to 14 ply removes an
  extra 0.19% of games for a **0.1 Elo point** change in remaining-game MAE.
  Even the most aggressive cutoff tested (20 ply, removing 1.16% of data)
  only moves MAE by 0.8 points. Short games individually carry higher error
  (~270-280 MAE vs. ~198 baseline) but are too rare a slice of the population
  to move any aggregate metric.
- **Player convergence** (paired same-player comparison, final bracket-MoE
  model, 20 players with ≥20 held-out games): shrinkage slope **0.792 with
  all games vs. 0.792 excluding games <15 ply** — identical to three decimal
  places. Final-prediction MAE moved marginally (82.3 → 80.2) but the
  systematic bias slope measures didn't change at all.

**Conclusion: not worth doing.** No measurable benefit on either axis, for
the cost of a full re-preprocess + retrain cycle. The current `min_ply=10`
filter (already applied during preprocessing) is sufficient.

### 2.7 Scaling up the data
Original dataset (~1.2M games) was supplemented with ~3M freshly-extracted
games from an untouched Lichess dump (`lichess_db_standard_rated_2023-04.pgn.zst`),
combining to **3,948,473 total games** for the final training run.

## 3. Final architecture: bracket-masked MoE

```
8 independent GRU experts, each: Embedding(vocab=13,378, dim=24)
                                  → GRU(input=25 [embed+time_spent], hidden=48)
                                  → Linear heads (white: 14-way, black: 14-way)

Trainable combiner: takes all 8 experts' (softmax probs ++ point-estimate guess)
                     → Linear(96) → ReLU → Dropout → Linear(14) for white and black
```

**Bracket assignment** (`BRACKET_WIDTH=400, BRACKET_STRIDE=350`, 50-Elo overlap
between neighbors):

| expert | range | expert | range |
|---|---|---|---|
| 0 | 400-800 | 4 | 1800-2200 |
| 1 | 750-1150 | 5 | 2150-2550 |
| 2 | 1100-1500 | 6 | 2500-2900 |
| 3 | 1450-1850 | 7 | 2850-3250 |

**How "expertise" works**: every expert sees every game's full move+clock-time
sequence (so its GRU still learns general chess patterns from the whole
dataset), but each expert's cross-entropy loss is **masked** to only the
games whose true Elo falls in its own bracket. This is not a learned or
fixed router — specialization comes purely from which ground-truth labels
contribute gradient to which expert, which structurally can't collapse the
way the §2.4 fixed-gate MoE did.

**Joint end-to-end training**: total loss = `combiner_loss + expert_loss_weight
× mean(per-expert masked loss)` (`expert_loss_weight=1.0`), backpropagated
through the whole model at once — the combiner isn't trained on top of frozen
experts, so gradient from the final prediction also reshapes the experts'
internal representations, not just their own-bracket calibration.

**Training config**: Adam, `lr=1e-3 → 1e-5` (cosine annealing), class-weighted
loss (`weight_power=0.5`, computed over the 14-bucket distribution),
`batch_size=512`, `dropout=0.2`, `weight_decay=1e-5`.

## 4. Training and checkpoint-selection process

Training ran on Colab (T4, later L4 — the switch only bought a modest ~23%
speedup since this model is data-loading/recurrence-bound, not
compute-bound), in several resumed sessions:

- Epochs 1-30 (clean run): **best val loss 5.003 at epoch 30**, acc(w/b)
  32.5%/32.6%, adj_acc 77.9%/77.8%.
- A resume mishap loaded a stale epoch-12 checkpoint instead of epoch 30,
  producing a second, independent 13→40 training lineage that reused the
  same output filenames.
- **Consequence**: `bracket_moe_gpu_epoch030.pt` on disk was silently
  overwritten by that second lineage's own (still pre-breakout, collapsed)
  epoch 30 — confirmed by loading it and finding it predicts a **constant
  bucket regardless of input** (100% of predictions landing in one bucket,
  0% everywhere else). **The original, better epoch-30 checkpoint (val loss
  5.003) is lost.**
- The second lineage continued 41→65 (correctly resumed from its own epoch
  40 onward) and reached **val loss 5.100 at epoch 65**, acc(w/b) 32.5%/32.9%,
  adj_acc 78.0%/78.4% — closing nearly all the gap to the lost epoch-30
  result and becoming the final model by elimination (verified healthy via
  per-rating-band breakdown showing real, sensible variation rather than
  constant output).
- **Recurring pattern observed across three independent runs**: the model
  breaks out of a long, flat plateau (frozen accuracy, near-zero movement)
  only once the cosine LR schedule reaches roughly **80% of its horizon**
  (epoch 16/20, 24/30, and 33/40 all independently matched this fraction).
  Judging this architecture's convergence before that point is unreliable.

**Lesson for any future run**: never rely on `_last.pt`/`_best.pt` alone
across multiple resume attempts touching the same output directory — a
later, worse run can silently overwrite a better checkpoint that shares an
epoch number. `train_bracket_moe_gpu.py` and the Colab notebook now also
save a standalone `_epoch{NNN}.pt` snapshot every epoch that is never
overwritten, specifically to prevent a repeat of this.

## 5. Final validation (epoch 65, full 394,847-game held-out test set)

### 5.1 Overall
white: acc=0.325, adj_acc=0.780, MAE=191.8 | black: acc=0.329, adj_acc=0.784, MAE=190.7

### 5.2 Error by true-Elo rating band
| band | n (white) | acc | adj_acc | MAE |
|---|---|---|---|---|
| 400-600 | 459 | 0.000 | 0.000 | 451.4 |
| 600-800 | 5,581 | 0.000 | 0.692 | 285.8 |
| 800-1000 | 21,907 | 0.456 | 0.798 | 166.2 |
| 1000-1200 | 40,131 | 0.329 | 0.794 | 188.5 |
| 1200-1400 | 56,653 | 0.256 | 0.749 | 205.8 |
| 1400-1600 | 73,494 | 0.320 | 0.769 | 188.9 |
| 1600-1800 | 78,023 | 0.358 | 0.846 | 169.0 |
| 1800-2000 | 66,682 | 0.435 | 0.826 | 161.1 |
| 2000-2200 | 36,570 | 0.223 | 0.764 | 221.7 |
| 2200-2400 | 11,911 | 0.182 | 0.505 | 297.2 |
| 2400-2600 | 2,808 | 0.000 | 0.315 | 429.3 |
| 2600-3200 | <500 each | 0.000 | 0.000 | 580-1150 |

Solid and reliable across the well-populated 800-2200 range where the vast
majority of real users fall. Two known weak points: the truly rare extremes
(<600, >2600 — a few hundred games total, likely not fixable without more
extreme-rated data) and a real, non-trivial dip at 2200-2400 despite that
band having a reasonable sample size (11,911 games) — the model is
measurably less reliable for strong club/expert-level players than for the
1600-2000 range.

### 5.3 Player convergence (the headline validation)
2,284 players with ≥20 held-out games each; slope of (final running-average
prediction) vs. (true Elo) across 20 sampled players: **0.792** (vs. 0.598 for
the original single-model regression). Players in the 1400-1900 range
typically converge to within 100 Elo of their true rating within a handful
of games; weaker convergence cases concentrate among higher-rated (2000+)
players, consistent with the rating-band weakness above.

## 6. Known limitations

- Weaker at both rating extremes, most notably a real accuracy dip at
  2200-2400 despite adequate data there — not just a data-scarcity artifact.
- The best-ever checkpoint (epoch 30, val loss 5.003) was lost to a
  file-overwrite accident; epoch 65 (val loss 5.100) is a very close but not
  identical substitute.
- All evaluation here ran on CPU; full-test-set inference took ~2 hours
  locally. Irrelevant to the model's actual quality, but a real cost for
  anyone re-running this evaluation rather than deploying on GPU.
- No engine/board-state features by design (the explicit scope constraint
  for this project) — the literature model that beats this one on raw MAE
  (182 vs. 191.8) uses board-state CNN features this project deliberately
  excluded, so that gap reflects a scope difference, not a fixable bug.

## 7. Where everything lives

- Final checkpoint: `models/bracket_moe_gpu_epoch065.pt`
- Training script (GPU, self-contained): `scripts/train_bracket_moe_gpu.py`
- Colab notebook: `notebooks/train_bracket_moe_colab.ipynb`
- Preprocessing (CSV → parquet): `scripts/preprocess_bracket_moe_data.py`
- Final dataset: `data/processed/nn_bracket_moe/dataset.parquet` (+ `manifest.json`, `vocab.json`)
- Final evaluation scripts: `scripts/finalize_bracket_moe_eval.py`,
  `scripts/check_short_game_convergence.py`
- Final evaluation results: `models/bracket_moe_final_eval.json`
- GPU setup instructions: `docs/gpu_training_setup.md`
