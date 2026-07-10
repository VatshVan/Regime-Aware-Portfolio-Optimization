# MLAM Scientific Audit Report: Stage 8-11 Leakage

## Executive Summary
A comprehensive scientific audit was performed on the financial machine learning pipeline, focusing on the implausibly high out-of-sample metrics reported (Primary Hit Rate ≈ 88%, Meta-Label Hit Rate ≈ 99.84%). 

**CONCLUSION: MASSIVE TARGET LEAKAGE DETECTED.**

The pipeline is mathematically invalid due to profound target leakage injected during Stage 9 (Primary Model Training). The reported 88% accuracy and 99.8% meta-accuracy are strictly artifacts of the model being trained on the future outcome it is attempting to predict.

---

## 🚨 Critical Vulnerability Identified: Stage 9 Target Leakage

### The Mechanism of Leakage
In `pipeline/stage9/training.py`, the master training dataset is built by merging the Triple Barrier Labels (`labels_df`) with the Features (`features_df`). 

The `labels_df` generated in Stage 4 contains the following columns used to compute the label:
- `ExitTime`, `ExitPrice`, `Return`, `HoldingPeriod`, `BarrierHit`

In `pipeline/stage9/pipeline.py`, the feature columns for the ML model are identified by excluding specific columns:
```python
exclude_cols = ['Datetime', 'EventTime', 'Ticker', 'Label', 'Label_Index', 'SampleWeight', 'bootstrapped_index']
potential_features = [c for c in master_df.columns if c not in exclude_cols]
```
**CRITICAL FLAW:** `ExitPrice`, `ExitTime`, `Return`, and `HoldingPeriod` were **not excluded**. 

Because `select_dtypes(include=[np.number, bool])` is subsequently called, the strings (`BarrierHit`, `ExitTime`) are dropped, but **`ExitPrice`, `Return`, and `HoldingPeriod` are passed directly into the XGBoost and LightGBM models as predictive features.**

### Impact
1. **Primary Model (Stage 9):** The gradient boosting models mathematically know the exact `Return` over the holding period at time $t$. Predicting the sign of the return when given the exact future return is trivial, resulting in an ~88% hit rate.
2. **Meta Model (Stage 10):** The Meta Model inherited the original master dataset's features, meaning it was *also* trained on `Return` and `ExitPrice`. Combined with the artificially inflated primary probabilities, it achieved 99.84% accuracy in predicting correct trades.

---

## 🟢 Passed Audits

Despite the critical flaw in Stage 9, the underlying engineering architecture is scientifically rigorous:

1. **Stage 1 (Feature Generation):** No centered windows or negative shifts detected in `features.py`. All indicators (MACD, ATR, Momentum) strictly use `shift(1)` and backward-rolling windows.
2. **Stage 4 (Triple Barrier):** Labeling occurs sequentially. Future path data does not corrupt the current price index.
3. **Stage 7 (Purged CV):** The Purged K-Fold overlap logic `(event_times <= test_end_date) & (exit_times >= test_start_date)` correctly purges the training set of any overlapping holding periods. Embargo is enforced strictly.
4. **Stage 8 (HMM):** The `GaussianHMM` is wrapped in an expanding window iterator. `predict_proba()` is strictly evaluated on the last sequence index using parameters fitted on the past. No future leakage.

## Final Verdict
The infrastructure (chunking, CV splits, Bootstrapping, HMM expanding windows) is structurally sound. However, the final metric evaluation is completely voided due to the missing exclusion list in Stage 9 dataset construction. 
