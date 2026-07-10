# Scientific Audit: Required Corrective Actions

## Immediate Fix Required: Stage 9 Target Leakage

The pipeline is technically excellent in structure but currently completely unscientific due to a single line in `pipeline/stage9/pipeline.py`.

### The Problem
During the joining of `labels_df` with `features_df`, the resulting `master_df` inherits all columns generated during Stage 4 Triple Barrier Labeling. These include:
- `ExitPrice`
- `ExitTime`
- `Return`
- `HoldingPeriod`
- `BarrierHit`
- `ProfitBarrier`
- `StopLossBarrier`

The feature selection logic merely excludes `['Datetime', 'EventTime', 'Ticker', 'Label', 'Label_Index', 'SampleWeight', 'bootstrapped_index']` and then filters for numeric types. This inadvertently passes exact future returns (`Return`) and exit prices into the ML models.

### Recommended Action
You must update `pipeline/stage9/pipeline.py` immediately to expand the exclusion list:

```python
exclude_cols = [
    'Datetime', 'EventTime', 'Ticker', 'Label', 'Label_Index', 
    'SampleWeight', 'bootstrapped_index', 
    'ExitPrice', 'ExitTime', 'Return', 'HoldingPeriod', 
    'BarrierHit', 'ProfitBarrier', 'StopLossBarrier',
    'VerticalBarrierTime', 'EventPrice', 'DailyVolatility'
]
```

*(Note: While `EventPrice` and `DailyVolatility` are known at `EventTime` and technically don't leak the future, they shouldn't be used as direct raw features unless explicitly intended as such. The actual predictive features should strictly come from the feature engineering pipeline).*

### Expected Impact
After fixing this leakage, the primary model accuracy will drop precipitously from 88% down to a mathematically plausible baseline (typically 50% - 55% for financial data). 

The Meta Model's accuracy will similarly drop from 99.8% to something realistic, forcing the Walk-Forward Validation to evaluate the true edge of the strategy rather than its ability to read the leaked `Return` column.

## Secondary Observations
1. **Purged K-Fold:** The cross-validation implementation is flawless. It mathematically removes overlapping holding periods and correctly embargoes post-test regions.
2. **Sequential Bootstrapping:** Properly limits itself to only drawing samples available inside the purged training split.
3. **HMM Regimes:** Correctly fitted on strictly expanding windows.

Fix the Stage 9 exclusion list, delete the existing predictions and meta-models, and re-run Stages 9, 10, and 11 to generate scientifically valid metrics.
