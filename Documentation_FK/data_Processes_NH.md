# Data workflow for SwissHourly dataset (loading → processing → sequencing → batching

This document describes the complete end-to-end workflow used by the `SwissHourly` custom dataset and the shared `BaseDataset` in your repository.

Use it as a reference to understand what happens, where, and when.

## Contract — inputs and outputs

- Inputs:
  - Configuration object `cfg` (instance of `Config`)
  - Per-basin CSV time series files in `cfg.data_dir`, named `<basin>.csv`. Each CSV must contain a datetime-like column (commonly `date`) and various feature columns.
- Outputs:
  - A PyTorch-style dataset that yields dictionary samples on indexing (via `__getitem__`).
  - `self.lookup_table` mapping indices to `(basin, per-frequency indices)`.

## Stage-by-stage mapping (functions and operations)

### 1) Initialization

- Key actions in `BaseDataset.__init__`:
  - Validate `period` (`train`, `validation`, `test`).
  - Call `_load_data()` to create normalized dataset and lookup table.

### 2) Load static attributes

- Function: `BaseDataset._load_combined_attributes()`
- Hook: `SwissHourly._load_attributes()`

### 3) Per-basin time series loading & preprocessing

- Main function: `BaseDataset._load_or_create_xarray_dataset()` — loops over `self.basins` and calls dataset-specific `_load_basin_data(basin)` for each basin.
- SwissHourly `_load_basin_data(basin)` steps:
  - Read CSV: `pd.read_csv(csv_path, parse_dates=["date"])`. Must contain `date` column (the code checks and raises if missing at this reading step).
  - Normalize index: call `_ensure_datetime_index(df)`:
  - Try `df.asfreq("h")` (Swiss hourly dataset), else fallback to `pd.infer_freq(df.index)` and set `asfreq(inferred)`. Logs a warning if frequency not enforced.
  - Check duplicates: If `df.index.duplicated().any()`, a `ValueError` is raised (keeps first occurrence is not implemented here; the code currently raises).
  - Required dynamic inputs:
  - Compute `keep_cols` — the union of:
    - `cfg.target_variables`, `cfg.evolving_attributes`, `cfg.mass_inputs`, `dynamic_cols_to_keep` (flattened dynamic inputs), and `cfg.dynamic_conceptual_inputs`.
  - Subset DataFrame: `df = df[keep_cols]`.
  - Ensure final index is `DatetimeIndex` (or `TimedeltaIndex`/`PeriodIndex`), else coerce; raise if not possible.
  - Return the cleaned DataFrame.

### 4) Add duplicates and lagged features (BaseDataset)

- `BaseDataset._duplicate_features(df)`:
  - Duplicates configured features (e.g., feature_copy1 ...) per `cfg.duplicate_features`.
- `BaseDataset._add_lagged_features(df)`:
  - Ensure `cfg.lagged_features` provide names and lags for features to be shifted.
  - For each `feature` and `shift` in that config: create `df[f"{feature}_shift{shift}"] = df[feature].shift(periods=shift, freq="infer")`.
  - `_check_autoregressive_inputs()` validates that names in `cfg.autoregressive_inputs` correspond to `<feature>_shift<lag>` that were created.
  - WARNING - Does not erase the original, non-lagged feature (Can't use it as replacement as a shift for the measurement value)

### 5) Convert to xarray and combine basins

- After each basin DataFrame is prepared, the code converts it to `xarray` and stores in a `data_list`.
- Finally, `xr = xarray.concat(data_list, dim="basin")` produces one `xarray.Dataset` with `basin` as a coordinate.

### 6) Optional: target normalization

- The code may apply `cfg.target_normalization` logic before saving/stacking datasets.

### 8) Compute normalization scalers (training only)

`BaseDataset._setup_normalization(xr)`:
When: Called in _load_data() when self._compute_scaler is True (training only).
Operations:

- Default: computes self.scaler["xarray_feature_center"] = xr.mean(skipna=True) and self.scaler["xarray_feature_scale"] = xr.std(skipna=True).
- Apply per-feature `cfg.custom_normalization` overrides (if provided).
- Save these scalers in `self.scaler` for later use (dumped to `train_data_scaler.yml` by `_dump_scaler()`).

### 9) Normalize the dataset

- Operation in `_load_data()` after scalers exist:
  - `xr = (xr - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"]`.
  - This yields normalized features across all basins and times.

### 10) Build lookup table & validate samples - Sequencing

- `BaseDataset._create_lookup_table(xr)`:
  - For each basin:
    - Extract dynamic inputs (x_d), static inputs per time (x_s) if exist, target y, and dates as numpy arrays.
    - Convert to lowest-frequency samples and create frequency maps: for multi-frequency setups, map each lowest-frequency sample to indices in higher-frequency arrays (frequency_maps).
    - Use `_validate_samples(x_d, x_s, y, seq_length, predict_last_n, frequency_maps)`:
      - This Numba function marks valid samples (1) vs invalid (0) based on:
        - Sequence length availability for each frequency.
        - NaN presence in required portions (targets/inputs).
        - AR inputs presence if `cfg.autoregressive_inputs` are required.
    - After validation, AR inputs are concatenated to the dynamic inputs array (so they are part of the dynamic vector at the end).
    - For each valid sample index, an entry is added to `lookup` as `(basin, indices)` where `indices` is the last index per frequency — these define windows to slice in `__getitem__`.
  - After processing all basins:
    - `self.lookup_table = {i: elem for i, elem in enumerate(lookup)}`
    - `self.num_samples = len(self.lookup_table)`

NOTE: `predict_last_n` indicates how many last steps are predicted vs. hindcasted and is used in validation.

### 11) Save train artifacts (training only)

- `_save_xarray_dataset(xr)` saves pickled xarray to `cfg.train_dir / "train_data.p"` if `cfg.save_train_data` True.
- `_dump_scaler()` writes `train_data_scaler.yml` containing scalers (pd.Series / xarray Dataset serialized as dict).

### 12) Runtime sampling`__getitem__`

- `BaseDataset.__getitem__(index)`:
  - Fetch `(basin, indices)` from `self.lookup_table[index]`.
  - For each frequency:
    - Compute slice boundaries: `hindcast_start_idx = idx + 1 - seq_len`, `global_end_idx = idx + 1`, and `forecast_start_idx = idx + 1 - cfg.forecast_seq_length`.
    - Fill `sample['x_d{suffix}']` dictionary: for each dynamic input feature `k` include the temporal window `self._x_d[basin][freq][k][hindcast_start_idx:global_end_idx]`.
    - Also build `sample['x_d{suffix}_hindcast']` and `sample['x_d{suffix}_forecast']` by selecting features listed in `cfg.hindcast_inputs_flattened` and `cfg.forecast_inputs_flattened` respectively.
    - `sample[f'y{suffix}']` gets the target window slice.
    - `sample[f'date{suffix}']` gets the date window slice.
    - Static inputs: concatenate `self._attributes[basin]` and `self._x_s[basin][freq][idx]` if present → `sample[f'x_s{suffix}']`.
    - If `cfg.timestep_counter`, also add `hindcast_counter` and `forecast_counter`.
  - Apply NaN-sequence augmentation when training via `_add_nan_streaks(sample_x_d, groups)` if `cfg.nan_step_probability` or `cfg.nan_sequence_probability`.
  - Add `per_basin_target_stds` and `x_one_hot` if available.
  - Return the `sample` dict.

**Important:**
The dataset returns both full dynamic windows and separate hindcast/forecast windows for features split by role — models that use these distinctions can index the right parts.

### 13) Batching: `collate_fn`

- Function: `BaseDataset.collate_fn(samples)`
- Action:
  - Stacks samples (lists of tensors/dicts) into single batched tensors with shape `(batch_size, seq_len, n_features)` for sequence features and `(batch_size, n_features)` for static inputs.
  - Handles nested dicts (e.g., `x_d` being a dict of feature tensors).
- Use:
  - Pass this function as `collate_fn` to `torch.utils.data.DataLoader` to produce batches ready for model input.

Output:

- A batch dict with keys like 'x_d', 'y', 'x_s', each with batched tensors shaped (batch, seq_length, features) or (batch, features) for static inputs.

---

## Key configuration knobs (where behavior changes)

- cfg.SH_addRollingFeatures: Controls automatic creation of rolling precipitation & temperature features in SwissHourly._load_basin_data.
- cfg.dynamic_inputs (list vs dict): If dict, features are per-frequency and sequence-building uses this structure (BaseDataset flattens appropriately).
- cfg.lagged_features: Names and lags of features to create with add_lagged_features.
- cfg.autoregressive_inputs: Names of AR features expected — must correspond to lagged feature names.
- cfg.seq_length, cfg.use_frequencies, cfg.predict_last_n: Control the window lengths, multi-frequency handling, and which part of the sequence is predicted.
- cfg.nan_step_probability, cfg.nan_sequence_probability: Data augmentation for NaN streaks in training.
- cfg.save_train_data: Whether to save the constructed xarray dataset to training directory.
- cfg.use_basin_id_encoding: Whether to create/use one-hot basin id embeddings.
