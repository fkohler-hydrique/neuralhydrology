
## Key points (summary)
- Initialization: `BaseDataset.__init__` sets up config, basins, frequency/sequence configuration and calls `_load_data()`.
- Attributes: `_load_combined_attributes()` calls `SwissHourly._load_attributes()` to load static features (TXT files) and hydroatlas attributes.
- Per-basin time series load: `SwissHourly._load_basin_data()` reads `<basin>.csv`, ensures a datetime index, optionally creates rolling features (`add_rolling_features`), checks/keeps required columns, and returns a cleaned DataFrame.
- Global dataset creation: `BaseDataset._load_or_create_xarray_dataset()` collects per-basin DataFrames, duplicates features, creates lagged features, converts to xarray and concatenates across basins.
- Normalization: `_setup_normalization()` computes per-feature center/scale (training), then xarray is normalized with those scalers.
- Validation & lookup: `_create_lookup_table()` converts xarray to per-basin numpy arrays, builds `frequency_maps`, validates samples with `_validate_samples()`, and creates `self.lookup_table` mapping integer index → (basin, per-frequency indices).
- Runtime sampling: `BaseDataset.__getitem__` uses `lookup_table` to slice windows (hindcast/forecast splits), apply NaN-streak augmentation (`_add_nan_streaks`), and return a sample dict.
- Batching: `BaseDataset.collate_fn` stacks samples into batch tensors for DataLoader.
- Config flags control behavior: rolling features, lagged features, multi-frequency settings, forecast/hindcast splitting, NaN augmentation, attribute selection, and scaler handling.

---
# Data workflow for SwissHourly dataset (loading → processing → sequencing → batching)

This document describes the complete end-to-end workflow used by the `SwissHourly` custom dataset and the shared `BaseDataset` in your repository. Use it as a reference to understand what happens, where, and when.

## Contract — inputs and outputs
- Inputs:
  - Per-basin CSV time series files in `cfg.data_dir`, named `<basin>.csv`. Each CSV must contain a datetime-like column (commonly `date`) and various feature columns.
  - Optional per-basin attribute TXT files in `<cfg.data_dir>/swiss_attributes_v1.0/` (semicolon-separated files with `gauge_id` column).
  - Configuration object `cfg` (instance of `Config`) with lists such as `dynamic_inputs`, `target_variables`, `static_attributes`, frequency/sequence settings, and flags such as `SH_addRollingFeatures`, `save_train_data`, etc.
  - Optional `additional_features` list of dicts and optional `scaler`, `id_to_int` for evaluation mode.
- Outputs:
  - A PyTorch-style dataset that yields dictionary samples on indexing (via `__getitem__`).
  - `self.lookup_table` mapping indices to `(basin, per-frequency indices)`.
  - Optional saved artifacts in `cfg.train_dir`: `train_data_scaler.yml` and `train_data.p` (pickled xarray dict).

## Stage-by-stage mapping (functions and operations)

### 1) Initialization
- Entry point:
  - `SwissHourly.__init__` → delegates to `BaseDataset.__init__`.
- Key actions in `BaseDataset.__init__`:
  - Validate `period` (`train`, `validation`, `test`).
  - Load basins list (single `basin` passed or read basin file via `utils.load_basin_file`).
  - Set flags: `_compute_scaler` (compute global scalers during training), `self._disable_pbar` (progress bar control).
  - Initialize frequency and sequence configuration via `_initialize_frequency_configuration()`.
  - Determine start/end dates for each basin via `_get_start_and_end_dates()`.
  - Load `additional_features` if configuration points to such files.
  - Create `id_to_int` if `cfg.use_basin_id_encoding` and training.
  - Call `_load_data()` to create normalized dataset and lookup table.

### 2) Load static attributes
- Function: `BaseDataset._load_combined_attributes()`
- Hook: `SwissHourly._load_attributes()`
- SwissHourly behavior (`SwissHourly._load_attributes()`):
  - Load all `swiss_*.txt` files under `<cfg.data_dir>/swiss_attributes_v1.0/`.
  - Each file is read with `sep=";"` and must contain `gauge_id`. Convert `gauge_id` to index.
  - Concatenate the attribute files horizontally, filter to `self.basins`, and strip column names.
- Further steps in `BaseDataset._load_combined_attributes()`:
  - Optionally append hydroatlas attributes via `_load_hydroatlas_attributes()`.
  - Keep only attributes listed in `cfg.static_attributes` and `cfg.hydroatlas_attributes`.
  - If `_compute_scaler` is True, compute attribute normalization and perform attribute sanity checks.
  - Convert attribute vectors per basin to torch tensors and store in `self._attributes[basin]`.

### 3) Per-basin time series loading & preprocessing
- Main function: `BaseDataset._load_or_create_xarray_dataset()` — loops over `self.basins` and calls dataset-specific `_load_basin_data(basin)` for each basin.
- SwissHourly `_load_basin_data(basin)` steps:
  - Path: `Path(cfg.data_dir) / f"{basin}.csv"`.
  - Read CSV: `pd.read_csv(csv_path, parse_dates=["date"])`. Must contain `date` column (the code checks and raises if missing at this reading step).
  - Normalize index: call `_ensure_datetime_index(df)`:
    - `_find_datetime_col(df)` picks `date` if present, else tries to find columns with keywords ("date", "time", "datetime"). If necessary, attempts to parse each column and accepts a column if >95% parses as datetime.
    - Converts that column to datetime, drops rows where conversion failed, sorts and sets as index.
    - Flattens MultiIndex if present by `df.reset_index()` first.
  - Frequency handling:
    - Try `df.asfreq("h")` (Swiss hourly dataset), else fallback to `pd.infer_freq(df.index)` and set `asfreq(inferred)`. Logs a warning if frequency not enforced.
  - Check duplicates: If `df.index.duplicated().any()`, a `ValueError` is raised (keeps first occurrence is not implemented here; the code currently raises).
  - Rolling features:
    - If `cfg.SH_addRollingFeatures` is True → call `add_rolling_features(df)`:
      - Adds precipitation rolling sums for specified precipitation columns: `<prec_col>_3h` and `<prec_col>_24h`.
      - Adds temperature-derived features `temp_binn_6h_mean`, `degree_day_24h`, `is_snow_bruchji` when `temp_binn` present.
  - Required dynamic inputs:
    - Determine `required_dyn` from `cfg.dynamic_inputs` (works for both list and dict per-frequency formats).
    - Build `rolling_derived` feature names that are generated by `add_rolling_features` and therefore do not need to exist in the CSV.
    - If any `required_dyn` feature is missing from df and is not in `rolling_derived`, raise `KeyError`.
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

### 5) Convert to xarray and combine basins
- After each basin DataFrame is prepared, the code converts it to `xarray` (omitted details in attachments) and stores in a `data_list`.
- Finally, `xr = xarray.concat(data_list, dim="basin")` produces one `xarray.Dataset` with `basin` as a coordinate.

### 6) Optional: target normalization (dataset-specific; omitted in excerpt)
- The code may apply `cfg.target_normalization` logic before saving/stacking datasets.

### 7) Per-basin target std (for NSE/weighted NSE)
- `BaseDataset._calculate_per_basin_std(xr)`:
  - For each basin, extract observed `cfg.target_variables` values.
  - Compute standard deviation ignoring NaNs.
  - Store in `self._per_basin_target_stds[basin]`.

### 8) Compute normalization scalers (training only)
- `BaseDataset._setup_normalization(xr)`:
  - Compute default center/scale: `xarray_feature_center = xr.mean(skipna=True)`, `xarray_feature_scale = xr.std(skipna=True)`.
  - Apply per-feature `cfg.custom_normalization` overrides (if provided).
  - Save these scalers in `self.scaler` for later use (dumped to `train_data_scaler.yml` by `_dump_scaler()`).

### 9) Normalize the dataset
- Operation in `_load_data()` after scalers exist:
  - `xr = (xr - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"]`.
  - This yields normalized features across all basins and times.

### 10) Build lookup table & validate samples
- `BaseDataset._create_lookup_table(xr)`:
  - For each basin:
    - Convert `xr.sel(basin=basin)` to pandas DataFrame then to numpy arrays for dynamic inputs, static per-time inputs, target `y`, and dates.
    - Build `frequency_maps` that map lowest-frequency samples to their positions in all used frequencies.
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

### 11) Save train artifacts (training only)
- `_save_xarray_dataset(xr)` saves pickled xarray to `cfg.train_dir / "train_data.p"` if `cfg.save_train_data` True.
- `_dump_scaler()` writes `train_data_scaler.yml` containing scalers (pd.Series / xarray Dataset serialized as dict).

### 12) Runtime sampling (`__getitem__`)
- `BaseDataset.__getitem__(index)`:
  - Fetch `(basin, indices)` from `self.lookup_table[index]`.
  - For each frequency:
    - Compute slice boundaries: `hindcast_start_idx = idx + 1 - seq_len`, `global_end_idx = idx + 1`, and forecast splits based on `cfg.forecast_seq_length` and `cfg.forecast_overlap`.
    - Fill `sample['x_d{suffix}']` dictionary: for each dynamic input feature `k` include the temporal window `self._x_d[basin][freq][k][hindcast_start_idx:global_end_idx]`.
    - Also build `sample['x_d{suffix}_hindcast']` and `sample['x_d{suffix}_forecast']` by selecting features listed in `cfg.hindcast_inputs_flattened` and `cfg.forecast_inputs_flattened` respectively.
    - `sample[f'y{suffix}']` gets the target window slice.
    - `sample[f'date{suffix}']` gets the date window slice.
    - Static inputs: concatenate `self._attributes[basin]` and `self._x_s[basin][freq][idx]` if present → `sample[f'x_s{suffix}']`.
    - If `cfg.timestep_counter`, also add `hindcast_counter` and `forecast_counter`.
  - Apply NaN-sequence augmentation when training via `_add_nan_streaks(sample_x_d, groups)` if `cfg.nan_step_probability` or `cfg.nan_sequence_probability`.
  - Add `per_basin_target_stds` and `x_one_hot` if available.
  - Return the `sample` dict.

### 13) Batching: `collate_fn`
- Function: `BaseDataset.collate_fn(samples)`
- Action:
  - Stacks samples (lists of tensors/dicts) into single batched tensors with shape `(batch_size, seq_len, n_features)` for sequence features and `(batch_size, n_features)` for static inputs.
  - Handles nested dicts (e.g., `x_d` being a dict of feature tensors).
- Use:
  - Pass this function as `collate_fn` to `torch.utils.data.DataLoader` to produce batches ready for model input.

## Edge cases & common errors
- Missing datetime column or parsing failures: `_find_datetime_col` may recover but an absent `date` at initial read will raise; parsing failures drop rows and issue a warning.
- Duplicate timestamps in CSV: triggers `ValueError` in `SwissHourly._load_basin_data`.
- Missing dynamic inputs (that are not rolling-derived) cause `KeyError`.
- Basins without valid samples across the time period will be listed; if none produce samples the dataset will error out.
- Zero-variance features may cause divide-by-zero in normalization → check `custom_normalization` or add safeguards.
- AR inputs must be present in `cfg.lagged_features` and referenced in `cfg.autoregressive_inputs` using the `_shift` naming convention.

## Debugging checklist
- Print `keep_cols` inside `SwissHourly._load_basin_data()` to see which columns will be retained.
- Inspect `dataset.num_samples` after instantiation to verify you have expected samples.
- Inspect `dataset.lookup_table[:5]` for early samples and `dataset._x_d[basin]` arrays to check shapes and NaNs.
- Open `cfg.train_dir / train_data_scaler.yml` to confirm saved scalers.
- If many samples are invalidated, verify:
  - `cfg.seq_length` and `cfg.predict_last_n` per frequency make sense relative to data length.
  - That lagged features exist or are being correctly created by `_add_lagged_features`.

---

## Small next steps I can do for you (pick any)
- Add a small debug script that instantiates the dataset for a single basin and prints `keep_cols`, head of `df` after `_ensure_datetime_index`, `num_samples`, and the first few `lookup_table` entries.
- Add safe-guards against zero-variance features in `_setup_normalization`.
- Add detailed logging in `SwissHourly._load_basin_data` for column detection and rolling-feature creation.

---

You can now edit and adapt this Markdown document locally. If you want, I can also write the debug script and run it in your environment (I will need the basin id or a sample CSV to test).