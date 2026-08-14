# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`Measurement.get_constrained_deconvolution()`.** Deconvolution under
  boundary conditions on area ratios and peak timings. Each component is
  parameterised by the heat it contributes inside the fit window instead of by
  its amplitude, so `area_fraction_bounds` acts on areas; the existing
  `relative_intensity_upper_bounds` of `get_deconvolution` bounds amplitude
  ratios, which differ for asymmetric components of unequal width.
  `peak_time_bounds` prescribes an interval for each peak position,
  `peak_time_delta_bounds` an interval for the offset of each peak from a
  `reference_component` (component 1 by default), `peak_width_bounds` an
  interval for each peak width, and `total_heat_bounds` an interval for the sum
  of the component areas. Offsets are useful across a series whose main peak
  shifts, where the spacing between components is known better than their
  absolute position; the fitted offset is reported in `peak_time_delta_s`. Width bounds are given in the
  unit of the width itself, seconds for `fraser_suzuki` and `gaussian` and the
  dimensionless log-time width for `lognormal`; a `None` entry falls back to the
  global `width_bounds`.

  **`PeakConstraints` and `DeconvolutionConstraints`.** The boundary conditions
  can be stated as one object per component instead of as four index-aligned
  lists. The component count and the reference component follow from the list of
  peaks, and `time_unit` is declared once so times need no conversion factor.
  Pass one as `constraints=` for all samples, or per sample in `sample_specs`.

  ```python
  from calocem import DeconvolutionConstraints, PeakConstraints

  four_peaks = DeconvolutionConstraints(
      time_unit="h",
      peaks=[
          PeakConstraints(time=(6, 18), area=(0.55, 0.95), width=(3, 20)),
          PeakConstraints(delta=(2, 12), area=(0.01, 0.20), width=(1, 10)),
          PeakConstraints(delta=(8, 25), area=(0.01, 0.20), width=(1, 20)),
          PeakConstraints(delta=(15, 45), area=(0.01, 0.20), width=(1, 30)),
      ],
  )
  deconv = m.get_constrained_deconvolution(processparams, constraints=four_peaks)
  ```

  `sample_specs` selects which files are fitted and gives each one its own
  settings, keyed by `sample_short`, as either a `DeconvolutionConstraints` or a
  dict of arguments. Anything a sample does not state falls back
  to the arguments of the call. A name matching no sample raises, as does an
  unknown setting name. A per-component list given at call level is dropped with
  a warning for a sample that asks for a different `n_peaks`, since its length
  no longer fits.

  ```python
  deconv = m.get_constrained_deconvolution(
      processparams,
      target_col="normalized_heat_flow_w_g_baseline_corrected",
      sample_specs={
          "sample_a": {"n_peaks": 4, "area_fraction_bounds": [...]},
          "sample_b": {"n_peaks": 3},
      },
  )
  ```

  The default peak shape is the Fraser-Suzuki function, whose width and
  asymmetry are separate parameters, alongside `lognormal` and `gaussian`.
  `shared_asymmetry` fits one asymmetry for all components, which stabilises
  the weak ones where the descending flank carries little structure. The fit is
  run from several starting points; `optimum_hit_fraction` reports the share of
  converged starts that reached the reported optimum again and is the check for
  whether the boundary conditions determine a unique solution.

  ```python
  m.get_baseline_corrected_data(processparams, inplace=True)
  deconv = m.get_constrained_deconvolution(
      processparams,
      target_col="normalized_heat_flow_w_g_baseline_corrected",
      n_peaks=3,
      peak_time_bounds=[(6 * 3600, 18 * 3600), (12 * 3600, 30 * 3600), (25 * 3600, 60 * 3600)],
      area_fraction_bounds=[(0.55, 0.90), (0.05, 0.30), (0.02, 0.20)],
  )
  ```

- **`Measurement.get_baseline()` and `Measurement.get_baseline_corrected_data()`.**
  A simple linear baseline for the heat-flow curve, fitted as the straight line
  through two anchor points. By default the first anchor is the heat-flow
  minimum of the dormant period and the second the last point of the curve; both
  can be placed explicitly. The heat flow at an anchor is averaged over
  `ProcessingParameters.baseline.window_s` to suppress noise. `get_baseline`
  returns slope, intercept and both anchors per sample and leaves the
  measurement unchanged; `get_baseline_corrected_data` returns a copy of the
  data with an additional `<target_col>_baseline_corrected` column.

  ```python
  baseline = m.get_baseline(processparams, show_plot=True)
  corrected = m.get_baseline_corrected_data(processparams, anchor_start_s=8 * 3600)
  ```

  With `inplace=True` the corrected column is attached to the measurement, so
  that it can be used by any analysis accepting `target_col`. In particular it
  allows the baseline to be fixed before a deconvolution instead of being fitted
  along with the peaks:

  ```python
  m.get_baseline_corrected_data(processparams, inplace=True)
  deconv = m.get_deconvolution(
      processparams,
      target_col="normalized_heat_flow_w_g_baseline_corrected",
      baseline_mode="none",
  )
  ```

## [0.3.4] - 2026-05-28

### Added

- **`Measurement.save(path)` and `Measurement.load(path)`.** Persist a processed
  `Measurement` to a single, user-named pickle file and restore it in a
  downstream script. The whole object is stored, so processed (e.g. downsampled)
  data, info, added metadata and `processparams` are all preserved. This is a
  convenience cache, not an archive — the raw source files remain the source of
  truth, and a file that fails to load can be regenerated by re-running from the
  original folder. The `.pkl` extension is recommended to keep the format
  explicit.

  ```python
  m = Measurement(folder, processparams=processparams)
  m.save("run01.pkl")

  # later, in another script
  m = Measurement.load("run01.pkl")
  ```

- **`Measurement.get_data_with_metadata()`.** Returns the measurement data
  joined with its added metadata as one tidy DataFrame, ready to export with
  `.to_csv(...)`. Works on per-sample data (joining on the metadata id column)
  and after `average_by_metadata` (joining on the group label and attaching
  only metadata that is constant within each group). Returns the data unchanged
  when no metadata has been added.

### Changed

- **`DownSamplingParameters.smoothing_factor` is now scale-relative.** It is
  scaled internally by the number of points and the heat-flow variance instead
  of being passed to the spline as an absolute residual target, so the same
  value behaves consistently regardless of heat-flow magnitude. The default
  changed from `1e-10` to `1e-6`. This changes which points adaptive
  downsampling selects.

### Fixed

- **Adaptive downsampling works again.** `Measurement(..., processparams=pp)`
  with `pp.downsample.apply = True` had silently returned the full-resolution
  data since the package refactor, because `_apply_adaptive_downsampling` was
  left as a no-op stub. It now downsamples per sample as documented.
- **`UnivariateSpline` "s too small" warning removed.** The spline fit during
  downsampling no longer emits a non-convergence `UserWarning`.

## [0.3.3] - 2026-05-06

### Breaking Changes

- **Pickle caching is now opt-in.** `Measurement(folder=...)` no longer writes
  `_data.pickle` and `_info.pickle` to the working directory by default. Pass
  `save_cache=True` explicitly to enable it.

  Migrate:

  ```python
  # old (implicit caching)
  m = Measurement(folder)

  # new — equivalent caching behavior
  m = Measurement(folder, save_cache=True)
  ```

### Added

- **`save_cache: bool = False` kwarg on `Measurement.__init__`.** Controls whether
  pickle cache files are written when loading from a folder. The existing
  `cold_start=False` read path is unchanged.

### Documentation

- New tutorial notebook `docs/example_get_mainpeak_params.ipynb` walking through
  `get_mainpeak_params`, with notebooks executed at docs build time so outputs
  stay in sync with the code.
- Replaced stale `TAInstCalorimetry` references and corrected the package name
  and PyPI URL across `docs/index.md`, `docs/how-to-guide.md`,
  `docs/quantification.md`, and `docs/tian.md`.

### Internal

- Notebooks are now exercised under pytest via `nbmake` so doc examples cannot
  silently rot.
- `nbstripout` configured to keep notebook diffs clean.

## [0.3.2] - 2026-05-02

### Added

- CSV files with a trailing comments column (e.g. annotations like
  "experiment interrupted and restarted") are now parsed correctly.

## [0.3.0] - 2026-04-03

### Breaking Changes

- **`calocem.tacalorimetry` has been removed.** The legacy monolithic
  `Measurement` implementation (`tacalorimetry.py`, ~3 400 lines) has been
  retired. The module now exists as a stub that raises a `FutureWarning` on
  import and forwards to the refactored implementation. The old and new
  implementations are **not guaranteed to produce identical numerical results**.
  Users who require the previous behaviour should pin to `calocem<0.3.0`.

  Migrate your imports:

  ```python
  # old
  from calocem.tacalorimetry import Measurement

  # new — preferred
  from calocem import Measurement

  # new — explicit
  from calocem.measurement import Measurement
  ```

### Added

- **Public package API.** `Measurement` and `ProcessingParameters` are now
  importable directly from the top-level package:
  ```python
  from calocem import Measurement, ProcessingParameters
  ```

### Fixed

- Eliminated all pandas Copy-on-Write `FutureWarning` and `DeprecationWarning`
  occurrences triggered by chained assignment patterns in `utils.py` and
  `tacalorimetry.py`. The package is now silent on pandas 2.x and compatible
  with the Copy-on-Write behaviour that becomes the default in pandas 3.0.
- `float()` called on a single-element `Series` replaced with
  `float(series.iloc[0])` to silence pandas `DeprecationWarning`.
- Removed stray `print("hallo")` debug statement from the data loading path.

## [0.2.5] - 2026-04-03

### Fixed
- `ipykernel` moved to dev dependencies — it is no longer installed as part of
  a normal `pip install calocem`. Users who need it for notebooks should install
  it explicitly (`pip install ipykernel`).
- Replaced `logging.basicConfig()` call at import time with a module-level
  logger. CaloCem no longer writes `CaloCem.log` to the working directory on
  import or hijacks the host application's log configuration.

### Internal
- `/.cache` added to `.gitignore` to stop the mkdocs-git-committers cache from
  being tracked.
- Manual exploration scripts (`script_test_*.py`) moved from `tests/` to
  `examples/` where the other example scripts live. They are not pytest tests
  and were silently collected without asserting anything.
- Removed unused live `import pysnooper` from `tests/test_read_calo_data_xls.py`.

## [0.2.4] - 2025-09-28

- See repository history for changes prior to this changelog.
