import pathlib

import numpy as np
import pandas as pd

from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


def _load_measurement() -> Measurement:
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    return Measurement(
        path,
        regex=r".*calorimetry_data_1\.csv$",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )


def test_get_baseline_uses_automatic_anchors():
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = _load_measurement()
    baseline = tam.get_baseline(processparams=processparams)

    assert isinstance(baseline, pd.DataFrame)
    assert not baseline.empty
    assert baseline["anchor_start_auto"].all()
    assert baseline["anchor_end_auto"].all()
    assert (baseline["anchor_start_s"] >= processparams.cutoff.cutoff_min * 60).all()
    assert (baseline["anchor_end_s"] > baseline["anchor_start_s"]).all()
    assert baseline["baseline_slope_w_g_s"].notna().all()
    assert baseline["baseline_intercept_w_g"].notna().all()


def test_get_baseline_passes_through_explicit_anchors():
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = _load_measurement()
    baseline = tam.get_baseline(
        processparams=processparams,
        anchor_start_s=5 * 3600,
        anchor_end_s=40 * 3600,
    )

    assert not baseline.empty
    assert (baseline["anchor_start_s"] == 5 * 3600).all()
    assert (baseline["anchor_end_s"] == 40 * 3600).all()
    assert not baseline["anchor_start_auto"].any()
    assert not baseline["anchor_end_auto"].any()

    # the fitted line must reproduce the anchor values
    predicted_start = (
        baseline["baseline_intercept_w_g"]
        + baseline["baseline_slope_w_g_s"] * baseline["anchor_start_s"]
    )
    predicted_end = (
        baseline["baseline_intercept_w_g"]
        + baseline["baseline_slope_w_g_s"] * baseline["anchor_end_s"]
    )
    assert np.allclose(predicted_start, baseline["anchor_start_w_g"])
    assert np.allclose(predicted_end, baseline["anchor_end_w_g"])


def test_get_baseline_corrected_data_subtracts_the_line():
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = _load_measurement()
    baseline = tam.get_baseline(processparams=processparams)
    corrected = tam.get_baseline_corrected_data(processparams=processparams)

    corrected_col = "normalized_heat_flow_w_g_baseline_corrected"
    assert corrected_col in corrected.columns
    assert len(corrected) == len(tam.get_data())

    row = baseline.iloc[0]
    sample = corrected[corrected["sample_short"] == row["sample_short"]]
    expected = sample["normalized_heat_flow_w_g"] - (
        row["baseline_intercept_w_g"] + row["baseline_slope_w_g_s"] * sample["time_s"]
    )
    assert np.allclose(sample[corrected_col], expected)

    # the original data are left untouched
    assert corrected_col not in tam.get_data().columns


def test_get_baseline_corrected_data_inplace_enables_downstream_use():
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = _load_measurement()
    corrected_col = "normalized_heat_flow_w_g_baseline_corrected"

    tam.get_baseline_corrected_data(processparams=processparams, inplace=True)
    assert corrected_col in tam.get_data().columns
    # the source column is only added to, never overwritten
    assert "normalized_heat_flow_w_g" in tam.get_data().columns

    deconv = tam.get_deconvolution(
        processparams=processparams,
        target_col=corrected_col,
        n_peaks=2,
        peak_shape="lognormal",
        baseline_mode="none",
        show_plot=False,
    )

    assert not deconv.empty
    assert (deconv["baseline_mode"] == "none").all()
    assert deconv["fit_r2"].notna().all()


def test_get_baseline_window_averages_the_anchor_value():
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 60

    tam = _load_measurement()

    processparams.baseline.window_s = 0
    single_point = tam.get_baseline(processparams=processparams, anchor_start_s=5 * 3600)

    processparams.baseline.window_s = 7200
    averaged = tam.get_baseline(processparams=processparams, anchor_start_s=5 * 3600)

    data = tam.get_data()
    window = data[
        (data["time_s"] >= 5 * 3600 - 3600) & (data["time_s"] <= 5 * 3600 + 3600)
    ]
    assert np.isclose(
        averaged["anchor_start_w_g"].iloc[0],
        window["normalized_heat_flow_w_g"].mean(),
    )
    assert not np.isclose(
        averaged["anchor_start_w_g"].iloc[0],
        single_point["anchor_start_w_g"].iloc[0],
    )
