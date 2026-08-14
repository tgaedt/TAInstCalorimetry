import pathlib

import numpy as np
import pandas as pd
import pytest

from calocem.analysis import ConstrainedDeconvolutionAnalyzer
from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters

HOUR = 3600.0


def _load_measurement() -> Measurement:
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    return Measurement(
        path,
        regex=r".*calorimetry_data_1\.csv$",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )


def _params() -> ProcessingParameters:
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 120
    return processparams


def test_area_fraction_bounds_are_respected():
    tam = _load_measurement()
    bounds = [(0.55, 0.90), (0.05, 0.30), (0.02, 0.20)]

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        area_fraction_bounds=bounds,
        n_starts=6,
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    for _, sample_result in result.groupby("sample_short"):
        sample_result = sample_result.sort_values("component")
        fractions = sample_result["component_area_fraction"].to_numpy(dtype=float)
        assert np.isclose(fractions.sum(), 1.0)
        for fraction, (low, high) in zip(fractions, bounds):
            assert low - 1e-6 <= fraction <= high + 1e-6


def test_peak_time_bounds_are_respected():
    tam = _load_measurement()
    bounds = [(6 * HOUR, 18 * HOUR), (18 * HOUR, 30 * HOUR), (30 * HOUR, 60 * HOUR)]

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        peak_time_bounds=bounds,
        n_starts=6,
    )

    assert not result.empty
    for _, sample_result in result.groupby("sample_short"):
        sample_result = sample_result.sort_values("component")
        centers = sample_result["center_time_s"].to_numpy(dtype=float)
        for center, (low, high) in zip(centers, bounds):
            assert low - 1e-6 <= center <= high + 1e-6
        assert np.all(np.diff(centers) > 0)


def test_peak_time_delta_bounds_are_respected():
    tam = _load_measurement()
    deltas = [None, (4 * HOUR, 14 * HOUR), (10 * HOUR, 35 * HOUR)]

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        peak_time_bounds=[(6 * HOUR, 18 * HOUR), None, None],
        peak_time_delta_bounds=deltas,
        n_starts=6,
    )

    assert not result.empty
    for _, sample_result in result.groupby("sample_short"):
        sample_result = sample_result.sort_values("component")
        centers = sample_result["center_time_s"].to_numpy(dtype=float)
        reported = sample_result["peak_time_delta_s"].to_numpy(dtype=float)

        # the reported delta must be the offset from the reference component
        assert np.allclose(reported, centers - centers[0])
        assert reported[0] == 0.0

        for delta, entry in zip(reported, deltas):
            if entry is None:
                continue
            assert entry[0] - 1e-6 <= delta <= entry[1] + 1e-6


def test_peak_time_delta_bounds_allow_a_component_before_the_reference():
    tam = _load_measurement()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        reference_component=2,
        peak_time_bounds=[None, (8 * HOUR, 20 * HOUR), None],
        peak_time_delta_bounds=[(-8 * HOUR, -1 * HOUR), None, (5 * HOUR, 30 * HOUR)],
        peak_width_bounds=[(1 * HOUR, 8 * HOUR), (3 * HOUR, 20 * HOUR), None],
        n_starts=6,
    )

    assert not result.empty
    for _, sample_result in result.groupby("sample_short"):
        sample_result = sample_result.sort_values("component")
        deltas = sample_result["peak_time_delta_s"].to_numpy(dtype=float)
        assert -8 * HOUR - 1e-6 <= deltas[0] <= -1 * HOUR + 1e-6
        assert deltas[1] == 0.0
        assert 5 * HOUR - 1e-6 <= deltas[2] <= 30 * HOUR + 1e-6


def test_inconsistent_delta_bounds_are_rejected():
    tam = _load_measurement()

    # a component after the reference cannot be required to precede it
    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(),
            n_peaks=2,
            peak_time_delta_bounds=[None, (-10 * HOUR, -1 * HOUR)],
        )

    # the reference cannot be offset from itself
    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(),
            n_peaks=2,
            peak_time_delta_bounds=[(0.0, 1 * HOUR), (1 * HOUR, 5 * HOUR)],
        )


def test_peak_width_bounds_are_respected():
    tam = _load_measurement()
    bounds = [(5 * HOUR, 15 * HOUR), (5 * HOUR, 25 * HOUR), (10 * HOUR, 40 * HOUR)]

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        peak_width_bounds=bounds,
        n_starts=6,
    )

    assert not result.empty
    for _, sample_result in result.groupby("sample_short"):
        sample_result = sample_result.sort_values("component")
        widths = sample_result["width"].to_numpy(dtype=float)
        for width, (low, high) in zip(widths, bounds):
            assert low - 1e-6 <= width <= high + 1e-6


def test_peak_width_bounds_accept_none_entries():
    """A None entry falls back to the global width_bounds."""
    tam = _load_measurement()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        peak_width_bounds=[(6 * HOUR, 10 * HOUR), None, None],
        n_starts=6,
    )

    assert not result.empty
    constrained = result[result["component"] == 1]["width"].to_numpy(dtype=float)
    assert np.all(constrained >= 6 * HOUR - 1e-6)
    assert np.all(constrained <= 10 * HOUR + 1e-6)


def test_lognormal_width_bounds_are_dimensionless():
    """The lognormal width lives in log time and must not be rescaled."""
    tam = _load_measurement()
    bounds = [(0.2, 0.5), (0.2, 0.8)]

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=2,
        peak_shape="lognormal",
        peak_width_bounds=bounds,
        n_starts=6,
    )

    assert not result.empty
    widths = result.sort_values("component")["width"].to_numpy(dtype=float)
    for width, (low, high) in zip(widths, bounds):
        assert low - 1e-6 <= width <= high + 1e-6


def test_total_heat_bounds_are_respected():
    tam = _load_measurement()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=2,
        total_heat_bounds=(100.0, 140.0),
        n_starts=6,
    )

    assert not result.empty
    for _, sample_result in result.groupby("sample_short"):
        total = float(sample_result["component_area"].sum())
        assert 100.0 - 1e-3 <= total <= 140.0 + 1e-3


@pytest.mark.parametrize("peak_shape", ["fraser_suzuki", "lognormal", "gaussian"])
def test_result_frame_reproduces_the_fitted_curve(peak_shape):
    """The reported shape parameters must rebuild the components exactly."""
    tam = _load_measurement()
    processparams = _params()

    result = tam.get_constrained_deconvolution(
        processparams=processparams,
        n_peaks=2,
        peak_shape=peak_shape,
        n_starts=4,
    )
    assert not result.empty

    data = tam.get_data()
    data = data[data["time_s"] >= processparams.cutoff.cutoff_min * 60]
    data = data.dropna(subset=["time_s", "normalized_heat_flow_w_g"]).sort_values(
        "time_s"
    )
    x = data["time_s"].to_numpy(dtype=float)
    y = data["normalized_heat_flow_w_g"].to_numpy(dtype=float)

    shape_fn = ConstrainedDeconvolutionAnalyzer._shape_function(peak_shape)
    model = np.zeros_like(x)
    for _, row in result.iterrows():
        curve = float(row["amplitude"]) * shape_fn(
            x,
            float(row["center_time_s"]),
            float(row["width"]),
            float(row["asymmetry"]),
        )
        model += curve
        assert np.isclose(
            float(np.trapezoid(curve, x)), float(row["component_area"]), rtol=1e-3
        )

    ss_res = float(np.sum((y - model) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    assert np.isclose(1 - ss_res / ss_tot, float(result["fit_r2"].iloc[0]), rtol=1e-6)


def test_fraser_suzuki_reduces_to_a_gaussian_for_vanishing_asymmetry():
    x = np.linspace(0.0, 10.0, 501)
    skewed = ConstrainedDeconvolutionAnalyzer._shape_fraser_suzuki(x, 5.0, 2.0, 0.0)
    expected = np.exp(-4.0 * np.log(2.0) * ((x - 5.0) / 2.0) ** 2)

    assert np.allclose(skewed, expected)
    # unit height at the position, and the half width is the width parameter
    assert np.isclose(skewed.max(), 1.0)
    assert np.isclose(
        ConstrainedDeconvolutionAnalyzer._shape_fraser_suzuki(
            np.array([6.0]), 5.0, 2.0, 0.0
        )[0],
        0.5,
    )


def _load_all_samples() -> Measurement:
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    return Measurement(
        path,
        regex=r".*calorimetry_data_[1-4]\.csv$",
        auto_clean=False,
        show_info=False,
        cold_start=True,
    )


def test_sample_specs_select_the_samples_and_their_settings():
    tam = _load_all_samples()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_starts=4,
        sample_specs={
            "calorimetry_data_1": {"n_peaks": 4},
            "calorimetry_data_3": {"n_peaks": 2},
        },
    )

    assert not result.empty
    assert set(result["sample_short"]) == {"calorimetry_data_1", "calorimetry_data_3"}
    per_sample = result.groupby("sample_short").size()
    assert per_sample["calorimetry_data_1"] == 4
    assert per_sample["calorimetry_data_3"] == 2
    assert (
        result[result["sample_short"] == "calorimetry_data_1"]["n_peaks_fitted"] == 4
    ).all()


def test_sample_specs_fall_back_to_the_call_level_settings():
    tam = _load_all_samples()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=2,
        peak_shape="gaussian",
        n_starts=4,
        sample_specs={
            "calorimetry_data_1": {},
            "calorimetry_data_2": {"peak_shape": "fraser_suzuki"},
        },
    )

    shapes = result.groupby("sample_short")["peak_shape"].first()
    assert shapes["calorimetry_data_1"] == "gaussian"
    assert shapes["calorimetry_data_2"] == "fraser_suzuki"
    assert (result.groupby("sample_short").size() == 2).all()


def test_inherited_per_component_bounds_are_dropped_when_n_peaks_changes():
    """A length-3 list from the call level cannot apply to a 4-component fit."""
    tam = _load_all_samples()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=3,
        peak_time_bounds=[(6 * HOUR, 18 * HOUR), None, None],
        n_starts=4,
        sample_specs={
            "calorimetry_data_2": {},
            "calorimetry_data_4": {"n_peaks": 4},
        },
    )

    assert not result.empty
    inherited = result[result["sample_short"] == "calorimetry_data_2"]
    assert len(inherited) == 3
    # the call-level bound still applies where n_peaks is unchanged
    first = inherited.sort_values("component")["center_time_s"].iloc[0]
    assert 6 * HOUR - 1e-6 <= first <= 18 * HOUR + 1e-6

    assert len(result[result["sample_short"] == "calorimetry_data_4"]) == 4


def test_sample_specs_reject_unknown_samples_and_settings():
    tam = _load_all_samples()

    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(), sample_specs={"not_a_sample": {}}
        )

    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(),
            sample_specs={"calorimetry_data_1": {"unknown_setting": 1}},
        )

    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(), sample_specs={"calorimetry_data_1": 5}
        )


def test_four_components_are_supported():
    tam = _load_measurement()

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        n_peaks=4,
        peak_time_bounds=[(6 * HOUR, 18 * HOUR), None, None, None],
        peak_time_delta_bounds=[
            None,
            (2 * HOUR, 12 * HOUR),
            (8 * HOUR, 25 * HOUR),
            (15 * HOUR, 45 * HOUR),
        ],
        area_fraction_bounds=[(0.55, 0.95), (0.01, 0.20), (0.01, 0.20), (0.01, 0.20)],
        n_starts=6,
    )

    assert len(result) == 4
    assert (result["n_peaks_fitted"] == 4).all()
    centers = result.sort_values("component")["center_time_s"].to_numpy(dtype=float)
    assert np.all(np.diff(centers) > 0)
    assert np.isclose(result["component_area_fraction"].sum(), 1.0)


def test_infeasible_boundary_conditions_are_rejected():
    tam = _load_measurement()

    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(),
            n_peaks=3,
            # upper limits cannot sum to less than one
            area_fraction_bounds=[(0.0, 0.2), (0.0, 0.2), (0.0, 0.2)],
        )

    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(),
            n_peaks=2,
            area_fraction_bounds=[(0.0, 1.0)],
        )

    with pytest.raises(Exception):
        tam.get_constrained_deconvolution(
            processparams=_params(),
            n_peaks=2,
            peak_width_bounds=[(10 * HOUR, 5 * HOUR), None],
        )
