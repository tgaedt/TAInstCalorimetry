import pathlib

import numpy as np
import pytest

from calocem import (
    DeconvolutionConstraints,
    Measurement,
    PeakConstraints,
    ProcessingParameters,
)

HOUR = 3600.0


def _params() -> ProcessingParameters:
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 120
    return processparams


def _measurement(regex: str = r".*calorimetry_data_1\.csv$") -> Measurement:
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    return Measurement(
        path, regex=regex, auto_clean=False, show_info=False, cold_start=True
    )


def test_component_count_and_reference_follow_from_the_peaks():
    constraints = DeconvolutionConstraints(
        peaks=[
            PeakConstraints(delta=(-8 * HOUR, -1 * HOUR)),
            PeakConstraints(time=(8 * HOUR, 20 * HOUR), reference=True),
            PeakConstraints(delta=(5 * HOUR, 30 * HOUR)),
        ]
    )

    assert constraints.n_peaks == 3
    assert constraints.reference_component == 2
    assert constraints.to_kwargs()["n_peaks"] == 3
    assert constraints.to_kwargs()["reference_component"] == 2


def test_reference_defaults_to_the_first_component():
    constraints = DeconvolutionConstraints(
        peaks=[PeakConstraints(), PeakConstraints(delta=(1.0, 2.0))]
    )
    assert constraints.reference_component == 1


def test_time_unit_converts_times_but_not_areas():
    constraints = DeconvolutionConstraints(
        time_unit="h",
        peaks=[
            PeakConstraints(time=(6, 18), area=(0.55, 0.95), width=(3, 20)),
            PeakConstraints(delta=(2, 12), area=(0.01, 0.20), width=(1, 10)),
        ],
    )
    kwargs = constraints.to_kwargs()

    assert kwargs["peak_time_bounds"] == [(6 * HOUR, 18 * HOUR), None]
    assert kwargs["peak_time_delta_bounds"] == [None, (2 * HOUR, 12 * HOUR)]
    assert kwargs["peak_width_bounds"] == [(3 * HOUR, 20 * HOUR), (1 * HOUR, 10 * HOUR)]
    assert kwargs["area_fraction_bounds"] == [(0.55, 0.95), (0.01, 0.20)]


def test_lognormal_width_is_not_converted():
    """The lognormal width lives in log time and carries no unit."""
    constraints = DeconvolutionConstraints(
        time_unit="h",
        peak_shape="lognormal",
        peaks=[PeakConstraints(time=(6, 18), width=(0.2, 0.5))],
    )
    kwargs = constraints.to_kwargs()

    assert kwargs["peak_width_bounds"] == [(0.2, 0.5)]
    assert kwargs["peak_time_bounds"] == [(6 * HOUR, 18 * HOUR)]


def test_unconstrained_properties_become_none():
    constraints = DeconvolutionConstraints(
        peaks=[PeakConstraints(), PeakConstraints()]
    )
    kwargs = constraints.to_kwargs()

    assert kwargs["peak_time_bounds"] is None
    assert kwargs["peak_time_delta_bounds"] is None
    assert kwargs["peak_width_bounds"] is None
    assert kwargs["area_fraction_bounds"] is None


def test_partially_constrained_areas_leave_the_others_free():
    constraints = DeconvolutionConstraints(
        peaks=[PeakConstraints(area=(0.6, 0.9)), PeakConstraints()]
    )
    assert constraints.to_kwargs()["area_fraction_bounds"] == [(0.6, 0.9), (0.0, 1.0)]


def test_malformed_constraints_are_rejected():
    with pytest.raises(ValueError):
        DeconvolutionConstraints(peaks=[])

    with pytest.raises(ValueError):
        DeconvolutionConstraints(peaks=[PeakConstraints()], time_unit="fortnights")

    with pytest.raises(ValueError):
        DeconvolutionConstraints(
            peaks=[
                PeakConstraints(reference=True),
                PeakConstraints(reference=True),
            ]
        )

    with pytest.raises(ValueError):
        DeconvolutionConstraints(
            peaks=[PeakConstraints(delta=(1.0, 2.0), reference=True)]
        )


def test_constraints_reproduce_the_flat_argument_form():
    tam = _measurement()
    processparams = _params()

    flat = tam.get_constrained_deconvolution(
        processparams=processparams,
        n_peaks=3,
        peak_time_bounds=[(6 * HOUR, 18 * HOUR), None, None],
        peak_time_delta_bounds=[None, (4 * HOUR, 14 * HOUR), (10 * HOUR, 35 * HOUR)],
        area_fraction_bounds=[(0.65, 0.95), (0.02, 0.15), (0.02, 0.15)],
        peak_width_bounds=[(3 * HOUR, 20 * HOUR), (1 * HOUR, 8 * HOUR), None],
        n_starts=12,
    )

    structured = tam.get_constrained_deconvolution(
        processparams=processparams,
        constraints=DeconvolutionConstraints(
            time_unit="h",
            n_starts=12,
            peaks=[
                PeakConstraints(time=(6, 18), area=(0.65, 0.95), width=(3, 20)),
                PeakConstraints(delta=(4, 14), area=(0.02, 0.15), width=(1, 8)),
                PeakConstraints(delta=(10, 35), area=(0.02, 0.15)),
            ],
        ),
    )

    assert len(flat) == len(structured) == 3
    assert np.allclose(
        flat.sort_values("component")["component_area"].to_numpy(dtype=float),
        structured.sort_values("component")["component_area"].to_numpy(dtype=float),
    )


def test_sample_specs_accept_constraints_objects():
    tam = _measurement(regex=r".*calorimetry_data_[1-4]\.csv$")

    four = DeconvolutionConstraints(
        time_unit="h",
        peaks=[
            PeakConstraints(time=(6, 18), area=(0.55, 0.95), width=(3, 20)),
            PeakConstraints(delta=(2, 12), area=(0.01, 0.20), width=(1, 10)),
            PeakConstraints(delta=(8, 25), area=(0.01, 0.20), width=(1, 20)),
            PeakConstraints(delta=(15, 45), area=(0.01, 0.20), width=(1, 30)),
        ],
        n_starts=4,
    )
    two = DeconvolutionConstraints(
        time_unit="h",
        peaks=[
            PeakConstraints(time=(6, 18), area=(0.60, 0.98)),
            PeakConstraints(delta=(8, 25), area=(0.02, 0.40)),
        ],
        n_starts=4,
    )

    result = tam.get_constrained_deconvolution(
        processparams=_params(),
        sample_specs={"calorimetry_data_1": four, "calorimetry_data_3": two},
    )

    counts = result.groupby("sample_short").size()
    assert set(counts.index) == {"calorimetry_data_1", "calorimetry_data_3"}
    assert counts["calorimetry_data_1"] == 4
    assert counts["calorimetry_data_3"] == 2
