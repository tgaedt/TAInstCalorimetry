
# %%
"""
Deconvolution of two to three asymmetric components under boundary conditions.

The components are parameterised by the heat they contribute inside the fit
window, so the boundary conditions act on area ratios rather than on
amplitudes. The Fraser-Suzuki shape separates width from asymmetry, which the
lognormal shape does not.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import (
    DeconvolutionConstraints,
    Measurement,
    PeakConstraints,
    ProcessingParameters,
)

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
H = 3600

CORRECTED = "normalized_heat_flow_w_g_baseline_corrected"

tam = Measurement(
    folder=datapath,
    regex=r".*calorimetry_data_[1-4].csv$",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

pp = ProcessingParameters()
pp.cutoff.cutoff_min = 120

# Fix the baseline first, so the components are the only free part of the model.
tam.get_baseline_corrected_data(processparams=pp, inplace=True)


def summary(label, deconv):
    print(f"\n{label}")
    for sample, result in deconv.groupby("sample_short"):
        result = result.sort_values("component")
        fractions = ", ".join(f"{v:.2f}" for v in result["component_area_fraction"])
        times = ", ".join(f"{v / H:.1f}" for v in result["peak_time_s"])
        widths = ", ".join(f"{v / H:.1f}" for v in result["width"])
        deltas = ", ".join(f"{v / H:+.1f}" for v in result["peak_time_delta_s"])
        print(
            f"  {sample:<20} R2={result['fit_r2'].iloc[0]:.4f} "
            f"hit={result['optimum_hit_fraction'].iloc[0]:.2f} "
            f"heat={result['component_area'].sum():5.1f} J/g "
            f"areas=[{fractions}] t_h=[{times}] dt_h=[{deltas}] w_h=[{widths}]"
        )





four_peaks = DeconvolutionConstraints(
    time_unit="h",
    peaks=[
        PeakConstraints(time=(6, 18), area=(0.55, 0.95), width=(3, 20)),
        PeakConstraints(delta=(0.5, 5), area=(0.05, 0.12), width=(0.5, 10)),
        PeakConstraints(delta=(4, 12), area=(0.05, 0.12), width=(1, 15)),
        PeakConstraints(delta=(5, 45), area=(0.01, 0.05), width=(5, 30)),
    ],
)

# summary(
#     "4 components",
#     tam.get_constrained_deconvolution(
#         processparams=pp, target_col=CORRECTED, constraints=four_peaks, show_plot=True
#     ),
# )


summary(
    "4 components",
    tam.get_constrained_deconvolution(
        processparams=pp, constraints=four_peaks, show_plot=True
    ),
)