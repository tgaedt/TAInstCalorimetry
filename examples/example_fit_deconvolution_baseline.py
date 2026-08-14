
# %%
"""
Deconvolution with a baseline fixed before the fit.

The residual is a poor criterion on its own: a flexible baseline inside the fit
lowers it while absorbing an arbitrary share of the peak area. Subtracting a
fixed baseline beforehand costs residual and buys component areas that can be
compared between samples.
"""

from pathlib import Path

import numpy as np

from calocem import Measurement, ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

CORRECTED = "normalized_heat_flow_w_g_baseline_corrected"

tam = Measurement(
    folder=datapath,
    regex=r".*calorimetry_data_[1-4].csv$",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

pp = ProcessingParameters()
# Removes the wetting spike, which is two orders of magnitude higher than the
# hydration peaks and would dominate the least-squares residual. Above about
# 300 min the ascending flank of the silicate peak is truncated.
pp.cutoff.cutoff_min = 120

# %% reference: heat above the baseline
#
# The sum of the fitted component areas has to reproduce this number. It decides
# whether a fit is usable and is independent of the residual.

baseline = tam.get_baseline(processparams=pp, show_plot=True, xunit="h")
print(baseline[["sample_short", "baseline_slope_w_g_s", "baseline_intercept_w_g"]])

corrected = tam.get_baseline_corrected_data(processparams=pp, inplace=True)

measured_heat = {}
for sample, sample_data in corrected.groupby("sample_short"):
    sample_data = sample_data[sample_data["time_s"] >= pp.cutoff.cutoff_min * 60]
    sample_data = sample_data.sort_values("time_s")
    measured_heat[sample] = float(
        np.trapezoid(
            sample_data[CORRECTED].to_numpy(), sample_data["time_s"].to_numpy()
        )
    )


def report(label, deconv):
    print(f"\n{label}")
    for sample, sample_result in deconv.groupby("sample_short"):
        sample_result = sample_result.sort_values("peak_time_s")
        fitted_heat = float(sample_result["component_area"].sum())
        deviation = 100 * (fitted_heat / measured_heat[sample] - 1)
        times = ", ".join(f"{v / 3600:.1f}" for v in sample_result["peak_time_s"])
        print(
            f"  {sample:<20} rmse={sample_result['fit_rmse'].iloc[0]:.2e} "
            f"r2={sample_result['fit_r2'].iloc[0]:.4f} "
            f"heat={fitted_heat:6.1f} J/g ({deviation:+6.1f} %)  peaks at {times} h"
        )


# %% baseline inside the fit: good residual, unusable areas
#
# Raising the polynomial degree lowers the residual monotonically while the
# fitted heat drifts away from the measured heat. Degree 5 gives the best
# residual in this script and heats wrong by a factor of two.

for degree in [2, 3, 5]:
    pp.deconvolution.chebyshev_degree = degree
    report(
        f"chebyshev baseline in the fit, degree {degree}",
        tam.get_deconvolution(
            processparams=pp,
            n_peaks=3,
            peak_shape="lognormal",
            baseline_mode="chebyshev",
        ),
    )

# %% baseline subtracted beforehand
#
# inplace=True attaches the corrected column to the measurement, so the fit can
# use it via target_col. baseline_mode="none" leaves the components as the only
# degrees of freedom and the heat balance closes to within five percent.

deconv = tam.get_deconvolution(
    processparams=pp,
    target_col=CORRECTED,
    n_peaks=3,
    peak_shape="lognormal",
    baseline_mode="none",
    show_plot=True,
)
report("linear baseline subtracted beforehand", deconv)

# A free constant offset lowers the residual again, and the components grow to
# compensate it. A lower residual does not imply a better deconvolution.
report(
    "same, with a free constant offset",
    tam.get_deconvolution(
        processparams=pp,
        target_col=CORRECTED,
        n_peaks=3,
        peak_shape="lognormal",
        baseline_mode="constant",
    ),
)

# %% remaining parameters
#
# peak_shape "gaussian" is about five times worse in residual on these curves.
# Components that are not needed collapse onto the cutoff boundary with zero
# area, which is the signal to reduce n_peaks. The bounds keep a single broad
# component from swallowing the whole curve.

report(
    "two components with intensity and width bounds",
    tam.get_deconvolution(
        processparams=pp,
        target_col=CORRECTED,
        n_peaks=2,
        peak_shape="lognormal",
        baseline_mode="none",
        relative_intensity_upper_bounds=[0.95, 0.15],
        peak_width_upper_bounds=[0.6, 0.5],
    ),
)

# %% seeding from detected features
#
# get_multipeak_deconvolution seeds from detect_peaks_and_shoulders instead of a
# fixed component count. peakdetection.distance counts data points, not seconds,
# so a small value on a densely sampled file yields one component per noise
# feature: prominence=1e-4 with distance=100 gives 18 to 99 seeds here and the
# fit degenerates into interpolation at an excellent residual.

pp.peakdetection.prominence = 5e-4
pp.peakdetection.distance = 500
pp.deconvolution.savgol_window = 101

detections = tam.detect_peaks_and_shoulders(processparams=pp, target_col=CORRECTED)
print("\nseeds per sample")
print(detections.groupby("sample_short").size().to_string())

# %% downstream analysis on the corrected fit

left_peak_onset = tam.get_left_peak_inflection_tangent_intersection(
    processparams=pp,
    target_col=CORRECTED,
    deconvolution_results=deconv,
)

print(
    left_peak_onset[
        [
            "sample_short",
            "left_peak_center_time_s",
            "inflection_time_s",
            "x_intersection_abscissa_s",
        ]
    ]
)

# %%
