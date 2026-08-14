# %%
"""
Examples for processing the aluminate OPC calorimetry file.

The file ``calocem/DATA/aluminate_opc.csv`` is a more complex isothermal
calorimetry measurement of an ordinary Portland cement (OPC). Compared to the
simpler example files it contains two temperature channels
(``AmbientT(Therm3T)`` and ``Temperature``) and shows the characteristic
aluminate/sulfate-depletion feature in addition to the main silicate peak.

This script serves as a scratchpad to demonstrate the new methods we are
implementing to process this measurement. It starts from the currently
available functionality and will grow as new methods are added.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# %% load the aluminate OPC measurement

tam = Measurement(
    folder=datapath,
    regex=r"aluminate_opc.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

data = tam.get_data()
print(data.columns.tolist())
print(data[["time_s", "normalized_heat_flow_w_g"]].describe())

# %% basic plot of the normalized heat flow

ax = tam.plot(
    y="normalized_heat_flow_w_g",
    t_unit="h",
    y_unit_milli=True,
)
ax.set_xlim(0, 48)
plt.show()

# %% peak detection as a starting point

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 30
processparams.peakdetection.prominence = 1e-4
processparams.peakdetection.distance = 100

fig, ax = plt.subplots()
tam.get_peaks(
    processparams=processparams,
    ax=ax,
    show_plot=True,
    xunit="h",
    plot_labels=True,
    xmarker=True,
)
plt.show()

# %% simple linear baseline
#
# The measured curve sits on a slowly decaying background. get_baseline fits a
# straight line through two anchor points and returns its slope and intercept
# per sample. Without further arguments the first anchor is placed at the
# heat-flow minimum of the dormant period and the second at the last point of
# the curve; the heat flow at each anchor is averaged over
# processparams.baseline.window_s to suppress noise.

processparams.baseline.window_s = 1800

baseline = tam.get_baseline(processparams=processparams, show_plot=True, xunit="h")
plt.show()
print(baseline.to_string(index=False))

# %% baseline with explicitly placed anchors
#
# The automatic placement follows the usual definition of the dormant period,
# i.e. the heat-flow minimum preceding the highest peak. For this measurement
# the highest peak is the sharp aluminate peak at about 2.9 h, so the automatic
# anchor lands at roughly 0.5 h, and the resulting line cuts through the deeper
# minimum near 8 h. Placing the anchors explicitly is the remedy: the minimum
# between the aluminate peak and the broad silicate hump is the appropriate
# start, the last measured point the appropriate end.

baseline_manual = tam.get_baseline(
    processparams=processparams,
    anchor_start_s=8 * 3600,
    show_plot=True,
    xunit="h",
)
plt.show()
print(baseline_manual.to_string(index=False))

# %% working with the baseline-corrected curve
#
# get_baseline_corrected_data returns a copy of the data with an additional
# column; the measurement itself is left unchanged.

corrected = tam.get_baseline_corrected_data(
    processparams=processparams, anchor_start_s=8 * 3600
)
print(
    corrected[
        [
            "time_s",
            "normalized_heat_flow_w_g",
            "normalized_heat_flow_w_g_baseline_corrected",
        ]
    ].describe()
)

# %% minimal per-peak parameters via get_multipeak_params
#
# The OPC curve shows several hydration peaks (early, main silicate and the
# later aluminate/sulfate-depletion peak). get_multipeak_params returns one
# row per peak with its time, heat flow and mean ascending-flank slope.

multipeak = tam.get_multipeak_params(processparams=processparams)
multipeak["peak_time_h"] = multipeak["peak_time_s"] / 3600
print(
    multipeak[
        ["peak_nr", "peak_time_h", "peak_heat_flow_w_g", "mean_slope_w_g_s"]
    ].to_string(index=False)
)

# %% visual control of the per-peak slope detection
#
# show_plot=True draws one panel per detected peak with the flank region and
# fitted tangent, so the detection parameters can be checked visually.

tam.get_multipeak_params(processparams=processparams, show_plot=True)
plt.show()

# %% peak and shoulder detection

detections = tam.detect_peaks_and_shoulders(processparams=processparams)
detections["center_time_h"] = detections["center_time_s"] / 3600
print(
    detections[["feature_type", "center_time_h", "heat_flow"]].to_string(index=False)
)

# %% deconvolution seeded from detected peaks and shoulders
#
# The detected features (their number and positions) seed a lognormal
# deconvolution. show_plot=True overlays the fitted components and the seeds
# for visual control.

processparams.deconvolution.chebyshev_degree = 3
components = tam.get_multipeak_deconvolution(
    processparams=processparams,
    baseline_mode="chebyshev",  # Chebyshev polynomial baseline
    show_plot=True,
    log_y=False,  # log heat-flow axis makes the fit quality comparable across decades
)
plt.show()
print(components)

# %% new methods for the aluminate OPC file will be demonstrated below
