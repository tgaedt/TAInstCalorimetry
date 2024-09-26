#%%
from pathlib import Path
import matplotlib.pyplot as plt

import CaloCem.tacalorimetry as ta

parentfolder = Path(__file__).parent.parent
datapath = parentfolder / "CaloCem" / "DATA"
plotpath = parentfolder / "docs" / "assets"
testpath = parentfolder / "tests"

processparams = ta.ProcessingParameters()
processparams.downsample.apply = True
processparams.downsample.num_points = 300
processparams.downsample.section_split = True
processparams.downsample.section_split_time_s = 3600
processparams.downsample.baseline_weight = 0.1

# experiments via class
tam_d = ta.Measurement(
    folder=datapath,
    regex=r".*data_[1-2].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    processparams=processparams,
)

tam = ta.Measurement(
    folder=datapath,
    regex=r".*data_[1-2].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

#%%
fig, ax = plt.subplots()
for name, group in tam_d._data.groupby("sample_short"):
    ax.plot(group["time_s"]/3600, group["normalized_heat_flow_w_g"]*1000, "x-", label=name)
    
ax.set_xlim(0,24)
ax.set_ylim(0,25)
plt.show()

# %%
print(len(tam._data))
# %%

fig, ax = plt.subplots()
for (name, group), (name2, group2) in zip(tam._data.groupby("sample_short"), tam_d._data.groupby("sample_short")) :
    ax.plot(group["time_s"]/3600, group["normalized_heat_flow_w_g"]*1000, "-", label=name)
    ax.plot(group2["time_s"]/3600, group2["normalized_heat_flow_w_g"]*1000, "-", label=name2)
# ax.plot(tam._data["time_s"]/3600, tam._data["normalized_heat_flow_w_g"]*1000, "-", label="original")
# ax.plot(tam_d._data["time_s"]/3600, tam_d._data["normalized_heat_flow_w_g"]*1000, "-", label="downsampled")
ax.legend()
ax.set_xlim(0,10)
ax.set_ylim(0,4)

# %%