# %%
import os
import sys

from TAInstCalorimetry import tacalorimetry

# get path of script and set it as current path
pathname = os.path.dirname(sys.argv[0])
os.chdir(pathname)
filename = os.path.basename(__file__).replace(".py", "")

os.sys.path.append(pathname + os.sep + os.pardir + os.sep + "src")

# import

# %% use class based approach

# define data path
path_to_data = pathname + os.sep + "DATA"

# experiments via class
tam = tacalorimetry.Measurement(
    folder=path_to_data,
    show_info=False
)

# get sample and information
data = tam.get_data()
info = tam.get_information()


# %% get samples
#

for sample, sample_data in tam.iter_samples():
    # print
    print(sample)


# %% basic plotting
#

tam.plot()
# show plot
tacalorimetry.plt.show()

# %% customized plotting

ax = tam.plot(
    t_unit="d",  # time axis in hours
    y_unit_milli=True,
    regex="1"  # regex expression for filtering
)

# set upper limits
ax.set_ylim(top=5)
ax.set_xlim(right=4)
# show plot
tacalorimetry.plt.show()


# %% get table of cumulated heat at certain age

cum_h = tam.get_cumulated_heat_at_hours(
    target_h=2,
    cutoff_min=0
)
print(cum_h)

# show cumulated heat plot
ax = tam.plot(
    t_unit="h",
    y='normalized_heat',
    y_unit_milli=False
)

# guide to the eye line
ax.axvline(2, color="gray", alpha=0.5, linestyle=":")           

# set upper limits
ax.set_ylim(top=250)
ax.set_xlim(right=12)
# show plot
tacalorimetry.plt.show()
