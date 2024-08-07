import pathlib

import pysnooper
import pytest

from CaloCem import tacalorimetry


#
# run "time check" on each sample file
# see https://www.youtube.com/watch?v=DhUpxWjOhME (~ at 12:20)
# to use the the test:
#       1) install the module locally with "pip install -e ."
#       2) run "pytest" from shell


# @pysnooper.snoop()
def test_get_maximum_slope():

    # path
    path = pathlib.Path().cwd() / "CaloCem" / "DATA"
    
    # init object
    tam = tacalorimetry.Measurement(path, auto_clean=False, show_info=True)
    
    tacalorimetry.plt.ylim(0, 0.01)

    processparams = tacalorimetry.ProcessingParameters() 
    # get cumulated heats
    max_slopes = tam.get_maximum_slope(processparams)
    
    # check
    assert isinstance(max_slopes, tacalorimetry.pd.DataFrame)
