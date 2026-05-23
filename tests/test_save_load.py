import pathlib

import pytest

from calocem.exceptions import DataProcessingException
from calocem.measurement import Measurement
from calocem.processparams import ProcessingParameters


DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
SAMPLE = "excel_example4.xls"


def _build_measurement():
    processparams = ProcessingParameters()
    processparams.cutoff.cutoff_min = 42  # non-default, to check it round-trips
    return Measurement(
        DATA_DIR, regex=SAMPLE, show_info=False, processparams=processparams
    )


def test_save_writes_named_file(tmp_path):
    target = tmp_path / "run.pkl"
    _build_measurement().save(target)
    assert target.exists()


def test_round_trip_preserves_state(tmp_path):
    original = _build_measurement()
    target = tmp_path / "run.pkl"
    original.save(target)

    restored = Measurement.load(target)

    assert restored.get_data().equals(original.get_data())
    assert restored.get_information().equals(original.get_information())
    assert restored.processparams.cutoff.cutoff_min == 42


def test_load_invalid_file_raises(tmp_path):
    bogus = tmp_path / "bogus.pkl"
    bogus.write_bytes(b"not a pickle")

    with pytest.raises(DataProcessingException):
        Measurement.load(bogus)
