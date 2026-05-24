import pathlib

from calocem.measurement import Measurement


DATA_DIR = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
METADATA = pathlib.Path(__file__).parent / "mini_metadata.csv"


def _build(regex):
    m = Measurement(DATA_DIR, regex=regex, show_info=False)
    m.add_metadata_source(METADATA, "experiment_nr", show_info=False)
    return m


def test_no_metadata_returns_data_unchanged():
    m = Measurement(DATA_DIR, regex=r"calorimetry_data_1\.csv", show_info=False)
    combined = m.get_data_with_metadata()
    assert combined.equals(m.get_data())


def test_per_sample_join_attaches_metadata():
    m = _build(r"calorimetry_data_[12]\.csv")
    combined = m.get_data_with_metadata()

    # every data row is preserved and annotated
    assert len(combined) == len(m.get_data())
    assert "cement_name" in combined.columns
    assert "water_amount_g" in combined.columns
    # the redundant metadata id column is dropped
    assert "experiment_nr" not in combined.columns

    sample_1 = combined[combined["sample_short"] == "calorimetry_data_1"]
    assert (sample_1["cement_name"] == "cem_a").all()
    assert (sample_1["water_amount_g"] == 50).all()


def test_averaged_join_keeps_only_group_constant_metadata():
    m = _build(r"calorimetry_data_[123]\.csv")
    m.average_by_metadata("cement_name")
    combined = m.get_data_with_metadata()

    # sample_short now holds the group label
    assert set(combined["sample_short"].unique()) == {"cem_a", "cem_b"}
    # the grouping column is constant within each group and is attached
    assert (combined["cement_name"] == combined["sample_short"]).all()
    # columns that vary within a group are dropped
    for col in ("experiment_nr", "water_amount_g", "date", "cement_amount_g"):
        assert col not in combined.columns


def test_averaged_join_multi_column_label():
    m = _build(r"calorimetry_data_[124]\.csv")
    m.average_by_metadata(["cement_name", "cement_amount_g"])
    combined = m.get_data_with_metadata()

    assert "cem_a | 100" in set(combined["sample_short"].unique())
    row = combined[combined["sample_short"] == "cem_a | 100"].iloc[0]
    assert row["cement_name"] == "cem_a"
    assert row["cement_amount_g"] == 100
    # varying-within-group columns are not attached
    assert "water_amount_g" not in combined.columns


def test_undo_average_restores_per_sample_join():
    m = _build(r"calorimetry_data_[123]\.csv")
    m.average_by_metadata("cement_name")
    m.undo_average_by_metadata()

    combined = m.get_data_with_metadata()
    # back to the per-sample join keyed by the original ids
    assert "calorimetry_data_1" in set(combined["sample_short"].unique())
    assert "cement_name" in combined.columns
    assert "experiment_nr" not in combined.columns


def test_averaged_join_handles_nan_in_group_column(tmp_path):
    # Regression: under pandas 3.0, astype(str) leaves NaN as a float, which
    # broke the " | ".join group-label reconstruction when a group column had
    # missing values. The label must be built by stringifying each value.
    meta = tmp_path / "meta_with_nan.csv"
    meta.write_text(
        "experiment_nr,polymer,dosage\n"
        "calorimetry_data_1,A,1\n"
        "calorimetry_data_2,,2\n"  # missing polymer -> NaN in a group column
        "calorimetry_data_3,A,1\n"
    )
    m = Measurement(DATA_DIR, regex=r"calorimetry_data_[123]\.csv", show_info=False)
    m.add_metadata_source(meta, "experiment_nr", show_info=False)
    m.average_by_metadata(["polymer", "dosage"])

    combined = m.get_data_with_metadata()  # must not raise

    assert "polymer" in combined.columns
    assert "A | 1" in set(combined["sample_short"].unique())
    assert (combined.loc[combined["sample_short"] == "A | 1", "polymer"] == "A").all()
