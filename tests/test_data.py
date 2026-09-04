import numpy as np
import pytest
from src.data import TARGET_FILES, case_files, window


def make_case(tmp_path, case_id="case_07", omit=None, extra=None):
    """Build a fake case folder of empty files, so the path logic can be
    tested on this laptop without the 4.9 GB dataset."""
    d = tmp_path / case_id
    d.mkdir()
    names = [f"{case_id}_IMG_CT.nrrd", f"{case_id}_IMG_MR_T1.nrrd"]
    names += [f"{case_id}_OAR_{n}.seg.nrrd" for n in TARGET_FILES]
    names += [f"{case_id}_OAR_Brainstem.seg.nrrd",
              f"{case_id}_OAR_A_Carotid_L.seg.nrrd"]
    for n in names:
        if omit and omit in n:
            continue
        (d / n).touch()
    if extra:
        (d / extra).touch()
    return str(d)


def test_case_files_finds_all_five(tmp_path):
    paths = case_files(make_case(tmp_path))
    assert set(paths) == {"CT", *TARGET_FILES}
    assert paths["CT"].endswith("_IMG_CT.nrrd")


def test_case_files_ignores_mr_and_other_organs(tmp_path):
    for p in case_files(make_case(tmp_path)).values():
        assert "MR_T1" not in p
        assert "Brainstem" not in p
        assert "Carotid" not in p


def test_missing_file_raises_loudly(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        case_files(make_case(tmp_path, omit="Cochlea_R"))
    assert "Cochlea_R" in str(e.value) and "case_07" in str(e.value)


def test_two_matches_also_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        case_files(make_case(tmp_path, extra="backup_case_07_IMG_CT.nrrd"))


def test_window_edges_and_middle():
    out = window(np.array([-1000.0, -160.0, 40.0, 240.0, 3000.0]), level=40, width=400)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == pytest.approx(0.5)
    assert out[3] == 1.0
    assert out[4] == 1.0