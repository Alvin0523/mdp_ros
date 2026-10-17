"""yolo_detector: which model file and device - CoreML twin on a Mac only, CUDA
untouched elsewhere."""
import os

from mdp_vision import yolo_detector as yd


def test_coreml_twin_only_on_mac(tmp_path, monkeypatch):
    pt = tmp_path / 'best_v5.pt'
    pt.write_bytes(b'')
    (tmp_path / 'best_v5.mlpackage').mkdir()
    monkeypatch.setattr(yd.sys, 'platform', 'darwin')
    assert yd.coreml_twin(str(pt)) == str(tmp_path / 'best_v5.mlpackage')
    monkeypatch.setattr(yd.sys, 'platform', 'linux')       # the NVIDIA laptop: the .pt (CUDA)
    assert yd.coreml_twin(str(pt)) == str(pt)


def test_no_twin_keeps_pt(tmp_path, monkeypatch):
    pt = tmp_path / 'best_v5.pt'
    pt.write_bytes(b'')
    monkeypatch.setattr(yd.sys, 'platform', 'darwin')
    assert yd.coreml_twin(str(pt)) == str(pt)               # not exported: MPS on the .pt


def test_coreml_picks_its_own_device():
    device, where = yd.pick_device(os.path.join('m', 'best_v5.mlpackage'))
    assert device is None and 'CoreML' in where
