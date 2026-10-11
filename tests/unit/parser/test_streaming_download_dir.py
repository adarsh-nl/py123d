"""Downloads stay inside the output directory; streaming conversions use one folder in the download root."""

import types
import zipfile
from pathlib import Path
from typing import List

import pytest
from hydra import compose, initialize_config_module

from py123d.parser.ncore.ncore_parser import NCoreLogParser
from py123d.parser.nureasoning import nureasoning_download
from py123d.parser.nureasoning.nureasoning_download import NureasoningDownloader
from py123d.parser.pandaset import pandaset_download
from py123d.parser.pandaset.pandaset_download import PandasetDownloader
from py123d.parser.pandaset.pandaset_parser import PandasetLogParser
from py123d.parser.truckdrive.truckdrive_download import TruckDriveDownloader
from py123d.script.config.conversion import dataset as dataset_configs

STREAM_CONFIGS = sorted(path.stem for path in Path(dataset_configs.__path__[0]).glob("*-stream.yaml"))


def _write_zip(path: Path, members: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, b"data")
    return path


@pytest.mark.parametrize("config_name", STREAM_CONFIGS)
def test_stream_config_downloads_into_one_folder(config_name: str, monkeypatch):
    """Every streaming config downloads into ``<py123d_download_root>/<dataset>``."""
    monkeypatch.setenv("PY123D_DATA_ROOT", "/data")
    monkeypatch.setenv("PY123D_DOWNLOAD_ROOT", "/scratch")
    with initialize_config_module(config_module="py123d.script.config.conversion", version_base=None):
        cfg = compose(config_name="default_conversion", overrides=[f"dataset={config_name}"])
    assert cfg.dataset_paths.py123d_download_root == "/scratch"
    output_dir = Path(cfg.dataset.parser.downloader.output_dir)
    assert output_dir.parent == Path("/scratch") and config_name.startswith(output_dir.name)


def test_download_root_defaults_to_data_root(monkeypatch):
    monkeypatch.setenv("PY123D_DATA_ROOT", "/data")
    monkeypatch.delenv("PY123D_DOWNLOAD_ROOT", raising=False)
    with initialize_config_module(config_module="py123d.script.config.conversion", version_base=None):
        cfg = compose(config_name="default_conversion", overrides=["dataset=nuplan-stream"])
    assert cfg.dataset.parser.downloader.output_dir == "/data/.download/nuplan"


class TestPerLogStreaming:
    """PandaSet and NCore download one log at a time and delete it once it is converted."""

    def test_pandaset_log_dir(self, tmp_path: Path):
        def download_single_log(log_name: str, output_dir: Path) -> Path:
            (Path(output_dir) / log_name).mkdir(parents=True)
            return Path(output_dir) / log_name

        downloader = types.SimpleNamespace(download_single_log=download_single_log, output_dir=tmp_path)
        log_parser = PandasetLogParser(None, "001", "pandaset_train", downloader=downloader)
        with log_parser._resolved_log() as log_path:
            assert log_path == tmp_path / "001" and log_path.is_dir()
        assert not (tmp_path / "001").exists()

    def test_ncore_clip_dir(self, tmp_path: Path):
        def download_single_clip(clip_id: str, output_dir: Path) -> Path:
            manifest = Path(output_dir) / "clips" / clip_id / f"pai_{clip_id}.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}")
            return manifest

        downloader = types.SimpleNamespace(download_single_clip=download_single_clip, output_dir=tmp_path)
        log_parser = NCoreLogParser(None, "clip-a", None, "ncore_train", downloader=downloader)
        with log_parser._resolved_clip() as (clip_root, manifest):
            assert clip_root == tmp_path / "clip-a" and manifest.is_file()
        assert not (tmp_path / "clip-a").exists()


class TestPandasetZip:
    @pytest.fixture
    def fake_hf_hub(self, monkeypatch) -> List[Path]:
        """``hf_hub_download`` that writes a two-log ``pandaset.zip`` into ``local_dir``."""
        zip_paths: List[Path] = []

        def hf_hub_download(repo_id, repo_type, filename, revision, token, local_dir):
            members = ["pandaset/001/meta/gps.json", "pandaset/002/meta/gps.json"]
            zip_paths.append(_write_zip(Path(local_dir) / filename, members))
            return str(zip_paths[-1])

        monkeypatch.setattr(pandaset_download, "_require_hf_hub", lambda: hf_hub_download)
        return zip_paths

    def test_bulk_download_deletes_zip(self, tmp_path: Path, fake_hf_hub: List[Path]):
        PandasetDownloader(output_dir=tmp_path, log_names=["001", "002"]).download()
        assert fake_hf_hub == [tmp_path / "_zip" / "pandaset.zip"]
        assert sorted(path.name for path in tmp_path.iterdir()) == ["001", "002"]

    def test_streaming_zip_below_output_dir(self, tmp_path: Path, fake_hf_hub: List[Path]):
        log_path = PandasetDownloader().download_single_log("002", tmp_path)
        assert fake_hf_hub == [tmp_path / "_zip" / "pandaset.zip"]
        assert log_path == tmp_path / "002" and (log_path / "meta" / "gps.json").is_file()


class TestArchiveStaging:
    """Archives are staged next to their destination instead of in the system temp directory."""

    def test_nureasoning_clip_zip(self, tmp_path: Path, monkeypatch):
        local_dirs: List[Path] = []

        def hf_hub_download(repo_id, repo_type, filename, revision, token, local_dir):
            local_dirs.append(Path(local_dir))
            return str(_write_zip(Path(local_dir) / filename, ["clip-a/metadata.json"]))

        monkeypatch.setattr(nureasoning_download, "_require_hf_hub", lambda: (None, hf_hub_download))
        clip_dir = NureasoningDownloader().download_single_clip("train", "part_1", "clip-a", tmp_path)
        assert local_dirs == [tmp_path / "_zip" / "clip-a"]
        assert (clip_dir / "metadata.json").is_file()
        assert not (tmp_path / "_zip" / "clip-a").exists()

    def test_truckdrive_modality_zip(self, tmp_path: Path):
        scene_dir = tmp_path / "scene_01_1"
        zip_path = _write_zip(scene_dir / "lidar.zip", ["scene_01_1/lidar/0.bin", "scene_01_1/lidar/1.bin"])
        TruckDriveDownloader(output_dir=tmp_path)._unzip_modality_archive(zip_path)
        assert sorted(path.name for path in (scene_dir / "lidar").iterdir()) == ["0.bin", "1.bin"]
        assert sorted(path.name for path in scene_dir.iterdir()) == ["lidar", "lidar.zip"]
