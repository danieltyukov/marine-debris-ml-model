"""Tests for the MARIDA loader.

The dataset is 1.1 GB, so the default run never downloads it. What is tested offline is
everything that can be wrong without it: the published constants, the class vocabulary and
its mapping onto SurfaceClass, the patch-id to file-path convention, and the error
messages a user hits when the dataset is simply not there. A tiny fake dataset tree
exercises the loader end to end using the real published split ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdebris.data.marida import (
    MARIDA_BANDS,
    MARIDA_CLASSES,
    MARIDA_CONFIDENCE,
    MARIDA_MD5,
    MARIDA_PATCH_SIZE,
    MARIDA_RECORD_ID,
    MARIDA_SITES,
    MARIDA_SIZE_BYTES,
    MARIDA_TO_SURFACE,
    MARIDA_URL,
    SPLIT_SIZES,
    MaridaError,
    MaridaPatch,
    class_id,
    is_downloaded,
    load_marida_split,
    marida_root,
    split_ids,
)
from mdebris.types import SurfaceClass

# Real patch ids taken from the published splits/*.txt files.
REAL_TRAIN_IDS = ["1-12-19_48MYU_0", "1-12-19_48MYU_1", "11-1-19_19QDA_0"]
REAL_VAL_IDS = ["16-2-18_16PEC_0"]


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """A MARIDA tree with the real layout and real ids, but empty rasters.

    Enough to test path construction, split parsing and label decoding without the 1.1 GB
    download. ``MaridaPatch.read`` is not exercised here; that needs real GeoTIFFs and is
    covered by the network test.
    """
    root = tmp_path / "marida"
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "train_X.txt").write_text("\n".join(REAL_TRAIN_IDS) + "\n")
    (root / "splits" / "val_X.txt").write_text("\n".join(REAL_VAL_IDS) + "\n")
    (root / "splits" / "test_X.txt").write_text("")

    labels: dict[str, list[int]] = {}
    for roi in REAL_TRAIN_IDS + REAL_VAL_IDS:
        folder = root / "patches" / ("S2_" + "_".join(roi.split("_")[:-1]))
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "_cl", "_conf"):
            (folder / f"S2_{roi}{suffix}.tif").write_bytes(b"")
        # Ship + Marine Water for every patch, plus Marine Debris on the first one.
        flags = [0] * len(MARIDA_CLASSES)
        flags[MARIDA_CLASSES.index("Ship")] = 1
        flags[MARIDA_CLASSES.index("Marine Water")] = 1
        if roi == REAL_TRAIN_IDS[0]:
            flags[MARIDA_CLASSES.index("Marine Debris")] = 1
        labels[f"S2_{roi}.tif"] = flags
    (root / "labels_mapping.txt").write_text(json.dumps(labels))
    return root


# -- published facts --------------------------------------------------------------


def test_zenodo_coordinates_match_the_published_record() -> None:
    assert MARIDA_RECORD_ID == 5151941
    assert MARIDA_URL == ("https://zenodo.org/api/records/5151941/files/MARIDA.zip/content")
    assert MARIDA_SIZE_BYTES == 1_164_612_748
    assert MARIDA_MD5 == "9bf32266f6e3711c9dfa3699b856c76f"
    assert len(MARIDA_MD5) == 32


def test_split_sizes_sum_to_the_published_patch_count() -> None:
    assert sum(SPLIT_SIZES.values()) == 1381
    assert SPLIT_SIZES == {"train": 694, "val": 328, "test": 359}


def test_the_class_list_is_the_real_one() -> None:
    assert len(MARIDA_CLASSES) == 15
    assert MARIDA_CLASSES[0] == "Marine Debris"
    for name in (
        "Dense Sargassum",
        "Sparse Sargassum",
        "Natural Organic Material",
        "Ship",
        "Clouds",
        "Marine Water",
        "Sediment-Laden Water",
        "Foam",
        "Turbid Water",
        "Shallow Water",
        "Waves",
        "Cloud Shadows",
        "Wakes",
        "Mixed Water",
    ):
        assert name in MARIDA_CLASSES


def test_class_ids_are_one_based_to_match_the_mask_pixel_values() -> None:
    assert class_id("Marine Debris") == 1
    assert class_id("Marine Water") == 7
    assert class_id("Mixed Water") == 15
    for name in MARIDA_CLASSES:
        assert MARIDA_CLASSES[class_id(name) - 1] == name


def test_class_id_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="Plastic"):
        class_id("Plastic")


def test_patches_are_the_documented_size_and_band_count() -> None:
    assert MARIDA_PATCH_SIZE == 256
    assert len(MARIDA_BANDS) == 11
    # B09 and B10 carry atmospheric signal, not surface reflectance, and are excluded.
    assert "B09" not in MARIDA_BANDS
    assert "B10" not in MARIDA_BANDS
    assert MARIDA_BANDS[0] == "B01"
    assert MARIDA_BANDS[-1] == "B12"


def test_confidence_levels_are_ordered_high_to_low() -> None:
    assert MARIDA_CONFIDENCE == ("High", "Moderate", "Low")


def test_every_site_key_is_a_plausible_mgrs_tile() -> None:
    assert len(MARIDA_SITES) == 17
    for tile in MARIDA_SITES:
        assert len(tile) == 5
        assert tile[:2].isdigit()
        assert tile[2:].isalpha() and tile[2:].isupper()


# -- SurfaceClass bridge ----------------------------------------------------------


def test_every_marida_class_maps_onto_a_surface_class() -> None:
    assert set(MARIDA_TO_SURFACE) == set(MARIDA_CLASSES)
    assert all(isinstance(v, SurfaceClass) for v in MARIDA_TO_SURFACE.values())


def test_debris_maps_to_the_one_target_class() -> None:
    assert MARIDA_TO_SURFACE["Marine Debris"] is SurfaceClass.DEBRIS
    targets = [k for k, v in MARIDA_TO_SURFACE.items() if v.is_target]
    assert targets == ["Marine Debris"]


def test_the_known_confusers_are_kept_as_confusers() -> None:
    """These are the classes the single-class legacy model could not distinguish."""
    for name in ("Dense Sargassum", "Sparse Sargassum", "Foam", "Wakes", "Waves"):
        assert MARIDA_TO_SURFACE[name].is_confuser, name
    assert MARIDA_TO_SURFACE["Sediment-Laden Water"].is_confuser
    assert MARIDA_TO_SURFACE["Turbid Water"].is_confuser


def test_water_variants_collapse_to_open_water() -> None:
    assert MARIDA_TO_SURFACE["Marine Water"] is SurfaceClass.WATER
    assert MARIDA_TO_SURFACE["Shallow Water"] is SurfaceClass.WATER
    assert MARIDA_TO_SURFACE["Mixed Water"] is SurfaceClass.WATER


def test_cloud_shadow_is_treated_as_cloud_not_as_water() -> None:
    """What matters downstream is that the pixel is unusable."""
    assert MARIDA_TO_SURFACE["Cloud Shadows"] is SurfaceClass.CLOUD
    assert MARIDA_TO_SURFACE["Clouds"] is SurfaceClass.CLOUD


def test_no_marida_class_falls_through_to_unknown() -> None:
    assert SurfaceClass.UNKNOWN not in set(MARIDA_TO_SURFACE.values())


# -- patch metadata ---------------------------------------------------------------


def _patch(roi: str, labels: tuple[str, ...] = ()) -> MaridaPatch:
    stub = Path("/nonexistent")
    return MaridaPatch(roi=roi, image=stub, classes=stub, confidence=stub, labels=labels)


@pytest.mark.parametrize(
    ("roi", "tile", "date"),
    [
        ("1-12-19_48MYU_0", "48MYU", "2019-12-01"),
        ("11-1-19_19QDA_3", "19QDA", "2019-01-11"),
        ("29-11-15_16PEC_0", "16PEC", "2015-11-29"),
        ("8-3-18_16QED_2", "16QED", "2018-03-08"),
    ],
)
def test_patch_ids_decode_into_tile_and_zero_padded_date(roi: str, tile: str, date: str) -> None:
    """MARIDA writes dates as d-m-yy, which does not sort and is not ISO."""
    patch = _patch(roi)
    assert patch.tile == tile
    assert patch.date == date


def test_patch_resolves_its_tile_to_a_place_name() -> None:
    assert _patch("1-12-19_48MYU_0").site == "Jakarta, Indonesia"
    assert _patch("11-1-19_19QDA_0").site == "Santo Domingo, Dominican Republic"


def test_patch_labels_translate_to_surface_classes_without_duplicates() -> None:
    patch = _patch("1-12-19_48MYU_0", ("Dense Sargassum", "Sparse Sargassum", "Marine Water"))
    assert patch.surface_labels == (SurfaceClass.SARGASSUM, SurfaceClass.WATER)


def test_has_debris_reflects_the_multi_label_vector() -> None:
    assert _patch("x_16PCC_0", ("Marine Debris", "Ship")).has_debris
    assert not _patch("x_16PCC_0", ("Ship",)).has_debris


# -- loading ----------------------------------------------------------------------


def test_split_ids_reads_the_published_order(fake_dataset: Path) -> None:
    assert split_ids("train", fake_dataset) == REAL_TRAIN_IDS
    assert split_ids("val", fake_dataset) == REAL_VAL_IDS
    assert split_ids("test", fake_dataset) == []


def test_load_split_builds_the_folder_convention_correctly(fake_dataset: Path) -> None:
    """Patch "1-12-19_48MYU_0" lives in folder "S2_1-12-19_48MYU" as "S2_..._0.tif"."""
    patches = load_marida_split("train", fake_dataset)
    assert [p.roi for p in patches] == REAL_TRAIN_IDS
    first = patches[0]
    assert first.image.name == "S2_1-12-19_48MYU_0.tif"
    assert first.image.parent.name == "S2_1-12-19_48MYU"
    assert first.classes.name == "S2_1-12-19_48MYU_0_cl.tif"
    assert first.confidence.name == "S2_1-12-19_48MYU_0_conf.tif"


def test_load_split_decodes_the_multi_label_vector(fake_dataset: Path) -> None:
    patches = load_marida_split("train", fake_dataset)
    assert patches[0].labels == ("Marine Debris", "Ship", "Marine Water")
    assert patches[1].labels == ("Ship", "Marine Water")


def test_debris_only_filters_to_annotated_debris(fake_dataset: Path) -> None:
    patches = load_marida_split("train", fake_dataset, debris_only=True)
    assert [p.roi for p in patches] == [REAL_TRAIN_IDS[0]]


def test_a_split_referencing_a_missing_patch_is_an_error(fake_dataset: Path) -> None:
    (fake_dataset / "patches" / "S2_1-12-19_48MYU" / "S2_1-12-19_48MYU_0.tif").unlink()
    with pytest.raises(MaridaError, match="unpacked incompletely"):
        load_marida_split("train", fake_dataset)


def test_an_invalid_split_name_is_rejected(fake_dataset: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        split_ids("validation", fake_dataset)  # type: ignore[arg-type]


def test_a_missing_dataset_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(MaridaError) as excinfo:
        load_marida_split("train", tmp_path / "absent")
    message = str(excinfo.value)
    assert "download_marida" in message
    assert "1.1 GB" in message


def test_is_downloaded_detects_a_complete_and_an_incomplete_tree(
    fake_dataset: Path, tmp_path: Path
) -> None:
    assert is_downloaded(fake_dataset)
    assert not is_downloaded(tmp_path / "absent")
    (fake_dataset / "splits" / "val_X.txt").unlink()
    assert not is_downloaded(fake_dataset)


def test_default_root_lives_under_the_configured_cache(tmp_path: Path) -> None:
    assert marida_root().name == "marida"
    assert marida_root(tmp_path) == tmp_path


# -- network ----------------------------------------------------------------------


@pytest.mark.network
def test_the_zenodo_record_still_matches_the_constants_in_this_module() -> None:
    """Catches the dataset being re-versioned or moved out from under us."""
    import requests

    resp = requests.get(f"https://zenodo.org/api/records/{MARIDA_RECORD_ID}", timeout=30)
    resp.raise_for_status()
    record = resp.json()
    assert record["metadata"]["license"]["id"] == "cc-by-4.0"
    files = {f["key"]: f for f in record["files"]}
    assert "MARIDA.zip" in files
    assert files["MARIDA.zip"]["size"] == MARIDA_SIZE_BYTES
    assert files["MARIDA.zip"]["checksum"] == f"md5:{MARIDA_MD5}"


@pytest.mark.network
@pytest.mark.slow
def test_download_marida_end_to_end(tmp_path: Path) -> None:
    """Downloads 1.1 GB and unpacks 4.4 GB. Excluded from the default run for a reason."""
    from mdebris.data.marida import download_marida

    root = download_marida(tmp_path / "marida", progress=False)
    assert is_downloaded(root)
    for split, expected in SPLIT_SIZES.items():
        assert len(split_ids(split, root)) == expected  # type: ignore[arg-type]
    patches = load_marida_split("train", root)
    bands, classes, confidence = patches[0].read()
    assert bands.shape == (len(MARIDA_BANDS), MARIDA_PATCH_SIZE, MARIDA_PATCH_SIZE)
    assert classes.shape == (MARIDA_PATCH_SIZE, MARIDA_PATCH_SIZE)
    assert confidence.shape == (MARIDA_PATCH_SIZE, MARIDA_PATCH_SIZE)
    assert classes.max() <= len(MARIDA_CLASSES)
