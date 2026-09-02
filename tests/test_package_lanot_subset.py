"""Tests for scripts/package_lanot_subset.py, offline.

A four-by-four MARIDA-shaped dataset is written with rasterio into a temporary
directory: real GeoTIFFs with a UTM transform, real class codes, one patch on a tile
inside LANOT's footprint and one outside it. That is enough to check what the script
promises: tile and class filtering, NaN handling, pixel-centre coordinates in both CRSs,
and that the expression (1) column is the operator from the sibling script.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from mdebris.data.marida import MARIDA_BANDS, MARIDA_CLASSES

pd = pytest.importorskip("pandas")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def script():
    """The script as a module. Scripts are not a package, so import by path."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module("package_lanot_subset")


SIZE = 4
ORIGIN_X, ORIGIN_Y, PIXEL = 500_000.0, 1_800_000.0, 10.0
EPSG = 32616

INSIDE_ROI = "27-1-19_16QED_0"  # Roatan, on the LANOT tile list
OUTSIDE_ROI = "16-2-18_18QWF_0"  # Haiti, not on it


def _code(name: str) -> int:
    return MARIDA_CLASSES.index(name) + 1


def _write_patch(
    root: Path, roi: str, image: np.ndarray, cls: np.ndarray, conf: np.ndarray
) -> None:
    folder = root / "patches" / ("S2_" + "_".join(roi.split("_")[:-1]))
    folder.mkdir(parents=True, exist_ok=True)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL)
    common = {
        "driver": "GTiff",
        "height": SIZE,
        "width": SIZE,
        "crs": f"EPSG:{EPSG}",
        "transform": transform,
    }
    with rasterio.open(
        folder / f"S2_{roi}.tif", "w", count=len(MARIDA_BANDS), dtype="float32", **common
    ) as dst:
        dst.write(image.astype("float32"))
    for suffix, data in (("_cl", cls), ("_conf", conf)):
        with rasterio.open(
            folder / f"S2_{roi}{suffix}.tif", "w", count=1, dtype="uint8", **common
        ) as dst:
            dst.write(data.astype("uint8"), 1)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Two patches. The inside one has every class the script cares about plus water."""
    root = tmp_path / "marida"
    (root / "splits").mkdir(parents=True)
    (root / "splits" / "train_X.txt").write_text(INSIDE_ROI + "\n")
    (root / "splits" / "val_X.txt").write_text(OUTSIDE_ROI + "\n")
    (root / "splits" / "test_X.txt").write_text("")

    # Dark water everywhere, then a few pixels that the operator either fires on or not.
    image = np.full((len(MARIDA_BANDS), SIZE, SIZE), 0.02, dtype="float32")
    b04, b08, b8a, b11 = (MARIDA_BANDS.index(b) for b in ("B04", "B08", "B8A", "B11"))
    # (0, 0): sargassum that satisfies expression (1).
    image[b04, 0, 0], image[b08, 0, 0], image[b8a, 0, 0], image[b11, 0, 0] = 0.03, 0.06, 0.05, 0.02
    # (0, 1): thick cloud, bright in SWIR, so the b11 term rejects it.
    image[:, 0, 1] = 0.3
    # (0, 2): thin cloud, dim in SWIR, red edge lifted, so expression (1) fires.
    image[b04, 0, 2], image[b08, 0, 2], image[b8a, 0, 2], image[b11, 0, 2] = 0.05, 0.065, 0.06, 0.03
    # (1, 0): shadow that has a NaN band and must be dropped.
    image[3, 1, 0] = np.nan

    cls = np.zeros((SIZE, SIZE), dtype="uint8")
    cls[0, 0] = _code("Dense Sargassum")
    cls[0, 1] = _code("Clouds")
    cls[0, 2] = _code("Clouds")
    cls[0, 3] = _code("Sparse Sargassum")
    cls[1, 0] = _code("Cloud Shadows")
    cls[1, 1] = _code("Cloud Shadows")
    cls[1, 2] = _code("Shallow Water")
    cls[1, 3] = _code("Marine Water")  # not in the subset
    cls[2, 0] = _code("Ship")  # not in the subset
    # Row 3 stays unlabelled.

    conf = np.where(cls > 0, 1, 0).astype("uint8")
    conf[0, 3] = 3  # Low
    _write_patch(root, INSIDE_ROI, image, cls, conf)

    outside = np.zeros((SIZE, SIZE), dtype="uint8")
    outside[:] = _code("Dense Sargassum")
    _write_patch(root, OUTSIDE_ROI, image, outside, np.ones((SIZE, SIZE), dtype="uint8"))

    (root / "labels_mapping.txt").write_text(json.dumps({}))
    return root


def test_overlap_tiles_are_the_four_caribbean_sites(script) -> None:
    assert script.overlap_tiles() == frozenset({"16PCC", "16PDC", "16PEC", "16QED"})


def test_only_overlap_tiles_and_subset_classes_survive(script, dataset: Path) -> None:
    df = script.build_subset(dataset)
    assert set(df["tile"]) == {"16QED"}
    assert set(df["split"]) == {"train"}
    assert set(df["label"]) <= set(script.SUBSET_CLASSES)
    assert "Marine Water" not in set(df["label"])
    # Six labelled subset pixels, one of them dropped for the NaN band.
    assert len(df) == 6
    assert list(df.columns) == list(script.COLUMNS)


def test_nan_pixels_are_dropped(script, dataset: Path) -> None:
    df = script.build_subset(dataset)
    assert not ((df["row"] == 1) & (df["col"] == 0)).any()
    assert not df[list(MARIDA_BANDS)].isna().any().any()


def test_coordinates_are_pixel_centres_in_both_crs(script, dataset: Path) -> None:
    df = script.build_subset(dataset).set_index(["row", "col"])
    pixel = df.loc[(1, 2)]
    assert pixel["epsg"] == EPSG
    assert pixel["easting"] == ORIGIN_X + 2 * PIXEL + PIXEL / 2
    assert pixel["northing"] == ORIGIN_Y - 1 * PIXEL - PIXEL / 2
    lon, lat = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True).transform(
        pixel["easting"], pixel["northing"]
    )
    assert pixel["lon"] == pytest.approx(lon, abs=1e-6)
    assert pixel["lat"] == pytest.approx(lat, abs=1e-6)


def test_expression_column_is_the_published_operator(script, dataset: Path) -> None:
    df = script.build_subset(dataset)
    by_pixel = df.set_index(["row", "col"])
    assert by_pixel.loc[(0, 0), "lanot_expr1"]  # sargassum, fires
    assert not by_pixel.loc[(0, 1), "lanot_expr1"]  # thick cloud, SWIR term rejects
    assert by_pixel.loc[(0, 2), "lanot_expr1"]  # thin cloud, gets through
    assert not by_pixel.loc[(0, 1), "b11_gate"]
    assert by_pixel.loc[(0, 2), "b11_gate"]

    from eval_lanot_operator import lanot_operator

    bands = {name: df[name].to_numpy() for name in MARIDA_BANDS}
    assert (df["lanot_expr1"].to_numpy() == lanot_operator(bands)).all()


def test_confidence_and_date_are_decoded(script, dataset: Path) -> None:
    df = script.build_subset(dataset).set_index(["row", "col"])
    assert df.loc[(0, 3), "confidence"] == "Low"
    assert df.loc[(0, 0), "confidence"] == "High"
    assert set(df["date"]) == {"2019-01-27"}


def test_summary_counts_and_thin_cloud(script, dataset: Path) -> None:
    summary = script.summarise(script.build_subset(dataset))
    assert summary["pixels"] == 6
    assert summary["per_class"]["Clouds"]["pixels"] == 2
    assert summary["per_class"]["Clouds"]["expr1_fires"] == 1
    assert summary["per_class"]["Clouds"]["b11_gate_passes"] == 1
    assert summary["per_class"]["Dense Sargassum"]["expr1_fraction"] == 1.0
    assert summary["thin_cloud"] == {
        "pixels": 1,
        "fraction_of_cloud": 0.5,
        "expr1_fires": 1,
        "expr1_fraction": 1.0,
        "median_b11": pytest.approx(0.03),
    }
    assert summary["per_split"] == {
        "train": {
            "Dense Sargassum": 1,
            "Sparse Sargassum": 1,
            "Clouds": 2,
            "Cloud Shadows": 1,
            "Shallow Water": 1,
        }
    }
    assert summary["per_tile"]["16QED"]["site"] == "Roatan, Honduras"
    assert summary["splits"] == ["train"]


def test_per_term_diagnostic_names_the_rejecting_term(script, dataset: Path) -> None:
    summary = script.summarise(script.build_subset(dataset))
    clouds = summary["per_term"]["Clouds"]
    assert list(clouds) == list(script.EXPRESSION_TERMS)
    # The thick cloud fails the SWIR term and the b8A ceiling; the thin one passes both.
    assert clouds["b11 < 0.05"] == 0.5
    assert clouds["b8A < 0.07"] == 0.5
    assert summary["per_term"]["Dense Sargassum"] == dict.fromkeys(script.EXPRESSION_TERMS, 1.0)
    assert summary["per_class"]["Dense Sargassum"]["median_b8a"] == pytest.approx(0.05)
    text = script._markdown(summary, "lanot_subset.csv.gz")
    assert "## Which term does the work" in text


def test_markdown_names_every_class_and_the_caveat(script, dataset: Path) -> None:
    summary = script.summarise(script.build_subset(dataset))
    text = script._markdown(summary, "lanot_subset.csv.gz")
    for name in script.SUBSET_CLASSES:
        assert name in text
    assert "Rayleigh-corrected" in text
    assert "It is a baseline for these classes" in text
    assert "CC-BY-4.0" in text


def test_csv_round_trips(script, dataset: Path, tmp_path: Path) -> None:
    df = script.build_subset(dataset)
    out = tmp_path / "subset.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    back = pd.read_csv(out)
    assert list(back.columns) == list(script.COLUMNS)
    assert len(back) == len(df)
    assert back["lanot_expr1"].dtype == bool
