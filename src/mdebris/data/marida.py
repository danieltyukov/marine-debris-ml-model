"""MARIDA, the Marine Debris Archive, as a downloadable and loadable dataset.

MARIDA is the public Sentinel-2 benchmark for marine debris: 1381 pixel-annotated
256x256 patches over 17 sites, published alongside Kikaki K, Kakogeorgiou I, Mikeli P,
Raitsos DE, Karantzalos K (2022), "MARIDA: A benchmark for Marine Debris detection from
Sentinel-2 remote sensing data", PLoS ONE 17(1): e0262247.

Everything below was read off the live Zenodo record and the zip's own central directory
rather than taken from documentation:

- record 5151941 (concept record 5151940), version 1.0.0, published 2021-08-01
- one file, ``MARIDA.zip``, 1164612748 bytes, md5 9bf32266f6e3711c9dfa3699b856c76f
- licensed CC-BY-4.0, open access, no credentials needed
- 4528 zip entries laid out as ``patches/``, ``shapefiles/``, ``splits/`` and
  ``labels_mapping.txt`` at the archive root, with no wrapper directory
- splits hold 694 train, 328 validation and 359 test patch ids, totalling 1381

Why this matters here
---------------------
The legacy model had a single ``marine_debris`` class and therefore no way to learn what
debris is *not*. MARIDA labels the confusers explicitly (sargassum, foam, wakes,
sediment-laden water), which is exactly the signal :class:`~mdebris.types.SurfaceClass`
was widened to carry. :data:`MARIDA_TO_SURFACE` is the bridge between the two.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mdebris.config import settings
from mdebris.types import SurfaceClass

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "MARIDA_BANDS",
    "MARIDA_CITATION",
    "MARIDA_CLASSES",
    "MARIDA_CONCEPT_DOI",
    "MARIDA_CONFIDENCE",
    "MARIDA_DOI",
    "MARIDA_LICENSE",
    "MARIDA_MD5",
    "MARIDA_PATCH_SIZE",
    "MARIDA_RECORD_ID",
    "MARIDA_REPORT",
    "MARIDA_SITES",
    "MARIDA_SIZE_BYTES",
    "MARIDA_TO_SURFACE",
    "MARIDA_URL",
    "SPLIT_SIZES",
    "MaridaError",
    "MaridaPatch",
    "Split",
    "class_id",
    "download_marida",
    "is_downloaded",
    "load_marida_split",
    "marida_root",
    "split_ids",
]

Split = Literal["train", "val", "test"]

MARIDA_RECORD_ID = 5151941
MARIDA_CONCEPT_RECORD_ID = 5151940
MARIDA_DOI = "10.5281/zenodo.5151941"
MARIDA_CONCEPT_DOI = "10.5281/zenodo.5151940"
MARIDA_URL = f"https://zenodo.org/api/records/{MARIDA_RECORD_ID}/files/MARIDA.zip/content"
MARIDA_SIZE_BYTES = 1_164_612_748
MARIDA_MD5 = "9bf32266f6e3711c9dfa3699b856c76f"
MARIDA_LICENSE = "CC-BY-4.0"
MARIDA_CITATION = (
    "Kikaki K, Kakogeorgiou I, Mikeli P, Raitsos DE, Karantzalos K (2022) MARIDA: A "
    "benchmark for Marine Debris detection from Sentinel-2 remote sensing data. "
    "PLoS ONE 17(1): e0262247. https://doi.org/10.1371/journal.pone.0262247"
)
MARIDA_PATCH_SIZE = 256

# The 15 annotated classes in dataset order. The pixel value stored in each ``*_cl.tif``
# is the 1-based index into this list; 0 means unannotated.
MARIDA_CLASSES: tuple[str, ...] = (
    "Marine Debris",
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
)

MARIDA_CONFIDENCE: tuple[str, ...] = ("High", "Moderate", "Low")

# Whether a marine debris field report exists near the annotation.
MARIDA_REPORT: tuple[str, ...] = ("Very close", "Away", "No")

# The 11 Sentinel-2 bands stacked in each patch, in band order. B09 and B10 are excluded
# by the dataset authors: they carry atmospheric signal, not surface reflectance.
MARIDA_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)

# Sentinel-2 tile id to the place it covers, from the dataset's own mapping.
MARIDA_SITES: dict[str, str] = {
    "16PCC": "Motagua, Guatemala",
    "16PDC": "Ulua, Honduras",
    "16PEC": "La Ceiba, Honduras",
    "16QED": "Roatan, Honduras",
    "18QWF": "Haiti",
    "18QYF": "Haiti",
    "18QYG": "Haiti",
    "19QDA": "Santo Domingo, Dominican Republic",
    "30VWH": "Scotland",
    "36JUN": "Durban, South Africa",
    "48MXU": "Jakarta, Indonesia",
    "48MYU": "Jakarta, Indonesia",
    "48PZC": "Danang, Vietnam",
    "50LLR": "Bali, Indonesia",
    "51PTS": "Manila, Philippines",
    "51RVQ": "Yangtze, China",
    "52SDD": "Nakdong, South Korea",
}

# MARIDA class to this project's coarser sea-surface vocabulary.
#
# MARIDA's own reference loaders aggregate Waves, Cloud Shadows, Wakes and Mixed Water
# into Marine Water, because at 0.05% to 1.3% of labelled pixels each they are too rare to
# train on. This mapping keeps Waves (as FOAM) and Wakes (as WAKE) instead, since
# SurfaceClass exists precisely to model those two as debris confusers. Only Mixed Water
# and Shallow Water collapse into WATER, and Cloud Shadows joins CLOUD rather than water
# because what matters downstream is that the pixel is unusable, not what shade it is.
MARIDA_TO_SURFACE: dict[str, SurfaceClass] = {
    "Marine Debris": SurfaceClass.DEBRIS,
    "Dense Sargassum": SurfaceClass.SARGASSUM,
    "Sparse Sargassum": SurfaceClass.SARGASSUM,
    "Natural Organic Material": SurfaceClass.SARGASSUM,
    "Ship": SurfaceClass.SHIP,
    "Clouds": SurfaceClass.CLOUD,
    "Marine Water": SurfaceClass.WATER,
    "Sediment-Laden Water": SurfaceClass.SEDIMENT,
    "Foam": SurfaceClass.FOAM,
    "Turbid Water": SurfaceClass.SEDIMENT,
    "Shallow Water": SurfaceClass.WATER,
    "Waves": SurfaceClass.FOAM,
    "Cloud Shadows": SurfaceClass.CLOUD,
    "Wakes": SurfaceClass.WAKE,
    "Mixed Water": SurfaceClass.WATER,
}

_SPLIT_FILES: dict[str, str] = {
    "train": "train_X.txt",
    "val": "val_X.txt",
    "test": "test_X.txt",
}

# Patch counts per split, read from the published split files. Used to sanity-check a
# download that unpacked only partially.
SPLIT_SIZES: dict[str, int] = {"train": 694, "val": 328, "test": 359}


class MaridaError(RuntimeError):
    """MARIDA is missing, incomplete, or could not be downloaded."""


@dataclass(frozen=True, slots=True)
class MaridaPatch:
    """One 256x256 annotated patch, resolved to file paths and metadata.

    Pixel arrays are loaded on demand through :meth:`read` so that listing a split stays
    instant: eagerly loading all 694 training patches costs several GB of RAM, which the
    reference implementation does and which makes exploratory work painful.
    """

    roi: str
    """Patch id as it appears in the split files, ``dd-mm-yy_TILE_index``."""

    image: Path
    """Multi-band GeoTIFF, 11 bands in :data:`MARIDA_BANDS` order."""

    classes: Path
    """Pixel class mask, values 1..15 indexing :data:`MARIDA_CLASSES`, 0 unannotated."""

    confidence: Path
    """Pixel confidence mask, values 1..3 indexing :data:`MARIDA_CONFIDENCE`."""

    labels: tuple[str, ...] = ()
    """Multi-label class names present in this patch, from ``labels_mapping.txt``."""

    @property
    def tile(self) -> str:
        """Sentinel-2 MGRS tile id, e.g. ``'16PCC'``."""
        return self.roi.split("_")[1]

    @property
    def site(self) -> str:
        """Human-readable place name for the tile."""
        return MARIDA_SITES.get(self.tile, self.tile)

    @property
    def date(self) -> str:
        """Acquisition date as ISO ``YYYY-MM-DD``.

        The dataset encodes dates as ``d-m-yy`` with no zero padding, which sorts wrongly
        as a string and is the reason this property exists.
        """
        day, month, year = self.roi.split("_")[0].split("-")
        return f"20{int(year):02d}-{int(month):02d}-{int(day):02d}"

    @property
    def surface_labels(self) -> tuple[SurfaceClass, ...]:
        """Multi-labels translated into :class:`~mdebris.types.SurfaceClass`, de-duplicated."""
        seen: dict[SurfaceClass, None] = {}
        for name in self.labels:
            seen.setdefault(MARIDA_TO_SURFACE[name], None)
        return tuple(seen)

    @property
    def has_debris(self) -> bool:
        """True when the patch is annotated with Marine Debris."""
        return "Marine Debris" in self.labels

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the patch as ``(bands, classes, confidence)`` arrays.

        Returns:
            ``bands`` is float32 ``(11, 256, 256)`` surface reflectance and may contain
            NaN where the source scene had no data. ``classes`` and ``confidence`` are
            uint8 ``(256, 256)``.
        """
        import numpy as np
        import rasterio

        with rasterio.open(self.image) as src:
            bands = src.read().astype("float32")
        with rasterio.open(self.classes) as src:
            cls = src.read()[0].astype("uint8")
        with rasterio.open(self.confidence) as src:
            conf = src.read()[0].astype("uint8")
        return bands, np.asarray(cls), np.asarray(conf)


def class_id(name: str) -> int:
    """Pixel value used for a class name in the ``*_cl.tif`` masks, 1-based.

    Raises:
        KeyError: if ``name`` is not one of :data:`MARIDA_CLASSES`.
    """
    try:
        return MARIDA_CLASSES.index(name) + 1
    except ValueError:
        raise KeyError(f"unknown MARIDA class {name!r}; expected one of {MARIDA_CLASSES}") from None


def marida_root(dest: Path | str | None = None) -> Path:
    """Directory the dataset unpacks into, ``<cache_dir>/marida`` by default."""
    return Path(dest) if dest is not None else settings.cache_dir / "marida"


def is_downloaded(dest: Path | str | None = None) -> bool:
    """True when a complete-looking MARIDA is already unpacked at ``dest``."""
    root = marida_root(dest)
    if not (root / "labels_mapping.txt").is_file() or not (root / "patches").is_dir():
        return False
    return all((root / "splits" / f).is_file() for f in _SPLIT_FILES.values())


def download_marida(
    dest: Path | str | None = None,
    *,
    force: bool = False,
    verify: bool = True,
    progress: bool = True,
) -> Path:
    """Download and unpack MARIDA, resuming an interrupted transfer.

    The archive is 1.1 GB compressed and 4.4 GB unpacked, so the download is resumable
    via HTTP range requests (Zenodo returns 206 for ranged reads) and the partial file is
    kept as ``MARIDA.zip.part`` until the md5 matches. A truncated download that got
    renamed to its final name would look valid forever, which is the failure this avoids.

    Args:
        dest: Target directory. Defaults to ``<settings.cache_dir>/marida``.
        force: Re-download and re-extract even if the dataset is already present.
        verify: Check the published md5 before extracting. Turning this off is only
            sensible when resuming is impossible and a partial dataset is acceptable.
        progress: Show a progress bar.

    Returns:
        The directory holding ``patches/``, ``shapefiles/``, ``splits/`` and
        ``labels_mapping.txt``. ``MARIDA.zip`` is kept next to them so that a later
        ``force=True`` does not re-download; delete it to reclaim 1.1 GB.

    Raises:
        MaridaError: on a network failure, a checksum mismatch, or a corrupt archive.
    """
    root = marida_root(dest)
    if is_downloaded(root) and not force:
        return root
    root.mkdir(parents=True, exist_ok=True)

    archive = root / "MARIDA.zip"
    partial = root / "MARIDA.zip.part"
    if force:
        partial.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)

    if not archive.is_file():
        _download_resumable(MARIDA_URL, partial, MARIDA_SIZE_BYTES, progress=progress)
        if verify:
            digest = _md5(partial, progress=progress)
            if digest != MARIDA_MD5:
                partial.unlink(missing_ok=True)
                raise MaridaError(
                    f"checksum mismatch for MARIDA.zip: got {digest}, expected {MARIDA_MD5}. "
                    "The partial file was removed; run download_marida again."
                )
        partial.replace(archive)

    _extract(archive, root, progress=progress)
    if not is_downloaded(root):
        raise MaridaError(
            f"extraction to {root} did not produce the expected layout "
            "(patches/, splits/, labels_mapping.txt)"
        )
    return root


def _download_resumable(url: str, target: Path, total: int, *, progress: bool) -> None:
    """Fetch ``url`` into ``target``, continuing from whatever is already on disk."""
    import requests

    have = target.stat().st_size if target.is_file() else 0
    if have >= total:
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            if have and resp.status_code != 206:
                # The server ignored the range request and is restarting from zero, so the
                # bytes already on disk have to be overwritten rather than appended to.
                have = 0
            mode = "ab" if have else "wb"
            bar = _bar(total, have, "MARIDA.zip", enabled=progress)
            with target.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            if bar is not None:
                bar.close()
    except Exception as exc:
        raise MaridaError(
            f"could not download MARIDA from {url}: {type(exc).__name__}: {exc}\n"
            f"The partial file is kept at {target} and the next call resumes from there."
        ) from exc


def _md5(path: Path, *, progress: bool) -> str:
    """Stream the md5 of a large file without holding it in memory."""
    # md5 is the algorithm Zenodo publishes, so it is what has to be recomputed here. It
    # is an integrity check against a truncated transfer, not a security boundary.
    digest = hashlib.md5(usedforsecurity=False)
    bar = _bar(path.stat().st_size, 0, "verifying md5", enabled=progress)
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
            if bar is not None:
                bar.update(len(chunk))
    if bar is not None:
        bar.close()
    return digest.hexdigest()


def _extract(archive: Path, root: Path, *, progress: bool) -> None:
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            bar = _bar(len(members), 0, "extracting", enabled=progress, unit="file")
            for member in members:
                _safe_extract(zf, member, root)
                if bar is not None:
                    bar.update(1)
            if bar is not None:
                bar.close()
    except zipfile.BadZipFile as exc:
        raise MaridaError(
            f"{archive} is not a readable zip; delete it and run download_marida again"
        ) from exc


def _safe_extract(zf: zipfile.ZipFile, member: zipfile.ZipInfo, root: Path) -> None:
    """Extract one member, refusing paths that would escape ``root``.

    Python's ``ZipFile.extractall`` is safe in modern versions, but the check is kept
    explicit because this code unpacks a file fetched over the network.
    """
    target = (root / member.filename).resolve()
    if not target.is_relative_to(root.resolve()):
        raise MaridaError(f"refusing to extract {member.filename!r} outside {root}")
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _bar(total: int, initial: int, desc: str, *, enabled: bool, unit: str = "B"):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a hard dependency
        return None
    return tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit=unit,
        unit_scale=unit == "B",
        unit_divisor=1024,
    )


def split_ids(split: Split, dest: Path | str | None = None) -> list[str]:
    """Patch ids in one split, in published order.

    Raises:
        MaridaError: if the dataset is not downloaded.
        ValueError: if ``split`` is not train, val or test.
    """
    if split not in _SPLIT_FILES:
        raise ValueError(f"split must be one of {sorted(_SPLIT_FILES)}, got {split!r}")
    root = marida_root(dest)
    path = root / "splits" / _SPLIT_FILES[split]
    if not path.is_file():
        raise MaridaError(
            f"MARIDA split file not found at {path}. Run mdebris.data.download_marida() "
            f"first (1.1 GB download, 4.4 GB unpacked), or point dest= at an existing copy."
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_marida_split(
    split: Split,
    dest: Path | str | None = None,
    *,
    debris_only: bool = False,
) -> list[MaridaPatch]:
    """List one split's patches with their labels and file paths.

    Pixels are not read here. Call :meth:`MaridaPatch.read` per patch, which keeps memory
    flat and lets callers filter before paying for I/O.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        dest: Dataset directory, defaults to ``<settings.cache_dir>/marida``.
        debris_only: Keep only patches annotated with Marine Debris.

    Returns:
        Patches in published order.

    Raises:
        MaridaError: if the dataset is missing or a patch file referenced by a split is
            absent, which means the archive unpacked incompletely.

    Example:
        >>> patches = load_marida_split("val")             # doctest: +SKIP
        >>> bands, classes, conf = patches[0].read()       # doctest: +SKIP
        >>> bands.shape                                    # doctest: +SKIP
        (11, 256, 256)
    """
    root = marida_root(dest)
    rois = split_ids(split, root)
    labels_path = root / "labels_mapping.txt"
    if not labels_path.is_file():
        raise MaridaError(f"labels_mapping.txt not found at {labels_path}")
    label_map: dict[str, list[int]] = json.loads(labels_path.read_text())

    patches: list[MaridaPatch] = []
    for roi in rois:
        # Patches live under a per-acquisition folder named after everything but the
        # trailing patch index, e.g. roi "1-12-19_48MYU_0" sits in "S2_1-12-19_48MYU".
        parts = roi.split("_")
        folder = root / "patches" / ("S2_" + "_".join(parts[:-1]))
        stem = f"S2_{roi}"
        image = folder / f"{stem}.tif"
        if not image.is_file():
            raise MaridaError(
                f"split {split!r} references {image} but it is missing; "
                "the archive may have unpacked incompletely, re-run download_marida(force=True)"
            )
        flags = label_map.get(f"{stem}.tif", [])
        names = tuple(MARIDA_CLASSES[i] for i, on in enumerate(flags) if on)
        if debris_only and "Marine Debris" not in names:
            continue
        patches.append(
            MaridaPatch(
                roi=roi,
                image=image,
                classes=folder / f"{stem}_cl.tif",
                confidence=folder / f"{stem}_conf.tif",
                labels=names,
            )
        )
    return patches
