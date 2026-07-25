"""Tests for the plotting primitives.

Matplotlib is forced onto the Agg backend so these run headless in CI. The tests
assert on array values and on the fact that drawing does not raise, rather than on
pixel output, since comparing rendered images is brittle across matplotlib versions.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mdebris.types import BBox, Detection, SurfaceClass
from mdebris.viz.plots import (
    CLASS_COLORS,
    figure_grid,
    plot_benchmark_bars,
    plot_confusion_matrix,
    plot_detections,
    plot_index_heatmap,
    plot_mask_overlay,
    save_figure,
    stretch_to_uint8,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def chip() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((64, 64, 3)) * 0.15).astype(np.float32)


@pytest.fixture
def detections() -> list[Detection]:
    return [
        Detection(bbox=BBox(5, 5, 25, 25), score=0.81, label=SurfaceClass.DEBRIS),
        Detection(bbox=BBox(30, 35, 55, 60), score=0.42, label=SurfaceClass.SARGASSUM),
    ]


class TestStretchToUint8:
    def test_returns_uint8_and_uses_full_range(self, chip):
        out = stretch_to_uint8(chip)
        assert out.dtype == np.uint8
        assert out.shape == chip.shape
        assert out.min() == 0 and out.max() == 255

    def test_all_nan_plane_is_zero_not_a_crash(self):
        # No-data regions are common at scene edges; they must not raise.
        out = stretch_to_uint8(np.full((8, 8), np.nan, dtype=np.float32))
        assert out.dtype == np.uint8
        assert (out == 0).all()

    def test_constant_plane_becomes_mid_grey(self):
        # There is no contrast to stretch, so saturating to black or white would
        # misrepresent the data. Mid-grey is the honest rendering.
        out = stretch_to_uint8(np.ones((8, 8), dtype=np.float32))
        assert (out == 128).all()

    def test_nan_does_not_leak_into_percentiles(self):
        plane = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        plane[0, 0] = np.nan
        out = stretch_to_uint8(plane)
        assert np.isfinite(out).all()
        assert out[0, 0] == 0  # NaN renders as the low end

    def test_two_dim_input_supported(self, chip):
        out = stretch_to_uint8(chip[..., 0])
        assert out.ndim == 2 and out.dtype == np.uint8

    def test_per_channel_false_preserves_relative_channel_order(self):
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[..., 0] = 0.1
        img[..., 1] = 0.5
        img[..., 2] = 0.9
        out = stretch_to_uint8(img, per_channel=False)
        assert out[..., 0].max() < out[..., 1].max() < out[..., 2].max()

    @pytest.mark.parametrize(("low", "high"), [(50.0, 50.0), (-1.0, 98.0), (2.0, 101.0)])
    def test_invalid_percentiles_rejected(self, low, high):
        with pytest.raises(ValueError, match="percentiles"):
            stretch_to_uint8(np.zeros((4, 4), dtype=np.float32), low=low, high=high)


class TestPlotDetections:
    def test_draws_one_patch_per_detection(self, chip, detections):
        ax = plot_detections(chip, detections, title="t")
        # Two boxes drawn, each as a Rectangle patch.
        assert len(ax.patches) == 2

    def test_min_score_filters(self, chip, detections):
        ax = plot_detections(chip, detections, min_score=0.5)
        assert len(ax.patches) == 1

    def test_empty_detections_is_not_an_error(self, chip):
        ax = plot_detections(chip, [])
        assert len(ax.patches) == 0

    def test_accepts_uint8_directly(self, detections):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        ax = plot_detections(img, detections)
        assert len(ax.patches) == 2

    def test_every_surface_class_has_a_colour(self):
        # A missing entry would silently fall back to grey and make two classes
        # indistinguishable in a figure.
        for cls in SurfaceClass:
            assert cls in CLASS_COLORS


class TestPlotIndexHeatmap:
    def test_draws_image(self):
        idx = np.random.default_rng(1).normal(0, 0.01, (32, 32)).astype(np.float32)
        ax = plot_index_heatmap(idx, name="FDI")
        assert len(ax.images) == 1

    def test_threshold_contour_drawn_when_in_range(self):
        idx = np.linspace(-0.02, 0.02, 1024, dtype=np.float32).reshape(32, 32)
        ax = plot_index_heatmap(idx, name="FDI", threshold=0.0)
        assert len(ax.collections) > 0

    def test_mask_blanks_pixels(self):
        idx = np.ones((16, 16), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        mask[:8] = True
        ax = plot_index_heatmap(idx, name="FDI", mask=mask)
        shown = ax.images[0].get_array()
        assert np.isnan(np.asarray(shown, dtype=np.float32)[8:]).all()

    def test_all_nan_index_does_not_raise(self):
        ax = plot_index_heatmap(np.full((8, 8), np.nan, dtype=np.float32), name="FDI")
        assert len(ax.images) == 1


class TestPlotMaskOverlay:
    def test_overlay_adds_second_image_layer(self, chip):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:20, 10:20] = True
        ax = plot_mask_overlay(chip, mask)
        assert len(ax.images) == 2

    def test_alpha_is_zero_outside_mask(self, chip):
        mask = np.zeros((64, 64), dtype=bool)
        mask[0:5, 0:5] = True
        ax = plot_mask_overlay(chip, mask, alpha=0.5)
        overlay = np.asarray(ax.images[1].get_array())
        assert overlay[0, 0, 3] == pytest.approx(0.5)
        assert overlay[-1, -1, 3] == pytest.approx(0.0)


class TestPlotConfusionMatrix:
    def test_legacy_shaped_matrix_renders(self):
        # The legacy README reported TP=38, FN=16, FP=11 in this layout.
        ax = plot_confusion_matrix(np.array([[38, 16], [11, 0]]), ["debris", "background"])
        assert ax.get_xlabel() == "Predicted"
        assert ax.get_ylabel() == "Ground truth"

    def test_normalize_rows_sum_to_one(self):
        m = np.array([[38, 16], [11, 0]], dtype=float)
        ax = plot_confusion_matrix(m, ["a", "b"], normalize=True)
        shown = np.asarray(ax.images[0].get_array())
        assert shown[0].sum() == pytest.approx(1.0)

    def test_zero_row_does_not_divide_by_zero(self):
        m = np.array([[1, 0], [0, 0]], dtype=float)
        ax = plot_confusion_matrix(m, ["a", "b"], normalize=True)
        assert np.isfinite(np.asarray(ax.images[0].get_array())).all()


class TestPlotBenchmarkBars:
    def test_bar_count_matches_labels(self):
        ax = plot_benchmark_bars(
            ["512px", "960px"], [0.013, 0.050], ylabel="MP/s", title="throughput", highlight=1
        )
        assert len(ax.patches) == 2

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            plot_benchmark_bars(["a", "b"], [1.0], ylabel="y", title="t")


class TestSaveFigure:
    def test_creates_parent_directories(self, tmp_path):
        fig, _ = plt.subplots()
        out = save_figure(fig, tmp_path / "nested" / "deeper" / "fig.png")
        assert out.exists() and out.stat().st_size > 0

    def test_figure_grid_returns_flat_axes(self):
        _fig, axes = figure_grid(2, 3)
        assert len(axes) == 6
