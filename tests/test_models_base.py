"""Offline correctness tests for NMS, tile merging and image coercion.

Nothing here touches the network or loads weights. These are the parts of the model
layer whose bugs are silent: a wrong NMS keeps duplicates that inflate every count,
and a wrong tile offset puts detections in the wrong place on the map without ever
raising. Expected values are computed by hand in the test, not read off the
implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdebris.models.base import (
    BaseDetector,
    as_uint8_rgb,
    clip_detections,
    merge_tile_detections,
    nms,
    offset_detections,
)
from mdebris.types import BBox, Detection, Detector, SurfaceClass


def det(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    score: float = 0.9,
    label: SurfaceClass = SurfaceClass.DEBRIS,
) -> Detection:
    return Detection(bbox=BBox(xmin, ymin, xmax, ymax), score=score, label=label)


# --------------------------------------------------------------------------------------
# nms
# --------------------------------------------------------------------------------------


class TestNMS:
    def test_identical_boxes_collapse_to_the_highest_score(self):
        dets = [det(0, 0, 10, 10, 0.5), det(0, 0, 10, 10, 0.9), det(0, 0, 10, 10, 0.7)]
        kept = nms(dets, 0.5)
        assert len(kept) == 1
        assert kept[0].score == 0.9

    def test_disjoint_boxes_are_all_kept(self):
        dets = [det(0, 0, 10, 10, 0.9), det(100, 100, 110, 110, 0.8), det(50, 50, 60, 60, 0.7)]
        kept = nms(dets, 0.5)
        assert len(kept) == 3
        # Output is score-ordered.
        assert [d.score for d in kept] == [0.9, 0.8, 0.7]

    def test_overlap_exactly_at_threshold_is_kept(self):
        # Two 10x10 boxes offset so intersection is 10x5=50 and union is 100+100-50=150.
        # IoU = 50/150 = 0.3333... Suppression is strictly greater than the threshold,
        # so at exactly 1/3 the box survives.
        a, b = det(0, 0, 10, 10, 0.9), det(0, 5, 10, 15, 0.8)
        assert a.bbox.iou(b.bbox) == pytest.approx(1 / 3)
        assert len(nms([a, b], 1 / 3)) == 2
        assert len(nms([a, b], 0.33)) == 1

    def test_partial_overlap_above_threshold_is_suppressed(self):
        # Intersection 8x10=80, union 100+100-80=120, IoU = 0.6667.
        a, b = det(0, 0, 10, 10, 0.9), det(2, 0, 12, 10, 0.8)
        assert a.bbox.iou(b.bbox) == pytest.approx(80 / 120)
        assert len(nms([a, b], 0.5)) == 1
        assert len(nms([a, b], 0.7)) == 2

    def test_class_aware_keeps_competing_hypotheses(self):
        # Same water, two explanations. Both must survive: that disagreement is the
        # signal the confuser prompts exist to produce.
        debris = det(0, 0, 10, 10, 0.9, SurfaceClass.DEBRIS)
        sargassum = det(0, 0, 10, 10, 0.8, SurfaceClass.SARGASSUM)
        kept = nms([debris, sargassum], 0.5)
        assert len(kept) == 2
        assert {d.label for d in kept} == {SurfaceClass.DEBRIS, SurfaceClass.SARGASSUM}

    def test_class_agnostic_forces_one_winner(self):
        debris = det(0, 0, 10, 10, 0.9, SurfaceClass.DEBRIS)
        sargassum = det(0, 0, 10, 10, 0.8, SurfaceClass.SARGASSUM)
        kept = nms([debris, sargassum], 0.5, class_agnostic=True)
        assert len(kept) == 1
        assert kept[0].label is SurfaceClass.DEBRIS

    def test_same_class_still_suppressed_when_another_class_overlaps(self):
        dets = [
            det(0, 0, 10, 10, 0.9, SurfaceClass.DEBRIS),
            det(1, 1, 11, 11, 0.8, SurfaceClass.DEBRIS),  # duplicate debris
            det(0, 0, 10, 10, 0.7, SurfaceClass.SHIP),  # different class, survives
        ]
        kept = nms(dets, 0.5)
        assert len(kept) == 2
        assert sorted(str(d.label) for d in kept) == ["marine_debris", "ship"]

    def test_chained_suppression_uses_kept_boxes_not_removed_ones(self):
        # A suppresses B. C overlaps B heavily but A only slightly, so C must survive:
        # a greedy NMS must compare against kept boxes, never against suppressed ones.
        a = det(0, 0, 10, 10, 0.9)
        b = det(5, 0, 15, 10, 0.8)  # IoU with A = 5/15 = 0.333
        c = det(9, 0, 19, 10, 0.7)  # IoU with A = 1/19 = 0.053, with B = 6/14 = 0.43
        assert a.bbox.iou(b.bbox) == pytest.approx(1 / 3)
        assert a.bbox.iou(c.bbox) == pytest.approx(1 / 19)
        kept = nms([a, b, c], 0.3)
        assert [d.score for d in kept] == [0.9, 0.7]

    def test_equal_scores_break_ties_by_input_order_deterministically(self):
        first, second = det(0, 0, 10, 10, 0.5), det(0, 0, 10, 10, 0.5)
        first.source_model, second.source_model = "first", "second"
        kept = nms([first, second], 0.5)
        assert len(kept) == 1
        assert kept[0].source_model == "first"

    def test_empty_input(self):
        assert nms([], 0.5) == []

    def test_threshold_zero_suppresses_any_touching_overlap(self):
        # IoU strictly greater than 0 suppresses; boxes sharing only an edge do not.
        overlapping = [det(0, 0, 10, 10, 0.9), det(9, 0, 19, 10, 0.8)]
        assert len(nms(overlapping, 0.0)) == 1
        edge_touching = [det(0, 0, 10, 10, 0.9), det(10, 0, 20, 10, 0.8)]
        assert len(nms(edge_touching, 0.0)) == 2

    def test_threshold_one_suppresses_nothing_short_of_total_overlap(self):
        dets = [det(0, 0, 10, 10, 0.9), det(1, 1, 11, 11, 0.8)]
        assert len(nms(dets, 1.0)) == 2

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_threshold_rejected(self, bad):
        with pytest.raises(ValueError, match="outside"):
            nms([det(0, 0, 1, 1)], bad)

    def test_does_not_mutate_input_list(self):
        dets = [det(0, 0, 10, 10, 0.5), det(0, 0, 10, 10, 0.9)]
        nms(dets, 0.5)
        assert [d.score for d in dets] == [0.5, 0.9]


# --------------------------------------------------------------------------------------
# offsetting and clipping
# --------------------------------------------------------------------------------------


class TestOffsetDetections:
    def test_boxes_shift_by_the_tile_origin(self):
        moved = offset_detections([det(10, 20, 30, 40)], 100, 200)
        assert moved[0].bbox.as_xyxy() == (110.0, 220.0, 130.0, 240.0)

    def test_returns_copies_so_the_per_tile_list_is_untouched(self):
        original = det(10, 20, 30, 40)
        original.indices["FDI"] = 0.01
        moved = offset_detections([original], 100, 200)
        assert original.bbox.as_xyxy() == (10.0, 20.0, 30.0, 40.0)
        # indices must be a copy, not a shared dict
        moved[0].indices["FDI"] = 0.99
        assert original.indices["FDI"] == 0.01

    def test_preserves_score_label_and_provenance(self):
        d = det(0, 0, 5, 5, 0.42, SurfaceClass.SARGASSUM)
        d.source_model = "owlv2:test"
        moved = offset_detections([d], 7, 9)[0]
        assert moved.score == 0.42
        assert moved.label is SurfaceClass.SARGASSUM
        assert moved.source_model == "owlv2:test"

    def test_zero_offset_is_identity_on_geometry(self):
        moved = offset_detections([det(1, 2, 3, 4)], 0, 0)
        assert moved[0].bbox.as_xyxy() == (1.0, 2.0, 3.0, 4.0)


class TestClipDetections:
    def test_box_extending_past_the_frame_is_trimmed(self):
        kept = clip_detections([det(90, 90, 130, 130)], 100, 100)
        assert kept[0].bbox.as_xyxy() == (90.0, 90.0, 100.0, 100.0)

    def test_box_entirely_outside_is_dropped(self):
        assert clip_detections([det(200, 200, 260, 260)], 100, 100) == []

    def test_box_reduced_below_min_area_is_dropped(self):
        # Survives clipping as a 0.5 x 10 sliver, area 5, below min_area=10.
        assert clip_detections([det(99.5, 0, 140, 10)], 100, 100, min_area=10) == []

    def test_interior_box_is_untouched(self):
        kept = clip_detections([det(10, 10, 20, 20)], 100, 100)
        assert kept[0].bbox.as_xyxy() == (10.0, 10.0, 20.0, 20.0)

    def test_negative_coordinates_clamp_to_zero(self):
        kept = clip_detections([det(-20, -5, 30, 30)], 100, 100)
        assert kept[0].bbox.as_xyxy() == (0.0, 0.0, 30.0, 30.0)


# --------------------------------------------------------------------------------------
# merge_tile_detections
# --------------------------------------------------------------------------------------


class TestMergeTileDetections:
    def test_offsets_are_applied_per_tile(self):
        merged = merge_tile_detections(
            [
                ((0, 0), [det(10, 10, 20, 20, 0.9)]),
                ((960, 0), [det(10, 10, 20, 20, 0.8)]),
                ((0, 960), [det(10, 10, 20, 20, 0.7)]),
            ],
            iou_threshold=0.5,
        )
        assert len(merged) == 3
        boxes = sorted(d.bbox.as_xyxy() for d in merged)
        assert boxes == [
            (10.0, 10.0, 20.0, 20.0),
            (10.0, 970.0, 20.0, 980.0),
            (970.0, 10.0, 980.0, 20.0),
        ]

    def test_offset_tuple_is_x_then_y(self):
        # Guards the one thing a caller is most likely to get backwards: pixel space
        # here is (x, y) like BBox, not (row, col) like a numpy index.
        merged = merge_tile_detections([((100, 0), [det(0, 0, 10, 10)])])
        assert merged[0].bbox.as_xyxy() == (100.0, 0.0, 110.0, 10.0)

    def test_same_patch_seen_in_two_overlapping_tiles_is_deduped(self):
        # Tiles of 960 with 96 px overlap: tile B starts at x=864. A debris patch at
        # scene x 900..940 is at local x 900..940 in tile A and 36..76 in tile B.
        # Both map back to exactly 900..940, IoU 1.0, so one detection survives.
        merged = merge_tile_detections(
            [
                ((0, 0), [det(900, 100, 940, 140, 0.72)]),
                ((864, 0), [det(36, 100, 76, 140, 0.81)]),
            ],
            iou_threshold=0.5,
        )
        assert len(merged) == 1
        assert merged[0].bbox.as_xyxy() == (900.0, 100.0, 940.0, 140.0)
        assert merged[0].score == 0.81  # the more confident view wins

    def test_slightly_misaligned_seam_duplicates_still_merge(self):
        # The same patch rarely lands on identical pixels in both tiles. Scene box A
        # is 900..940, scene box B is 902..942: intersection 38x40=1520, union
        # 40*40*2-1520=1680, IoU = 0.9048, comfortably above threshold.
        a = det(900, 100, 940, 140, 0.7)
        b_local = det(38, 100, 78, 140, 0.6)  # +864 -> 902..942
        merged = merge_tile_detections([((0, 0), [a]), ((864, 0), [b_local])], iou_threshold=0.5)
        assert len(merged) == 1
        assert merged[0].score == 0.7

    def test_distinct_patches_in_the_overlap_region_both_survive(self):
        # Two genuinely different objects both visible in the seam must not be merged.
        merged = merge_tile_detections(
            [
                ((0, 0), [det(900, 100, 940, 140, 0.9)]),
                ((864, 0), [det(36, 400, 76, 440, 0.8)]),  # -> 900..940 but y 400..440
            ],
            iou_threshold=0.5,
        )
        assert len(merged) == 2

    def test_cross_tile_dedup_is_class_aware(self):
        merged = merge_tile_detections(
            [
                ((0, 0), [det(900, 100, 940, 140, 0.9, SurfaceClass.DEBRIS)]),
                ((864, 0), [det(36, 100, 76, 140, 0.8, SurfaceClass.SARGASSUM)]),
            ],
            iou_threshold=0.5,
        )
        assert len(merged) == 2
        merged_agnostic = merge_tile_detections(
            [
                ((0, 0), [det(900, 100, 940, 140, 0.9, SurfaceClass.DEBRIS)]),
                ((864, 0), [det(36, 100, 76, 140, 0.8, SurfaceClass.SARGASSUM)]),
            ],
            iou_threshold=0.5,
            class_agnostic=True,
        )
        assert len(merged_agnostic) == 1

    def test_scene_size_clips_boxes_pushed_past_the_edge(self):
        merged = merge_tile_detections(
            [((900, 0), [det(50, 10, 200, 60, 0.9)])],  # -> x 950..1100 in a 1000 wide scene
            iou_threshold=0.5,
            scene_size=(1000, 500),
        )
        assert len(merged) == 1
        assert merged[0].bbox.as_xyxy() == (950.0, 10.0, 1000.0, 60.0)

    def test_empty_tiles_contribute_nothing(self):
        merged = merge_tile_detections([((0, 0), []), ((960, 0), [det(0, 0, 5, 5)])])
        assert len(merged) == 1

    def test_no_tiles_at_all(self):
        assert merge_tile_detections([]) == []

    def test_result_is_score_ordered(self):
        merged = merge_tile_detections(
            [
                ((0, 0), [det(0, 0, 10, 10, 0.3)]),
                ((100, 0), [det(0, 0, 10, 10, 0.9)]),
                ((200, 0), [det(0, 0, 10, 10, 0.6)]),
            ]
        )
        assert [d.score for d in merged] == [0.9, 0.6, 0.3]

    def test_default_threshold_comes_from_settings(self):
        from mdebris.config import settings

        # IoU 0.6667 with the default 0.5 threshold: suppressed.
        assert settings.nms_iou_threshold == 0.5
        merged = merge_tile_detections(
            [((0, 0), [det(0, 0, 10, 10, 0.9)]), ((2, 0), [det(0, 0, 10, 10, 0.8)])]
        )
        assert len(merged) == 1

    def test_source_tiles_are_not_mutated(self):
        original = det(10, 10, 20, 20)
        merge_tile_detections([((500, 500), [original])])
        assert original.bbox.as_xyxy() == (10.0, 10.0, 20.0, 20.0)


# --------------------------------------------------------------------------------------
# image coercion
# --------------------------------------------------------------------------------------


class TestAsUint8RGB:
    def test_uint8_rgb_passes_through_unchanged(self):
        img = np.full((4, 5, 3), 200, dtype=np.uint8)
        out = as_uint8_rgb(img)
        assert out.dtype == np.uint8
        assert out.shape == (4, 5, 3)
        np.testing.assert_array_equal(out, img)

    def test_grayscale_is_replicated_to_three_channels(self):
        out = as_uint8_rgb(np.full((4, 5), 77, dtype=np.uint8))
        assert out.shape == (4, 5, 3)
        assert (out == 77).all()

    def test_rgba_alpha_is_dropped(self):
        img = np.zeros((3, 3, 4), dtype=np.uint8)
        img[..., 3] = 255
        assert as_uint8_rgb(img).shape == (3, 3, 3)

    def test_float_reflectance_in_0_1_is_scaled_to_0_255(self):
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = as_uint8_rgb(img)
        assert out.dtype == np.uint8
        assert out[0, 0].tolist() == [0, 127, 255]

    def test_float_already_in_0_255_is_not_scaled_again(self):
        img = np.full((2, 2, 3), 128.0, dtype=np.float32)
        assert (as_uint8_rgb(img) == 128).all()

    def test_nan_becomes_zero_rather_than_undefined(self):
        img = np.array([[[np.nan, 0.5, 1.0]]], dtype=np.float32)
        assert as_uint8_rgb(img)[0, 0, 0] == 0

    def test_uint16_sentinel_data_is_rescaled_to_8_bit(self):
        img = np.full((2, 2, 3), 65535, dtype=np.uint16)
        assert (as_uint8_rgb(img) == 255).all()

    def test_wrong_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match="channels"):
            as_uint8_rgb(np.zeros((4, 4, 7), dtype=np.uint8))

    def test_four_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            as_uint8_rgb(np.zeros((1, 4, 4, 3), dtype=np.uint8))

    def test_output_is_contiguous_for_torch(self):
        # torch.from_numpy on a non-contiguous view is a common source of surprises.
        assert as_uint8_rgb(np.zeros((4, 4, 4), dtype=np.uint8)).flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------------------
# BaseDetector contract
# --------------------------------------------------------------------------------------


class FakeDetector(BaseDetector):
    """A detector that counts loads and returns a fixed box, with no weights at all."""

    name = "fake"

    def __init__(self, **kwargs):
        super().__init__("fake/model", **kwargs)
        self.load_calls = 0
        self.seen_dtype: object = None

    def _load(self) -> None:
        self.load_calls += 1
        self.model = object()
        self.processor = object()

    def _detect(self, image, threshold):
        self.seen_dtype = image.dtype
        return [
            det(0, 0, 10, 10, 0.4),
            det(50, 50, 60, 60, 0.95),
        ]


class TestBaseDetector:
    def test_constructing_does_not_load_weights(self):
        d = FakeDetector()
        assert d.model is None
        assert d.processor is None
        assert d.is_loaded is False
        assert d.load_calls == 0

    def test_first_detect_triggers_exactly_one_load(self):
        d = FakeDetector()
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        d.detect(img)
        d.detect(img)
        d.detect(img)
        assert d.load_calls == 1
        assert d.is_loaded

    def test_explicit_load_is_idempotent(self):
        d = FakeDetector()
        d.load()
        d.load()
        assert d.load_calls == 1

    def test_unload_allows_reload(self):
        d = FakeDetector()
        d.load()
        d.unload()
        assert not d.is_loaded
        d.load()
        assert d.load_calls == 2

    def test_detect_returns_score_ordered_results(self):
        d = FakeDetector()
        out = d.detect(np.zeros((20, 20, 3), dtype=np.uint8))
        assert [x.score for x in out] == [0.95, 0.4]

    def test_detect_coerces_the_image_before_the_subclass_sees_it(self):
        d = FakeDetector()
        d.detect(np.zeros((20, 20), dtype=np.float32))
        assert d.seen_dtype == np.uint8

    @pytest.mark.parametrize("bad", [-0.5, 1.5])
    def test_invalid_threshold_is_rejected_before_loading(self, bad):
        d = FakeDetector()
        with pytest.raises(ValueError, match="threshold"):
            d.detect(np.zeros((4, 4, 3), dtype=np.uint8), threshold=bad)
        assert d.load_calls == 0

    def test_detect_batch_defaults_to_a_loop(self):
        d = FakeDetector()
        out = d.detect_batch([np.zeros((8, 8, 3), dtype=np.uint8)] * 3)
        assert len(out) == 3
        assert all(len(r) == 2 for r in out)

    def test_satisfies_the_detector_protocol(self):
        assert isinstance(FakeDetector(), Detector)

    def test_repr_reports_load_state_without_loading(self):
        d = FakeDetector()
        assert "lazy" in repr(d)
        d.load()
        assert "loaded" in repr(d)

    def test_load_failure_is_wrapped_in_model_load_error(self):
        from mdebris.models.base import ModelLoadError

        class Broken(FakeDetector):
            def _load(self) -> None:
                raise OSError("no such file")

        with pytest.raises(ModelLoadError, match="mdebris\\[models\\]"):
            Broken().load()

    def test_explicit_device_is_not_overridden_by_settings(self):
        assert FakeDetector(device="cpu").device == "cpu"
