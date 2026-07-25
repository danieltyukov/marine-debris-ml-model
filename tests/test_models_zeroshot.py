"""Tests for prompt sets and the open-vocabulary detectors.

Everything outside the ``slow``/``network`` marked classes runs offline with no
weights: prompt-to-class mapping is pure data, and lazy loading is verified by
asserting the model attribute is still None after construction. The marked tests
run real forward passes and are excluded from the default ``pytest`` run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdebris.models.prompts import (
    CONFUSER_PROMPTS,
    DEFAULT_PROMPTS,
    MINIMAL_PROMPTS,
    PROMPT_SETS,
    TARGET_PROMPTS,
    PromptSet,
    get_prompt_set,
)
from mdebris.types import Detection, Detector, SurfaceClass

ASSET_SCENE = Path(__file__).resolve().parents[1] / "assets" / "detections_geo.png"


# --------------------------------------------------------------------------------------
# PromptSet: offline
# --------------------------------------------------------------------------------------


class TestPromptSet:
    def test_prompts_map_to_the_declared_class(self):
        ps = PromptSet.build(
            ["floating plastic debris"],
            {"a mat of floating seaweed": SurfaceClass.SARGASSUM, "a ship": SurfaceClass.SHIP},
        )
        assert ps.label_for_text("floating plastic debris") is SurfaceClass.DEBRIS
        assert ps.label_for_text("a mat of floating seaweed") is SurfaceClass.SARGASSUM
        assert ps.label_for_text("a ship") is SurfaceClass.SHIP

    def test_texts_and_labels_are_index_aligned(self):
        # OWLv2 returns a label as an index into the list it was given, so this
        # alignment is the contract that makes the output interpretable at all.
        ps = DEFAULT_PROMPTS
        assert len(ps.texts) == len(ps.labels) == len(ps)
        for i, text in enumerate(ps.texts):
            assert ps.label_for_index(i) is ps.labels[i]
            assert ps.label_for_text(text) is ps.labels[i]

    def test_index_lookup_matches_declaration_order(self):
        ps = PromptSet.build(["debris a", "debris b"], {"a ship": SurfaceClass.SHIP})
        assert ps.texts == ["debris a", "debris b", "a ship"]
        assert ps.label_for_index(0) is SurfaceClass.DEBRIS
        assert ps.label_for_index(1) is SurfaceClass.DEBRIS
        assert ps.label_for_index(2) is SurfaceClass.SHIP

    @pytest.mark.parametrize("bad_index", [-1, 99])
    def test_out_of_range_index_degrades_to_unknown(self, bad_index):
        # A surprising index must not abort a scene-wide run.
        assert DEFAULT_PROMPTS.label_for_index(bad_index) is SurfaceClass.UNKNOWN

    def test_text_lookup_is_case_and_whitespace_insensitive(self):
        assert DEFAULT_PROMPTS.label_for_text("  A SHIP  ") is SurfaceClass.SHIP

    def test_partial_phrase_maps_back_to_its_prompt(self):
        # GroundingDINO emits fragments of the caption rather than whole prompts.
        assert DEFAULT_PROMPTS.label_for_text("floating plastic") is SurfaceClass.DEBRIS
        assert DEFAULT_PROMPTS.label_for_text("floating seaweed") is SurfaceClass.SARGASSUM

    def test_longest_matching_prompt_wins(self):
        # "a ship" is a substring of "a shipwreck on the shore"; the longer, more
        # specific prompt must win so a short generic prompt cannot shadow it.
        ps = PromptSet.build(
            ["floating plastic debris"],
            {"a ship": SurfaceClass.SHIP, "a shipwreck on the shore": SurfaceClass.UNKNOWN},
        )
        assert ps.label_for_text("a shipwreck on the shore") is SurfaceClass.UNKNOWN
        assert ps.label_for_text("a ship") is SurfaceClass.SHIP

    def test_unmatched_phrase_is_unknown(self):
        assert DEFAULT_PROMPTS.label_for_text("a giraffe") is SurfaceClass.UNKNOWN

    def test_empty_phrase_is_unknown_not_a_wildcard_match(self):
        # "" is a substring of every prompt, so a naive containment check would
        # match the first one.
        assert DEFAULT_PROMPTS.label_for_text("") is SurfaceClass.UNKNOWN

    def test_default_set_includes_a_confuser_for_each_dominant_failure_mode(self):
        classes = set(DEFAULT_PROMPTS.labels)
        for expected in (
            SurfaceClass.SARGASSUM,
            SurfaceClass.SHIP,
            SurfaceClass.WAKE,
            SurfaceClass.FOAM,
            SurfaceClass.CLOUD,
            SurfaceClass.WATER,
        ):
            assert expected in classes, f"{expected} missing from the default prompt set"

    def test_default_set_has_more_confusers_than_targets(self):
        # The design claim is that competing hypotheses do the work, so the confuser
        # side must not be a token gesture.
        targets = sum(1 for c in DEFAULT_PROMPTS.labels if c.is_target)
        assert len(DEFAULT_PROMPTS) - targets > targets

    def test_target_prompts_all_map_to_debris(self):
        for prompt in TARGET_PROMPTS:
            assert DEFAULT_PROMPTS.label_for_text(prompt) is SurfaceClass.DEBRIS

    def test_confuser_prompts_never_map_to_debris(self):
        for prompt, cls in CONFUSER_PROMPTS.items():
            assert not cls.is_target, f"{prompt!r} is declared as a target"
            assert DEFAULT_PROMPTS.label_for_text(prompt) is cls

    def test_targets_only_drops_every_confuser(self):
        reduced = DEFAULT_PROMPTS.targets_only()
        assert len(reduced) < len(DEFAULT_PROMPTS)
        assert all(c.is_target for c in reduced.labels)

    def test_with_prompts_appends_and_skips_duplicates(self):
        extended = MINIMAL_PROMPTS.with_prompts(
            {"an oil slick": SurfaceClass.UNKNOWN, "a ship": SurfaceClass.SHIP}
        )
        assert len(extended) == len(MINIMAL_PROMPTS) + 1
        assert extended.label_for_text("an oil slick") is SurfaceClass.UNKNOWN

    def test_dot_string_format_for_grounding_dino(self):
        ps = PromptSet.build(["Floating Plastic Debris"], {"A Ship": SurfaceClass.SHIP})
        # Lowercase, ". " separated, single trailing period.
        assert ps.as_dot_string() == "floating plastic debris. a ship."

    def test_dot_string_does_not_double_the_period(self):
        ps = PromptSet.build(["floating debris."], {"a ship.": SurfaceClass.SHIP})
        assert ps.as_dot_string() == "floating debris. a ship."
        assert not ps.as_dot_string().endswith("..")

    def test_from_mapping_preserves_order(self):
        ps = PromptSet.from_mapping(
            {"floating debris": SurfaceClass.DEBRIS, "a ship": SurfaceClass.SHIP}
        )
        assert ps.texts == ["floating debris", "a ship"]

    def test_empty_prompt_set_is_rejected(self):
        with pytest.raises(ValueError, match="at least one prompt"):
            PromptSet(entries=())

    def test_duplicate_prompts_are_rejected(self):
        # Two identical prompts make the index-to-class mapping ambiguous.
        with pytest.raises(ValueError, match="duplicate"):
            PromptSet(
                entries=(
                    ("a ship", SurfaceClass.SHIP),
                    ("a ship", SurfaceClass.WAKE),
                )
            )

    def test_prompt_set_with_no_target_is_rejected(self):
        with pytest.raises(ValueError, match="never detect debris"):
            PromptSet(entries=(("a ship", SurfaceClass.SHIP),))

    def test_prompt_set_is_hashable_so_encodings_can_be_cached(self):
        assert hash(DEFAULT_PROMPTS) == hash(DEFAULT_PROMPTS)
        assert len({DEFAULT_PROMPTS, DEFAULT_PROMPTS}) == 1

    def test_named_sets_are_resolvable(self):
        assert get_prompt_set("default") is DEFAULT_PROMPTS
        assert get_prompt_set("minimal") is MINIMAL_PROMPTS
        assert set(PROMPT_SETS) == {"default", "minimal", "targets-only"}

    def test_unknown_set_name_lists_the_available_ones(self):
        with pytest.raises(KeyError, match="minimal"):
            get_prompt_set("nope")


# --------------------------------------------------------------------------------------
# Lazy loading: offline, must not touch the network
# --------------------------------------------------------------------------------------


class TestLazyConstruction:
    """Constructing a detector must not download anything.

    ``mdebris --help`` and every fast test constructs detectors. If construction
    fetched weights, both would need the network.
    """

    def test_owlv2_constructs_without_loading(self):
        from mdebris.models.zeroshot import OWLv2Detector

        det = OWLv2Detector()
        assert det.model is None
        assert det.processor is None
        assert det.is_loaded is False

    def test_grounding_dino_constructs_without_loading(self):
        from mdebris.models.zeroshot import GroundingDinoDetector

        det = GroundingDinoDetector()
        assert det.model is None
        assert det.processor is None
        assert det.is_loaded is False

    def test_rtdetr_constructs_without_loading(self):
        from mdebris.models.supervised import RTDetrDetector

        det = RTDetrDetector()
        assert det.model is None
        assert det.is_loaded is False

    def test_sam2_constructs_without_loading(self):
        from mdebris.models.segment import Sam2Segmenter

        seg = Sam2Segmenter()
        assert seg.model is None
        assert seg.is_loaded is False

    def test_importing_the_package_does_not_import_torch(self):
        # Run in a clean interpreter: an earlier test in this session may already
        # have imported torch, so checking sys.modules in-process proves nothing.
        import subprocess
        import sys

        code = (
            "import sys; import mdebris.models; "
            "assert 'torch' not in sys.modules, 'mdebris.models imported torch'; "
            "assert 'transformers' not in sys.modules, 'mdebris.models imported transformers'; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_pure_helpers_are_importable_without_the_detectors(self):
        from mdebris.models import merge_tile_detections, nms

        assert callable(nms)
        assert callable(merge_tile_detections)

    def test_get_detector_resolves_by_name(self):
        from mdebris.models import get_detector
        from mdebris.models.zeroshot import OWLv2Detector

        det = get_detector("owlv2")
        assert isinstance(det, OWLv2Detector)
        assert det.model is None

    def test_get_detector_rejects_unknown_names(self):
        from mdebris.models import get_detector

        with pytest.raises(KeyError, match="owlv2"):
            get_detector("yolo")

    def test_detectors_satisfy_the_detector_protocol(self):
        from mdebris.models.supervised import RTDetrDetector
        from mdebris.models.zeroshot import GroundingDinoDetector, OWLv2Detector

        for cls in (OWLv2Detector, GroundingDinoDetector, RTDetrDetector):
            assert isinstance(cls(), Detector)

    def test_custom_prompt_set_is_kept(self):
        from mdebris.models.zeroshot import OWLv2Detector

        assert OWLv2Detector(prompts=MINIMAL_PROMPTS).prompts is MINIMAL_PROMPTS

    def test_model_id_defaults_to_settings(self):
        from mdebris.config import settings
        from mdebris.models.zeroshot import OWLv2Detector

        assert OWLv2Detector().model_id == settings.zeroshot_model


# --------------------------------------------------------------------------------------
# Result decoding: offline, with a stubbed forward pass
# --------------------------------------------------------------------------------------


class _FakeTensor:
    """Just enough of the torch tensor surface for the decoding path."""

    def __init__(self, data):
        self._data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._data


class TestOWLv2Decoding:
    """Exercise the box/label decoding without a forward pass.

    The decoding is where a wrong assumption about the transformers v5 output
    silently produces detections in the wrong place or with the wrong label, so it
    is worth testing separately from the model.
    """

    def _detector(self):
        from mdebris.models.zeroshot import OWLv2Detector

        return OWLv2Detector(prompts=MINIMAL_PROMPTS)

    def test_labels_come_from_the_matched_prompt(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[10, 10, 20, 20], [30, 30, 40, 40]]),
            "scores": _FakeTensor([0.8, 0.6]),
            "labels": _FakeTensor([0, 2]),
            "text_labels": ["floating plastic debris", "a ship"],
        }
        dets = det._results_to_detections(result, (100, 100))
        by_label = {d.label: d for d in dets}
        assert by_label[SurfaceClass.DEBRIS].score == pytest.approx(0.8)
        assert by_label[SurfaceClass.SHIP].score == pytest.approx(0.6)

    def test_falls_back_to_the_index_when_no_text_label_is_returned(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[10, 10, 20, 20]]),
            "scores": _FakeTensor([0.8]),
            "labels": _FakeTensor([3]),  # index 3 in MINIMAL_PROMPTS is "a ship"
            "text_labels": None,
        }
        assert det._results_to_detections(result, (100, 100))[0].label is SurfaceClass.SHIP

    def test_source_model_is_recorded_for_provenance(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[10, 10, 20, 20]]),
            "scores": _FakeTensor([0.8]),
            "labels": _FakeTensor([0]),
            "text_labels": ["floating plastic debris"],
        }
        assert det._results_to_detections(result, (100, 100))[0].source_model.startswith("owlv2:")

    def test_background_water_boxes_are_dropped_by_default(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[0, 0, 90, 90], [10, 10, 20, 20]]),
            "scores": _FakeTensor([0.5, 0.4]),
            "labels": _FakeTensor([5, 0]),
            "text_labels": ["open blue ocean water", "floating plastic debris"],
        }
        dets = det._results_to_detections(result, (100, 100))
        assert [d.label for d in dets] == [SurfaceClass.DEBRIS]

    def test_background_boxes_are_kept_when_asked(self):
        from mdebris.models.zeroshot import OWLv2Detector

        det = OWLv2Detector(prompts=MINIMAL_PROMPTS, drop_background=False)
        result = {
            "boxes": _FakeTensor([[0, 0, 90, 90]]),
            "scores": _FakeTensor([0.5]),
            "labels": _FakeTensor([5]),
            "text_labels": ["open blue ocean water"],
        }
        assert det._results_to_detections(result, (100, 100))[0].label is SurfaceClass.WATER

    def test_boxes_outside_the_frame_are_clipped(self):
        # OWLv2 pads to a square before resizing, so a box can legitimately be
        # returned partly outside the real image.
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[80, 80, 150, 150]]),
            "scores": _FakeTensor([0.8]),
            "labels": _FakeTensor([0]),
            "text_labels": ["floating plastic debris"],
        }
        dets = det._results_to_detections(result, (100, 100))
        assert dets[0].bbox.as_xyxy() == (80.0, 80.0, 100.0, 100.0)

    def test_boxes_entirely_in_the_pad_region_are_dropped(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[150, 150, 200, 200]]),
            "scores": _FakeTensor([0.8]),
            "labels": _FakeTensor([0]),
            "text_labels": ["floating plastic debris"],
        }
        assert det._results_to_detections(result, (100, 100)) == []

    def test_degenerate_boxes_are_skipped_not_raised_on(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[20, 20, 20, 20], [10, 10, 30, 30]]),
            "scores": _FakeTensor([0.9, 0.5]),
            "labels": _FakeTensor([0, 0]),
            "text_labels": ["floating plastic debris"] * 2,
        }
        dets = det._results_to_detections(result, (100, 100))
        assert len(dets) == 1

    def test_duplicate_boxes_are_suppressed_within_one_image(self):
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[10, 10, 30, 30], [11, 11, 31, 31]]),
            "scores": _FakeTensor([0.9, 0.8]),
            "labels": _FakeTensor([0, 0]),
            "text_labels": ["floating plastic debris"] * 2,
        }
        assert len(det._results_to_detections(result, (100, 100))) == 1

    def test_nms_can_be_disabled(self):
        from mdebris.models.zeroshot import OWLv2Detector

        det = OWLv2Detector(prompts=MINIMAL_PROMPTS, apply_nms=False)
        result = {
            "boxes": _FakeTensor([[10, 10, 30, 30], [11, 11, 31, 31]]),
            "scores": _FakeTensor([0.9, 0.8]),
            "labels": _FakeTensor([0, 0]),
            "text_labels": ["floating plastic debris"] * 2,
        }
        assert len(det._results_to_detections(result, (100, 100))) == 2

    def test_scores_slightly_over_one_are_clamped(self):
        # Detection validates score in [0, 1]; float drift at the sigmoid boundary
        # must not raise mid-scene.
        det = self._detector()
        result = {
            "boxes": _FakeTensor([[10, 10, 30, 30]]),
            "scores": _FakeTensor([1.0000001]),
            "labels": _FakeTensor([0]),
            "text_labels": ["floating plastic debris"],
        }
        assert det._results_to_detections(result, (100, 100))[0].score == 1.0


# --------------------------------------------------------------------------------------
# Real forward passes. Excluded from the default run.
# --------------------------------------------------------------------------------------


def _red_circle_scene() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A wide non-square image with an unambiguous object at a known location.

    Non-square on purpose: OWLv2 pads to a square before resizing, so a square test
    image would hide any mistake in the coordinate handling.
    """
    from PIL import Image, ImageDraw

    width, height = 900, 400
    img = Image.new("RGB", (width, height), (250, 250, 250))
    ImageDraw.Draw(img).ellipse([600, 120, 760, 280], fill=(220, 30, 30))
    return np.asarray(img), (600, 120, 760, 280)


@pytest.mark.slow
@pytest.mark.network
class TestOWLv2Real:
    def test_detects_a_known_object_at_the_right_place(self):
        """End-to-end geometry check on a non-square image.

        Uses an unmistakable object rather than synthetic debris: the point is to
        verify the coordinate handling and label plumbing, and a shape the model is
        certain about is the only way to isolate those from model uncertainty.
        """
        from mdebris.models.zeroshot import OWLv2Detector

        scene, (gx0, gy0, gx1, gy1) = _red_circle_scene()
        prompts = PromptSet.build(
            ["a red circle"], {"a white background": SurfaceClass.WATER}, name="test"
        )
        det = OWLv2Detector(prompts=prompts)
        found = det.detect(scene, threshold=0.3)

        assert found, "OWLv2 returned nothing for an unambiguous object"
        top = found[0]
        assert top.label is SurfaceClass.DEBRIS  # "a red circle" is the target prompt
        assert top.source_model == f"owlv2:{det.model_id}"
        assert 0.0 <= top.score <= 1.0

        # Box within 15 px of ground truth on every edge. If target_sizes were
        # handled wrongly for a padded non-square image, the box would be off by a
        # factor of 900/400, not by 15 px.
        b = top.bbox
        assert abs(b.xmin - gx0) < 15, b.as_xyxy()
        assert abs(b.ymin - gy0) < 15, b.as_xyxy()
        assert abs(b.xmax - gx1) < 15, b.as_xyxy()
        assert abs(b.ymax - gy1) < 15, b.as_xyxy()

    def test_all_detections_are_well_formed_and_inside_the_frame(self):
        from mdebris.models.zeroshot import OWLv2Detector

        scene, _ = _red_circle_scene()
        height, width = scene.shape[:2]
        det = OWLv2Detector(prompts=DEFAULT_PROMPTS)
        found = det.detect(scene, threshold=0.05)

        for d in found:
            assert isinstance(d, Detection)
            assert 0.0 <= d.score <= 1.0
            assert d.label in set(DEFAULT_PROMPTS.labels)
            assert 0 <= d.bbox.xmin < d.bbox.xmax <= width
            assert 0 <= d.bbox.ymin < d.bbox.ymax <= height
        # Score ordering is part of the public contract.
        assert [d.score for d in found] == sorted((d.score for d in found), reverse=True)

    def test_batch_matches_sequential_results(self):
        """A batched forward must produce the same detections as looping.

        Guards against a batch-dimension mix-up, where image 0's boxes are attributed
        to image 1. That failure is invisible on a single image.
        """
        from mdebris.models.zeroshot import OWLv2Detector

        scene, _ = _red_circle_scene()
        blank = np.full((400, 900, 3), 250, dtype=np.uint8)
        prompts = PromptSet.build(
            ["a red circle"], {"a white background": SurfaceClass.WATER}, name="test"
        )
        det = OWLv2Detector(prompts=prompts)

        batched = det.detect_batch([scene, blank], threshold=0.3)
        sequential = [det.detect(scene, threshold=0.3), det.detect(blank, threshold=0.3)]

        assert len(batched) == 2
        for got, want in zip(batched, sequential, strict=True):
            assert len(got) == len(want)
            for a, b in zip(got, want, strict=True):
                assert a.label is b.label
                assert a.score == pytest.approx(b.score, abs=1e-4)
                assert a.bbox.as_xyxy() == pytest.approx(b.bbox.as_xyxy(), abs=1.0)
        # The blank image must not inherit the circle from its batch neighbour.
        assert batched[1] == []

    @pytest.mark.skipif(not ASSET_SCENE.exists(), reason="reference scene asset not present")
    def test_confusers_change_the_label_on_real_imagery(self):
        """The scientific claim of prompts.py, measured on a real Planet scene.

        With confuser prompts present, bright non-debris features are labelled as
        what they are. Remove the confusers and the same boxes come back as debris,
        because the debris prompt becomes the only option on offer. That difference
        is the precision benefit the confuser design buys.
        """
        from PIL import Image

        from mdebris.models.zeroshot import OWLv2Detector

        # Right-hand panel of the reference figure: a real Planet ocean scene.
        full = np.asarray(Image.open(ASSET_SCENE).convert("RGB"))
        scene = full[19:569, 665:1451]

        with_confusers = OWLv2Detector(prompts=DEFAULT_PROMPTS)
        found = with_confusers.detect(scene, threshold=0.05)
        assert found, "no detections on the reference scene; thresholds may have drifted"

        targets_only = OWLv2Detector(prompts=DEFAULT_PROMPTS.targets_only())
        targets_only.model = with_confusers.model
        targets_only.processor = with_confusers.processor
        naive = targets_only.detect(scene, threshold=0.05)

        # Without confusers every detection is necessarily debris: there is no other
        # label available. That is the failure mode, stated as an assertion.
        assert naive, "targets-only run found nothing to compare"
        assert {d.label for d in naive} == {SurfaceClass.DEBRIS}

        # With confusers, at least one box that the naive run called debris is
        # assigned to a confuser class instead.
        reclassified = [
            d
            for d in found
            if not d.label.is_target
            and any(d.bbox.iou(n.bbox) > 0.5 for n in naive)
        ]
        assert reclassified, (
            "confusers did not reclassify any box: "
            f"with={[(str(d.label), round(d.score, 3)) for d in found]} "
            f"without={[(str(d.label), round(d.score, 3)) for d in naive]}"
        )


@pytest.mark.slow
@pytest.mark.network
class TestGroundingDinoReal:
    def test_detects_a_known_object_and_maps_the_phrase_to_a_class(self):
        """GroundingDINO's phrase output must land on the same enum as OWLv2's index."""
        from mdebris.models.zeroshot import GroundingDinoDetector

        scene, (gx0, gy0, gx1, gy1) = _red_circle_scene()
        prompts = PromptSet.build(
            ["a red circle"], {"a white background": SurfaceClass.WATER}, name="test"
        )
        det = GroundingDinoDetector(prompts=prompts)
        found = det.detect(scene, threshold=0.3)

        assert found, "GroundingDINO returned nothing for an unambiguous object"
        circles = [d for d in found if d.label is SurfaceClass.DEBRIS]
        assert circles, f"phrase mapping failed: {[str(d.label) for d in found]}"
        b = circles[0].bbox
        assert abs(b.xmin - gx0) < 20, b.as_xyxy()
        assert abs(b.ymax - gy1) < 20, b.as_xyxy()
        assert circles[0].source_model.startswith("grounding-dino:")
