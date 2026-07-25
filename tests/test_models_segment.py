"""Tests for SAM2 mask refinement and the RT-DETRv2 supervised path.

The offline tests cover the parts that do not need weights: lazy construction, the
box format conversion the training loop depends on, and the label mapping. The
marked tests run real forward passes, including a two-step training smoke test that
proves the fine-tuning loop is wired correctly rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdebris.models.base import ModelLoadError
from mdebris.models.supervised import (
    FINETUNE_CLASSES,
    ChipDataset,
    LabelledChip,
    RTDetrDetector,
    _xyxy_to_cxcywh_normalized,
)
from mdebris.types import BBox, Detection, SurfaceClass


# --------------------------------------------------------------------------------------
# Sam2Segmenter: offline
# --------------------------------------------------------------------------------------


class TestSam2Offline:
    def test_constructing_does_not_load_weights(self):
        from mdebris.models.segment import Sam2Segmenter

        seg = Sam2Segmenter()
        assert seg.model is None
        assert seg.processor is None
        assert seg.is_loaded is False

    def test_defaults_to_the_configured_checkpoint(self):
        from mdebris.config import settings
        from mdebris.models.segment import Sam2Segmenter

        assert Sam2Segmenter().model_id == settings.segment_model
        assert settings.segment_model == "facebook/sam2.1-hiera-tiny"

    def test_refining_nothing_is_a_no_op_that_does_not_load(self):
        # The pipeline calls refine unconditionally when mask refinement is enabled,
        # so an empty tile must not pay the weight-loading cost.
        from mdebris.models.segment import Sam2Segmenter

        seg = Sam2Segmenter()
        assert seg.refine(np.zeros((16, 16, 3), dtype=np.uint8), []) == []
        assert seg.is_loaded is False

    def test_nothing_in_the_package_imports_sam2_eagerly(self):
        """Mask refinement is optional, so importing the package must not need it."""
        import subprocess
        import sys

        code = (
            "import sys; import mdebris.models; "
            "bad = [m for m in sys.modules if 'sam2' in m.lower()]; "
            "assert not bad, bad; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_unloadable_weights_raise_an_actionable_error(self):
        from mdebris.models.segment import Sam2Segmenter

        seg = Sam2Segmenter("definitely/not-a-real-checkpoint-xyz")
        with pytest.raises(ModelLoadError) as excinfo:
            seg.load()
        message = str(excinfo.value)
        # The message must say what to do, not merely that something failed.
        assert "optional" in message
        assert "MDEBRIS_SEGMENT_MODEL" in message

    def test_is_not_a_detector(self):
        # It refines existing boxes; it cannot originate detections, and conflating
        # the two would let it be used as a detector in the pipeline.
        from mdebris.models.segment import Sam2Segmenter

        assert not hasattr(Sam2Segmenter(), "detect")

    def test_repr_reports_lazy_state(self):
        from mdebris.models.segment import Sam2Segmenter

        assert "lazy" in repr(Sam2Segmenter())


# --------------------------------------------------------------------------------------
# Supervised: offline
# --------------------------------------------------------------------------------------


class TestBoxConversion:
    """The xyxy -> normalized cxcywh conversion RT-DETR's loss depends on.

    A silent error here trains the model on boxes in the wrong place, which shows up
    only as a model that never converges.
    """

    def test_centre_box_converts_to_the_expected_values(self):
        boxes = np.array([[100.0, 50.0, 300.0, 150.0]])  # 200x100 box centred at (200, 100)
        out = _xyxy_to_cxcywh_normalized(boxes, width=400, height=200)
        # cx = 200/400 = 0.5, cy = 100/200 = 0.5, w = 200/400 = 0.5, h = 100/200 = 0.5
        np.testing.assert_allclose(out, [[0.5, 0.5, 0.5, 0.5]])

    def test_off_centre_box(self):
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]])
        out = _xyxy_to_cxcywh_normalized(boxes, width=1000, height=500)
        # cx = 50/1000 = 0.05, cy = 50/500 = 0.1, w = 0.1, h = 0.2
        np.testing.assert_allclose(out, [[0.05, 0.1, 0.1, 0.2]])

    def test_full_frame_box(self):
        out = _xyxy_to_cxcywh_normalized(np.array([[0.0, 0.0, 640.0, 640.0]]), 640, 640)
        np.testing.assert_allclose(out, [[0.5, 0.5, 1.0, 1.0]])

    def test_all_outputs_are_within_the_unit_square(self):
        rng = np.random.default_rng(0)
        xy = np.sort(rng.uniform(0, 640, size=(20, 2)), axis=1)
        wh = np.sort(rng.uniform(0, 480, size=(20, 2)), axis=1)
        boxes = np.stack([xy[:, 0], wh[:, 0], xy[:, 1], wh[:, 1]], axis=1)
        out = _xyxy_to_cxcywh_normalized(boxes, 640, 480)
        assert (out >= 0).all() and (out <= 1).all()

    def test_empty_input_keeps_the_shape(self):
        out = _xyxy_to_cxcywh_normalized(np.zeros((0, 4)), 640, 640)
        assert out.shape == (0, 4)

    def test_result_is_float32_as_the_loss_expects(self):
        out = _xyxy_to_cxcywh_normalized(np.array([[0.0, 0.0, 10.0, 10.0]]), 100, 100)
        assert out.dtype == np.float32

    def test_normalization_is_invariant_to_a_uniform_rescale(self):
        # This is why normalizing by the ORIGINAL chip size is correct even though
        # the processor resizes to 640x640: a plain stretch does not change
        # normalized coordinates. If the processor ever starts padding instead,
        # this invariant is what breaks.
        boxes = np.array([[10.0, 20.0, 110.0, 220.0]])
        small = _xyxy_to_cxcywh_normalized(boxes, 400, 400)
        big = _xyxy_to_cxcywh_normalized(boxes * 1.6, 640, 640)
        np.testing.assert_allclose(small, big, rtol=1e-6)


class TestLabelledChip:
    def test_boxes_and_labels_must_have_matching_lengths(self):
        with pytest.raises(ValueError, match="boxes but"):
            LabelledChip(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                boxes=np.array([[0, 0, 4, 4], [1, 1, 5, 5]]),
                labels=np.array([0]),
            )

    def test_arrays_are_coerced_to_the_expected_dtypes(self):
        chip = LabelledChip(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            boxes=[[0, 0, 4, 4]],
            labels=[0],
        )
        assert chip.boxes.dtype == np.float32
        assert chip.labels.dtype == np.int64
        assert chip.boxes.shape == (1, 4)

    def test_empty_chip_is_allowed_as_a_negative_example(self):
        # Chips with no debris are valuable training signal, not an error.
        chip = LabelledChip(
            image=np.zeros((8, 8, 3), dtype=np.uint8), boxes=np.zeros((0, 4)), labels=[]
        )
        assert len(chip.boxes) == 0

    def test_from_detections_maps_classes_to_head_indices(self):
        dets = [
            Detection(bbox=BBox(0, 0, 10, 10), score=0.9, label=SurfaceClass.DEBRIS),
            Detection(bbox=BBox(20, 20, 30, 30), score=0.8, label=SurfaceClass.SHIP),
        ]
        chip = LabelledChip.from_detections(np.zeros((40, 40, 3), dtype=np.uint8), dets)
        assert chip.labels.tolist() == [
            FINETUNE_CLASSES.index(SurfaceClass.DEBRIS),
            FINETUNE_CLASSES.index(SurfaceClass.SHIP),
        ]
        np.testing.assert_allclose(chip.boxes[0], [0, 0, 10, 10])

    def test_from_detections_drops_classes_outside_the_head(self):
        # open_water is not an object; mapping it to a catch-all would teach the
        # model that empty ocean is a thing to detect.
        dets = [
            Detection(bbox=BBox(0, 0, 10, 10), score=0.9, label=SurfaceClass.DEBRIS),
            Detection(bbox=BBox(0, 0, 40, 40), score=0.5, label=SurfaceClass.WATER),
            Detection(bbox=BBox(5, 5, 15, 15), score=0.5, label=SurfaceClass.UNKNOWN),
        ]
        chip = LabelledChip.from_detections(np.zeros((40, 40, 3), dtype=np.uint8), dets)
        assert len(chip.boxes) == 1

    def test_finetune_classes_exclude_non_objects(self):
        assert SurfaceClass.WATER not in FINETUNE_CLASSES
        assert SurfaceClass.UNKNOWN not in FINETUNE_CLASSES
        assert FINETUNE_CLASSES[0] is SurfaceClass.DEBRIS


class TestChipDataset:
    def test_supports_the_map_style_dataset_protocol(self):
        chip = LabelledChip(
            image=np.zeros((8, 8, 3), dtype=np.uint8), boxes=np.zeros((0, 4)), labels=[]
        )
        ds = ChipDataset([chip, chip, chip])
        assert len(ds) == 3
        assert ds[1] is chip

    def test_is_not_a_torch_subclass_so_torch_stays_out_of_the_import_path(self):
        import subprocess
        import sys

        code = (
            "import sys; from mdebris.models.supervised import ChipDataset, FINETUNE_CLASSES; "
            "assert 'torch' not in sys.modules, 'supervised.py imported torch'; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


class TestRTDetrOffline:
    def test_constructing_does_not_load_weights(self):
        det = RTDetrDetector()
        assert det.model is None
        assert det.is_loaded is False

    def test_defaults_to_the_configured_checkpoint(self):
        from mdebris.config import settings

        assert RTDetrDetector().model_id == settings.supervised_model

    def test_coco_boat_maps_to_ship(self):
        # The pretrained checkpoint is COCO-trained, so before fine-tuning the only
        # thing it can usefully say about the ocean is where the boats are.
        det = RTDetrDetector()
        mapping = det._build_label_map({0: "person", 8: "boat", 2: "car"})
        assert mapping[8] is SurfaceClass.SHIP
        assert mapping[0] is SurfaceClass.UNKNOWN
        assert mapping[2] is SurfaceClass.UNKNOWN

    def test_a_finetuned_checkpoints_own_labels_round_trip(self):
        det = RTDetrDetector()
        mapping = det._build_label_map({0: "marine_debris", 1: "sargassum", 2: "ship"})
        assert mapping[0] is SurfaceClass.DEBRIS
        assert mapping[1] is SurfaceClass.SARGASSUM
        assert mapping[2] is SurfaceClass.SHIP

    def test_explicit_label_map_overrides_the_defaults(self):
        det = RTDetrDetector(label_map={"boat": SurfaceClass.WAKE})
        assert det._build_label_map({8: "boat"})[8] is SurfaceClass.WAKE

    def test_finetune_rejects_empty_training_data(self):
        with pytest.raises(ValueError, match="at least one labelled chip"):
            RTDetrDetector().finetune([])

    def test_finetune_validates_before_loading_weights(self):
        det = RTDetrDetector()
        with pytest.raises(ValueError):
            det.finetune([])
        assert det.is_loaded is False


# --------------------------------------------------------------------------------------
# Real weights. Excluded from the default run.
# --------------------------------------------------------------------------------------


def _debris_scene(height: int = 240, width: int = 320) -> np.ndarray:
    """A small ocean-like chip with two bright patches at known locations."""
    rng = np.random.default_rng(0)
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 14, 52, 96
    img += rng.normal(0, 4, img.shape)
    yy, xx = np.mgrid[0:height, 0:width]
    for cx, cy, r in ((80, 70, 26), (230, 170, 20)):
        blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        img[blob & (rng.random((height, width)) < 0.75)] = 232
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.slow
@pytest.mark.network
class TestSam2Real:
    def test_refine_sets_masks_and_records_the_refined_area(self):
        from mdebris.models.segment import Sam2Segmenter

        scene = _debris_scene()
        height, width = scene.shape[:2]
        dets = [
            Detection(bbox=BBox(54, 44, 106, 96), score=0.8, label=SurfaceClass.DEBRIS),
            Detection(bbox=BBox(210, 150, 250, 190), score=0.7, label=SurfaceClass.DEBRIS),
        ]
        seg = Sam2Segmenter()
        refined = seg.refine(scene, dets)

        assert len(refined) == 2
        for det in refined:
            assert det.mask is not None
            assert det.mask.dtype == bool
            # Masks are full-frame, aligned to the input image, not crop-relative.
            assert det.mask.shape == (height, width)
            assert det.mask.any(), "SAM2 returned an entirely empty mask"

            pixels = det.indices["mask_pixels"]
            assert pixels == pytest.approx(float(det.mask.sum()))
            assert 0 < pixels <= height * width

            # A mask inside a box cannot cover more than the box, so fill is in [0, 1].
            assert 0.0 < det.indices["mask_fill"] <= 1.0
            assert 0.0 <= det.indices["mask_iou"] <= 1.0

    def test_refine_mutates_in_place_and_returns_the_same_objects(self):
        from mdebris.models.segment import Sam2Segmenter

        scene = _debris_scene()
        det = Detection(bbox=BBox(54, 44, 106, 96), score=0.8, label=SurfaceClass.DEBRIS)
        returned = Sam2Segmenter().refine(scene, [det])
        assert returned[0] is det
        assert det.mask is not None

    def test_mask_area_is_smaller_than_the_box_for_a_round_blob(self):
        """The reason this module exists, stated as a test.

        A circle inscribed in its bounding box covers pi/4 = 78.5% of it. Any mask
        that tracks the blob rather than the box must come in under the box area,
        which is exactly the correction that makes an area estimate meaningful.
        """
        from mdebris.models.segment import Sam2Segmenter

        scene = _debris_scene()
        det = Detection(bbox=BBox(54, 44, 106, 96), score=0.8, label=SurfaceClass.DEBRIS)
        Sam2Segmenter().refine(scene, [det])
        assert det.indices["mask_fill"] < 0.95, det.indices


@pytest.mark.slow
@pytest.mark.network
class TestRTDetrReal:
    def test_inference_returns_well_formed_detections(self):
        scene = _debris_scene()
        height, width = scene.shape[:2]
        det = RTDetrDetector()
        found = det.detect(scene, threshold=0.3)

        for d in found:
            assert 0.0 <= d.score <= 1.0
            assert 0 <= d.bbox.xmin < d.bbox.xmax <= width
            assert 0 <= d.bbox.ymin < d.bbox.ymax <= height
            assert d.source_model.startswith("rtdetr-v2:")
        assert [d.score for d in found] == sorted((d.score for d in found), reverse=True)

    def test_pretrained_checkpoint_exposes_the_coco_boat_class(self):
        det = RTDetrDetector()
        det.load()
        assert SurfaceClass.SHIP in det._id_to_class.values()

    def test_processor_resizes_without_padding(self):
        """The assumption the box normalization relies on.

        ``_xyxy_to_cxcywh_normalized`` normalizes by the original chip size because
        RT-DETR's processor does a plain stretch to a fixed square. If it ever starts
        padding to preserve aspect ratio, normalized coordinates stop being invariant
        and every training box silently shifts. This test is the tripwire.
        """
        det = RTDetrDetector()
        det.load()
        wide = np.zeros((100, 400, 3), dtype=np.uint8)
        tall = np.zeros((400, 100, 3), dtype=np.uint8)
        wide_out = det.processor(images=wide, return_tensors="pt")["pixel_values"]
        tall_out = det.processor(images=tall, return_tensors="pt")["pixel_values"]
        # Both aspect ratios land on the same square with no pixel_mask, which is
        # only possible with a plain stretch.
        assert wide_out.shape == tall_out.shape
        assert wide_out.shape[-2] == wide_out.shape[-1]

    def test_finetune_runs_two_real_steps_and_reduces_loss(self):
        """Smoke test that the training loop is genuinely wired, not merely plausible.

        Two optimizer steps on four tiny synthetic chips, on CPU. This proves the
        label format, the loss, the backward pass, the optimizer and the scheduler
        are all connected. It proves nothing about whether the model would converge
        on real data, which needs a GPU and a labelled dataset.
        """
        rng = np.random.default_rng(0)
        chips = []
        for _ in range(4):
            image = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
            chips.append(
                LabelledChip(
                    image=image,
                    boxes=np.array([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]]),
                    labels=np.array([0, 1]),
                )
            )

        det = RTDetrDetector(num_labels=len(FINETUNE_CLASSES))
        history = det.finetune(
            ChipDataset(chips),
            epochs=1,
            batch_size=2,
            max_steps=2,
            lr=1e-4,
            log_every=1,
        )

        assert history["steps"] == 2
        assert len(history["train_loss"]) == 2
        for entry in history["train_loss"]:
            loss = entry["loss"]
            assert np.isfinite(loss), f"non-finite loss: {loss}"
            assert loss > 0
        assert history["config"]["train_chips"] == 4
        assert history["seconds"] > 0
        # Training must leave the model in eval mode, ready for inference.
        assert det.model.training is False

    def test_finetune_rehead_changes_the_class_space(self):
        det = RTDetrDetector(num_labels=len(FINETUNE_CLASSES))
        det.load()
        assert det.model.config.num_labels == len(FINETUNE_CLASSES)
        # The re-headed model labels its classes with our own enum values, so the
        # inference path maps them back without a hand-written table.
        assert det._id_to_class[0] is SurfaceClass.DEBRIS

    def test_finetune_saves_a_loadable_checkpoint(self, tmp_path):
        rng = np.random.default_rng(1)
        chips = [
            LabelledChip(
                image=rng.integers(0, 255, (96, 96, 3), dtype=np.uint8),
                boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
                labels=np.array([0]),
            )
        ]
        det = RTDetrDetector(num_labels=len(FINETUNE_CLASSES))
        out = tmp_path / "finetuned"
        det.finetune(chips, epochs=1, batch_size=1, max_steps=1, output_dir=out)

        assert (out / "config.json").exists()
        assert (out / "history.json").exists()
        # The saved checkpoint must be loadable back through the normal path.
        reloaded = RTDetrDetector(str(out))
        reloaded.load()
        assert reloaded.model.config.num_labels == len(FINETUNE_CLASSES)
        assert reloaded._id_to_class[0] is SurfaceClass.DEBRIS

    def test_finetune_reports_validation_loss(self):
        rng = np.random.default_rng(2)

        def chip():
            return LabelledChip(
                image=rng.integers(0, 255, (96, 96, 3), dtype=np.uint8),
                boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
                labels=np.array([0]),
            )

        det = RTDetrDetector(num_labels=len(FINETUNE_CLASSES))
        history = det.finetune(
            [chip()], val_data=[chip()], epochs=1, batch_size=1, max_steps=1
        )
        assert len(history["val_loss"]) == 1
        assert np.isfinite(history["val_loss"][0]["loss"])
