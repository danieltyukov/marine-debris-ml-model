"""Box-prompted mask refinement with SAM2.

A bounding box around a debris patch overstates its extent, often badly. Debris
drifts in thin filaments and windrows, so an axis-aligned box around one can be
mostly water: the box area is a poor proxy for how much material is actually there.
Turning each box into a mask gives an area estimate that means something, which is
what any downstream quantity ("how many square metres of debris") depends on.

This module is optional by design. SAM2 is a second set of weights and a second
forward pass, and the detection pipeline is fully useful without it, so nothing
else in the package imports it eagerly. Importing :mod:`mdebris.models` does not
pull in SAM2; you have to ask for it.

Verified on this machine with transformers 5.14.1:

- ``facebook/sam2.1-hiera-tiny`` loads through ``Sam2Model`` / ``Sam2Processor``.
- The processor takes ``input_boxes`` as ``[[[x0, y0, x1, y1], ...]]``, nested one
  level for the batch and one for the boxes, in ORIGINAL image pixel coordinates.
  It rescales them to the model's 1024x1024 input itself.
- ``model(**inputs, multimask_output=False)`` returns ``pred_masks`` of shape
  ``(batch, num_boxes, 1, 256, 256)`` plus ``iou_scores`` of ``(batch, num_boxes, 1)``.
- ``processor.post_process_masks(pred_masks, inputs["original_sizes"])`` returns a
  list of boolean tensors of shape ``(num_boxes, 1, H, W)`` at full resolution.
- Measured: 7.6 s for one 480x640 image with 2 boxes on CPU (fp32).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from mdebris.config import settings
from mdebris.models.base import ModelLoadError, as_uint8_rgb
from mdebris.types import Detection

__all__ = ["Sam2Segmenter"]

log = logging.getLogger(__name__)


class Sam2Segmenter:
    """Refine detection boxes into masks, and record the refined pixel area.

    Not a :class:`~mdebris.models.base.BaseDetector`: it does not detect anything.
    It takes boxes that a detector already produced and sharpens them, so it has a
    ``refine`` method rather than ``detect`` and does not implement the ``Detector``
    protocol.

    Args:
        model_id: HF checkpoint. Defaults to ``settings.segment_model``
            (``facebook/sam2.1-hiera-tiny``), verified loadable in transformers 5.14.1.
        device: Torch device, or None to resolve from settings.
        torch_threads: CPU thread count, or None for the torch default.
        max_boxes: Cap on boxes per forward pass. SAM2 decodes every box against one
            shared image embedding, so the encoder cost is paid once, but memory
            still grows with box count. Batches beyond this are chunked.

    Example:
        >>> seg = Sam2Segmenter()                  # no weights fetched yet
        >>> refined = seg.refine(chip, detections) # weights load here
        >>> refined[0].indices["mask_pixels"]
        1843.0
    """

    name = "sam2"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = None,
        torch_threads: int | None = None,
        max_boxes: int = 32,
    ) -> None:
        self.model_id = model_id or settings.segment_model
        self._device = device
        self._torch_threads = torch_threads
        self.max_boxes = max_boxes
        self.model = None
        self.processor = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = settings.resolve_device()
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Fetch SAM2 weights. Idempotent.

        Raises:
            ModelLoadError: If transformers is too old to expose ``Sam2Model``, or
                the checkpoint cannot be fetched. The message names the remedy
                because this is the one optional component most likely to be absent.
        """
        if self.is_loaded:
            return
        import torch

        threads = (
            self._torch_threads if self._torch_threads is not None else settings.torch_threads
        )
        if threads is not None:
            torch.set_num_threads(int(threads))

        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise ModelLoadError(
                "SAM2 needs transformers>=4.56 for Sam2Model/Sam2Processor "
                f"(import failed: {exc}). Mask refinement is optional: run without "
                "--refine, or upgrade with `pip install -U 'mdebris[models]'`."
            ) from exc

        try:
            self.processor = Sam2Processor.from_pretrained(self.model_id)
            model = Sam2Model.from_pretrained(self.model_id)
            self.model = model.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 - re-raised with an actionable message
            raise ModelLoadError(
                f"could not load SAM2 checkpoint {self.model_id!r}: {exc}. "
                "Mask refinement is optional, so the pipeline can run without it. "
                "To enable it, check network access to huggingface.co, or set "
                "MDEBRIS_SEGMENT_MODEL to a checkpoint present in your local HF cache."
            ) from exc
        log.info("loaded %s (%s) on %s", self.name, self.model_id, self.device)

    def unload(self) -> None:
        self.model = None
        self.processor = None

    # ---- refinement ------------------------------------------------------------

    def refine(
        self, image: np.ndarray, detections: Sequence[Detection]
    ) -> list[Detection]:
        """Set ``.mask`` on each detection and record the mask area.

        Each detection gets:

        - ``.mask``: a boolean HxW array aligned to the full input image, not to the
          box. Full-frame masks cost more memory but mean a mask never has to be
          re-registered against the image later, which is where off-by-one errors
          in crop-relative masks come from.
        - ``.indices['mask_pixels']``: how many pixels the mask covers. This is the
          number to use for area, not ``bbox.area``.
        - ``.indices['mask_fill']``: mask pixels divided by box area, in [0, 1]. A
          low value means the box was mostly water, which is the signature of thin
          filament debris and a useful shape feature in its own right.
        - ``.indices['mask_iou']``: SAM2's own predicted quality for that mask, so a
          consumer can distinguish a confident mask from a guess.

        Detections are mutated in place and also returned, so the call reads the same
        whether or not you keep the result.

        Args:
            image: The same HxWx3 image the boxes were detected on.
            detections: Boxes to refine. An empty sequence is a no-op that does not
                load weights.

        Returns:
            The same detection objects, with masks attached.
        """
        if not detections:
            return list(detections)
        import torch

        arr = as_uint8_rgb(image)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        self.load()

        out = list(detections)
        for start in range(0, len(out), self.max_boxes):
            chunk = out[start : start + self.max_boxes]
            boxes = [[list(d.bbox.as_xyxy()) for d in chunk]]
            inputs = self.processor(images=arr, input_boxes=boxes, return_tensors="pt")
            if self.device != "cpu":
                inputs = inputs.to(self.device)
            with torch.inference_mode():
                # One box gets one mask. multimask_output=True would return three
                # candidate granularities per box, which is the right call for an
                # ambiguous point prompt but not for a box that already states the
                # intended extent.
                outputs = self.model(**inputs, multimask_output=False)

            masks = self.processor.post_process_masks(
                outputs.pred_masks, inputs["original_sizes"]
            )[0]  # (num_boxes, 1, H, W) bool
            iou_scores = outputs.iou_scores.detach().cpu().numpy().reshape(len(chunk), -1)

            for i, det in enumerate(chunk):
                mask = masks[i, 0].detach().cpu().numpy().astype(bool)
                if mask.shape != (height, width):  # defensive, shapes have matched in testing
                    log.warning(
                        "SAM2 mask shape %s does not match image %s; skipping",
                        mask.shape,
                        (height, width),
                    )
                    continue
                pixels = float(mask.sum())
                det.mask = mask
                det.indices["mask_pixels"] = pixels
                box_area = det.bbox.area
                det.indices["mask_fill"] = pixels / box_area if box_area > 0 else 0.0
                det.indices["mask_iou"] = float(iou_scores[i, 0])
        return out

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "lazy"
        return f"Sam2Segmenter(model_id={self.model_id!r}, device={self._device!r}, {state})"
