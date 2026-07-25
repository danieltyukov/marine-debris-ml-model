"""Open-vocabulary detectors. This is the default detection path.

The legacy pipeline fine-tuned a single-class SSD on hand-labelled Planet chips.
That works only for the imagery, season and sensor it was labelled on, and it can
only ever answer "debris / not debris". An open-vocabulary detector needs no
labelled data at all and answers a richer question, because the class list is a
list of sentences supplied at inference time (see :mod:`mdebris.models.prompts`).

Two backends are provided. OWLv2 is the default: it takes a list of prompts, scores
every box against every prompt, and returns the winning prompt index, which is
exactly the structure the confuser-prompt design needs. GroundingDINO is the
alternative, and its prompt format is different enough (one dot-separated caption
string, phrase output rather than index output) that the difference is encapsulated
here rather than leaking into the pipeline.

Measured on this machine (22-core CPU, torch 2.13 fp32, no GPU), one 960x960
forward pass:

- OWLv2 base-patch16-ensemble, 155M params: about 18-23 s
- GroundingDINO tiny, 172M params: about 35 s

Neither is fast. That latency is the entire reason :mod:`mdebris.indices` exists:
screening tiles with an arithmetic spectral index first means the detector only
runs on the small fraction of tiles that could plausibly contain anything.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from mdebris.config import settings
from mdebris.models.base import BaseDetector, as_uint8_rgb, clip_detections, nms
from mdebris.models.prompts import DEFAULT_PROMPTS, PromptSet
from mdebris.types import BBox, Detection, SurfaceClass

__all__ = ["GroundingDinoDetector", "OWLv2Detector"]

log = logging.getLogger(__name__)


class OWLv2Detector(BaseDetector):
    """Open-vocabulary detection with OWLv2. The default detector.

    Args:
        model_id: HF checkpoint. Defaults to ``settings.zeroshot_model``.
        prompts: The prompt set defining the class space. Defaults to
            :data:`~mdebris.models.prompts.DEFAULT_PROMPTS`, which includes confusers.
        device: Torch device, or None to resolve from settings.
        torch_threads: CPU thread count, or None for the torch default.
        apply_nms: Suppress duplicate boxes within one image. OWLv2 emits a dense,
            highly redundant box set, so this is on by default.
        nms_iou: IoU threshold for that suppression. Defaults to settings.
        drop_background: Discard boxes whose winning prompt maps to ``open_water``.
            Those prompts exist to absorb empty ocean so it stops being forced into
            an object class; the boxes they win are by construction "nothing here"
            and are not useful output. Set False to inspect them while tuning prompts.

    Example:
        >>> det = OWLv2Detector()               # no weights fetched yet
        >>> dets = det.detect(chip, threshold=0.1)   # weights load here
    """

    name = "owlv2"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        prompts: PromptSet | None = None,
        device: str | None = None,
        torch_threads: int | None = None,
        apply_nms: bool = True,
        nms_iou: float | None = None,
        drop_background: bool = True,
    ) -> None:
        super().__init__(
            model_id or settings.zeroshot_model, device=device, torch_threads=torch_threads
        )
        self.prompts = prompts or DEFAULT_PROMPTS
        self.apply_nms = apply_nms
        self.nms_iou = settings.nms_iou_threshold if nms_iou is None else nms_iou
        self.drop_background = drop_background

    def _load(self) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(self.model_id)
        model = Owlv2ForObjectDetection.from_pretrained(self.model_id)
        self.model = model.to(self.device).eval()

    # ---- inference -------------------------------------------------------------

    def _forward(self, images: Sequence[np.ndarray]):
        """Run both towers over a batch, returning (outputs, target_sizes, batch_texts)."""
        import torch

        texts = self.prompts.texts
        batch_texts = [texts for _ in images]
        # The processor resizes and pads every image to 960x960 regardless of input
        # size, so a small tile costs exactly as much as a full-size one.
        #
        # No padding override: the processor default pads prompts to the 16-token
        # context the checkpoint was trained with. Passing padding=True instead pads
        # to the longest prompt in the batch (11 tokens for the default set), which
        # makes the text tower's input length depend on which prompts happen to be
        # in use. Leaving it alone keeps tokenization identical for every prompt set.
        inputs = self.processor(text=batch_texts, images=list(images), return_tensors="pt")
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # OWLv2 pads to a square before resizing, and transformers' _scale_boxes
        # accounts for that internally by scaling with max(height, width). Passing
        # the true (h, w) is therefore correct; boxes may still land in the padded
        # region, which is why the caller clips afterwards.
        target_sizes = [(int(im.shape[0]), int(im.shape[1])) for im in images]
        return outputs, target_sizes, batch_texts

    def _results_to_detections(self, result: dict, image_shape: tuple[int, int]) -> list[Detection]:
        height, width = image_shape
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()
        text_labels = result.get("text_labels") or [None] * len(scores)

        dets: list[Detection] = []
        for box, score, label_idx, text in zip(boxes, scores, labels, text_labels, strict=False):
            cls = (
                self.prompts.label_for_text(text)
                if text
                else self.prompts.label_for_index(int(label_idx))
            )
            if self.drop_background and cls is SurfaceClass.WATER:
                continue
            xmin, ymin, xmax, ymax = box
            if xmax <= xmin or ymax <= ymin:  # degenerate, BBox would raise
                continue
            dets.append(
                Detection(
                    bbox=BBox(float(xmin), float(ymin), float(xmax), float(ymax)),
                    # sigmoid output is already in [0, 1]; clamp guards float drift
                    # at the boundary, which Detection validates against.
                    score=min(1.0, max(0.0, float(score))),
                    label=cls,
                    source_model=f"{self.name}:{self.model_id}",
                )
            )
        dets = clip_detections(dets, width, height)
        return nms(dets, self.nms_iou) if self.apply_nms else dets

    def _detect(self, image: np.ndarray, threshold: float) -> list[Detection]:
        outputs, target_sizes, batch_texts = self._forward([image])
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes, text_labels=batch_texts
        )
        return self._results_to_detections(results[0], (image.shape[0], image.shape[1]))

    def detect_batch(
        self, images: Sequence[np.ndarray], *, threshold: float | None = None
    ) -> list[list[Detection]]:
        """Run a real batched forward pass.

        Measured on this 22-core CPU box there is no meaningful throughput win: the
        per-image matmuls already use every core, so a batch of 4 costs about what
        four sequential images cost while holding four times the activations. The
        batched path exists because it is a clear win on a GPU, and because it keeps
        one code path for both devices. On CPU, prefer :meth:`detect`.
        """
        if not images:
            return []
        thr = settings.score_threshold if threshold is None else threshold
        if not 0.0 <= thr <= 1.0:
            raise ValueError(f"threshold {thr} outside [0, 1]")
        arrays = [as_uint8_rgb(im) for im in images]
        self.load()
        outputs, target_sizes, batch_texts = self._forward(arrays)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=thr, target_sizes=target_sizes, text_labels=batch_texts
        )
        out: list[list[Detection]] = []
        for result, arr in zip(results, arrays, strict=True):
            dets = self._results_to_detections(result, (arr.shape[0], arr.shape[1]))
            dets.sort(key=lambda d: -d.score)
            out.append(dets)
        return out


class GroundingDinoDetector(BaseDetector):
    """Open-vocabulary detection with GroundingDINO.

    An alternative to OWLv2, kept because the two models fail differently and a
    disagreement between them is a useful signal on unlabelled data. The prompt
    handling is genuinely different and that difference is contained here:

    - OWLv2 takes a list of prompts and returns the index of the winning prompt.
    - GroundingDINO takes ONE lowercase caption with phrases joined by ". " and
      terminated by a period, and returns a text phrase, which may be a fragment of
      the prompt that fired. Mapping that fragment back to a class is what
      :meth:`~mdebris.models.prompts.PromptSet.label_for_text` handles.

    It also needs ``input_ids`` passed back into post-processing, and it has a second
    threshold (``text_threshold``) governing how strongly a token must match before
    it contributes to a phrase.

    Measured at about 35 s per image on this CPU with the tiny checkpoint, roughly
    1.5x slower than OWLv2 base.
    """

    name = "grounding-dino"
    DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        prompts: PromptSet | None = None,
        device: str | None = None,
        torch_threads: int | None = None,
        text_threshold: float = 0.25,
        apply_nms: bool = True,
        nms_iou: float | None = None,
        drop_background: bool = True,
    ) -> None:
        super().__init__(model_id or self.DEFAULT_MODEL, device=device, torch_threads=torch_threads)
        self.prompts = prompts or DEFAULT_PROMPTS
        self.text_threshold = text_threshold
        self.apply_nms = apply_nms
        self.nms_iou = settings.nms_iou_threshold if nms_iou is None else nms_iou
        self.drop_background = drop_background

    def _load(self) -> None:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        model = GroundingDinoForObjectDetection.from_pretrained(self.model_id)
        self.model = model.to(self.device).eval()

    def _detect(self, image: np.ndarray, threshold: float) -> list[Detection]:
        import torch

        caption = self.prompts.as_dot_string()
        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        inputs = self._to_device(inputs)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        height, width = int(image.shape[0]), int(image.shape[1])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        dets: list[Detection] = []
        boxes = results["boxes"].detach().cpu().tolist()
        scores = results["scores"].detach().cpu().tolist()
        # Unlike OWLv2 these are phrases, not indices into the prompt list.
        text_labels = results.get("text_labels") or []
        for box, score, text in zip(boxes, scores, text_labels, strict=False):
            cls = self.prompts.label_for_text(str(text))
            if self.drop_background and cls is SurfaceClass.WATER:
                continue
            xmin, ymin, xmax, ymax = box
            if xmax <= xmin or ymax <= ymin:
                continue
            dets.append(
                Detection(
                    bbox=BBox(float(xmin), float(ymin), float(xmax), float(ymax)),
                    score=min(1.0, max(0.0, float(score))),
                    label=cls,
                    source_model=f"{self.name}:{self.model_id}",
                )
            )
        dets = clip_detections(dets, width, height)
        return nms(dets, self.nms_iou) if self.apply_nms else dets
