"""Open-vocabulary prompt sets, and the reason confusers are in them.

An open-vocabulary detector scores every box against every prompt and keeps the
best-matching one. That single sentence is the whole design argument for this file.

If the prompt list contains only debris phrasings, the detector has no way to say
"this is something else". A sargassum mat is a bright, texturally busy, non-water
patch, so the debris prompt is the best of the offered options and the mat is
reported as debris with a high score. The model was never wrong: it answered the
only question it was asked.

Adding confuser prompts changes the question. Sargassum, ship wakes, sea foam and
cloud now compete for the same box, and because a sargassum mat genuinely matches
"a mat of floating seaweed" better than it matches "floating plastic debris", the
box comes back labelled sargassum. The false positive does not need to be filtered
out downstream because it was never emitted as debris in the first place. This
costs nothing at inference: prompts are encoded once and cached, and the text tower
is a rounding error next to the vision tower.

Sargassum, wakes and foam are precisely the confusers the marine-litter remote
sensing literature reports as dominant, which is why they get first-class
:class:`~mdebris.types.SurfaceClass` labels rather than being lumped into "not
debris". Keeping them as labelled output also makes the confusion structure
measurable: the evaluation module can report what debris is being confused *with*,
not merely how often it is wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from mdebris.types import SurfaceClass

__all__ = [
    "CONFUSER_PROMPTS",
    "DEFAULT_PROMPTS",
    "MINIMAL_PROMPTS",
    "PROMPT_SETS",
    "TARGET_PROMPTS",
    "PromptSet",
    "get_prompt_set",
]


# Target phrasings. Several wordings of the same concept are kept on purpose:
# CLIP-style text towers are sensitive to phrasing, and the ensemble checkpoint
# responds to "a patch of floating trash on the sea" and "marine litter on water"
# differently even though a human reads them as synonyms. Every one of these maps
# to DEBRIS, so extra phrasings raise recall without polluting the label space.
TARGET_PROMPTS: tuple[str, ...] = (
    "floating plastic debris",
    "a patch of floating trash on the sea",
    "marine litter on water",
    "floating garbage",
    "a raft of plastic bottles floating on the ocean",
    "discarded fishing net floating in the sea",
)

# Confusers. Each is a thing that is bright, non-water and roughly debris-shaped
# from orbit. They exist to win boxes that debris prompts would otherwise take.
CONFUSER_PROMPTS: dict[str, SurfaceClass] = {
    "a mat of floating seaweed": SurfaceClass.SARGASSUM,
    "sargassum algae on the ocean": SurfaceClass.SARGASSUM,
    "a ship": SurfaceClass.SHIP,
    "a boat on the water": SurfaceClass.SHIP,
    "a boat wake": SurfaceClass.WAKE,
    "a white trail of water behind a moving ship": SurfaceClass.WAKE,
    "white sea foam": SurfaceClass.FOAM,
    "breaking waves with white foam": SurfaceClass.FOAM,
    "a cloud over the ocean": SurfaceClass.CLOUD,
    "brown sediment in coastal water": SurfaceClass.SEDIMENT,
    # The background prompt matters as much as the confusers. Without an explicit
    # "this is just water" option, empty ocean has to be assigned to whichever
    # object prompt scores least badly, which manufactures low-score noise.
    "open blue ocean water": SurfaceClass.WATER,
    "calm empty sea surface": SurfaceClass.WATER,
}


@dataclass(frozen=True, slots=True)
class PromptSet:
    """An ordered prompt list plus the class each prompt votes for.

    Order is part of the contract, not an implementation detail: OWLv2 returns a
    label as an *index* into the list it was given, so the mapping from model output
    back to a :class:`SurfaceClass` is positional. Storing the pairs as a tuple keeps
    that ordering stable and the object hashable, which lets encoded prompts be
    cached per set.
    """

    entries: tuple[tuple[str, SurfaceClass], ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a prompt set needs at least one prompt")
        seen = [p for p, _ in self.entries]
        if len(set(seen)) != len(seen):
            dupes = sorted({p for p in seen if seen.count(p) > 1})
            raise ValueError(f"duplicate prompts would confuse index lookup: {dupes}")
        if not any(cls.is_target for _, cls in self.entries):
            raise ValueError("a prompt set with no target prompt can never detect debris")

    # ---- construction ----------------------------------------------------------

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, SurfaceClass], *, name: str = "custom"
    ) -> PromptSet:
        """Build from a ``{prompt: class}`` mapping, preserving insertion order."""
        return cls(entries=tuple(mapping.items()), name=name)

    @classmethod
    def build(
        cls,
        targets: Iterable[str],
        confusers: Mapping[str, SurfaceClass] | None = None,
        *,
        name: str = "custom",
    ) -> PromptSet:
        """Build from target phrasings plus an optional confuser mapping."""
        pairs: list[tuple[str, SurfaceClass]] = [(t, SurfaceClass.DEBRIS) for t in targets]
        pairs.extend((confusers or {}).items())
        return cls(entries=tuple(pairs), name=name)

    # ---- access ----------------------------------------------------------------

    @property
    def texts(self) -> list[str]:
        """Prompts in model order. This list is what gets handed to the processor."""
        return [p for p, _ in self.entries]

    @property
    def labels(self) -> list[SurfaceClass]:
        """Classes in model order, index-aligned with :attr:`texts`."""
        return [c for _, c in self.entries]

    @property
    def lookup(self) -> dict[str, SurfaceClass]:
        """Lowercased ``{prompt: class}``, for detectors that return a phrase not an index."""
        return {p.strip().lower(): c for p, c in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def label_for_index(self, index: int) -> SurfaceClass:
        """Map an OWLv2 label index back to a class.

        Out-of-range indices become UNKNOWN rather than raising: a detector that
        returns a surprising index should degrade to an unlabelled detection, not
        take down a scene-wide inference run.
        """
        if 0 <= index < len(self.entries):
            return self.entries[index][1]
        return SurfaceClass.UNKNOWN

    def label_for_text(self, text: str) -> SurfaceClass:
        """Map an emitted phrase back to a class, tolerating partial matches.

        GroundingDINO tokenizes the whole prompt string and can emit a fragment of
        the prompt that fired ("floating plastic" for "floating plastic debris"), so
        an exact lookup is not enough. The longest prompt that contains, or is
        contained by, the emitted phrase wins; a short generic prompt therefore
        cannot shadow a longer specific one.
        """
        low = text.strip().lower()
        table = self.lookup
        if (hit := table.get(low)) is not None:
            return hit
        best: tuple[int, SurfaceClass] | None = None
        for prompt, cls in table.items():
            if low and (low in prompt or prompt in low) and (best is None or len(prompt) > best[0]):
                best = (len(prompt), cls)
        return best[1] if best else SurfaceClass.UNKNOWN

    # ---- derived sets ----------------------------------------------------------

    def targets_only(self) -> PromptSet:
        """Drop the confusers. Useful for measuring what they are actually buying."""
        return PromptSet(
            entries=tuple((p, c) for p, c in self.entries if c.is_target),
            name=f"{self.name}-targets-only",
        )

    def with_prompts(self, extra: Mapping[str, SurfaceClass]) -> PromptSet:
        """Return a copy with extra prompts appended, skipping ones already present."""
        have = {p for p, _ in self.entries}
        added = tuple((p, c) for p, c in extra.items() if p not in have)
        return PromptSet(entries=self.entries + added, name=self.name)

    def as_dot_string(self) -> str:
        """The single dot-separated string GroundingDINO expects.

        GroundingDINO takes one lowercase caption with phrases separated by ". "
        and terminated by a period, not the list-of-lists OWLv2 wants. Keeping the
        conversion here means the two detectors can share one PromptSet.
        """
        return ". ".join(p.strip().lower().rstrip(".") for p, _ in self.entries) + "."


DEFAULT_PROMPTS = PromptSet.build(TARGET_PROMPTS, CONFUSER_PROMPTS, name="default")

# A cheaper set for latency-sensitive runs. Text encoding cost scales with prompt
# count, and while that cost is small next to the vision tower, the set is here so
# the trade can be measured rather than assumed. It keeps one confuser per major
# failure mode, which is where most of the precision benefit comes from.
MINIMAL_PROMPTS = PromptSet.build(
    ("floating plastic debris", "marine litter on water"),
    {
        "a mat of floating seaweed": SurfaceClass.SARGASSUM,
        "a ship": SurfaceClass.SHIP,
        "white sea foam": SurfaceClass.FOAM,
        "open blue ocean water": SurfaceClass.WATER,
    },
    name="minimal",
)

#: Sets addressable by name from the CLI and config.
PROMPT_SETS: dict[str, PromptSet] = {
    "default": DEFAULT_PROMPTS,
    "minimal": MINIMAL_PROMPTS,
    "targets-only": DEFAULT_PROMPTS.targets_only(),
}


def get_prompt_set(name: str) -> PromptSet:
    """Look up a named prompt set, erroring with the available names."""
    try:
        return PROMPT_SETS[name]
    except KeyError:
        raise KeyError(f"unknown prompt set {name!r}; available: {sorted(PROMPT_SETS)}") from None
