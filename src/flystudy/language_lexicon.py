"""Fixed 100-word vocabulary, sensory scene types and text rendering."""
from __future__ import annotations

from dataclasses import dataclass


NOUNS = "fly body leg legs side strength touch taste movement pressure signal sensation event events pain food ground danger place time pulse change".split()
ATTRIBUTES = "left right front middle hind weak strong short long quickly slowly first last same different one two three four".split()
VERBS = "feel felt detect detected happen happened start started stop stopped mean meant avoid avoided approach approached follow followed do".split()
FUNCTION = "a the that my its it i they both another is are was were did in on to from of and but so because before after while then not no yes what where when which why how many there or".split()
LEXICON = NOUNS + ATTRIBUTES + VERBS + FUNCTION
PUNCTUATION = [".", "?", ","]
CONTROL = ["<BOS>", "<EOS>", "<PAD>", "<SEP>", "<SENSE>", "<TEXT>"]
VOCAB = LEXICON + PUNCTUATION + CONTROL
TOKEN = {word: index for index, word in enumerate(VOCAB)}
BOS, EOS, PAD, SEP, SENSE, TEXT = (TOKEN[word] for word in CONTROL)
SIDES = ("left", "right")
PARTS = ("front", "middle", "hind")
LOCATIONS = tuple((side, part) for side in SIDES for part in PARTS)
KINDS = ("touch", "taste", "movement", "pressure")
LEVELS = ("weak", "strong")
DURATIONS = ("short", "long")
SPEEDS = ("slowly", "quickly")
NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four"}
MAX_ANSWER_TOKENS = 128
MAX_TARGETS = MAX_ANSWER_TOKENS + 1

if len(LEXICON) != 100 or len(set(LEXICON)) != 100:
    raise RuntimeError("the lexicon must contain exactly 100 unique words")
if len(VOCAB) > 128:
    raise RuntimeError("the fixed 128-dimensional orthogonal code cannot cover the vocabulary")


@dataclass(frozen=True)
class Event:
    location: int
    kind: int
    level: int
    duration: int
    speed: int


@dataclass(frozen=True)
class Scene:
    events: tuple[Event, ...]


def surface_text(words: tuple[str, ...]) -> str:
    """Convert lower-case corpus tokens into conventional written English."""
    result = ""
    capitalize = True
    for word in words:
        if word in PUNCTUATION:
            result = result.rstrip() + word
            if word in (".", "?"):
                capitalize = True
            continue
        shown = "I" if word == "i" else (word.capitalize() if capitalize else word)
        result += ("" if not result or result.endswith(" ") else " ") + shown
        capitalize = False
    return result
