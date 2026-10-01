"""Natural controlled English for grounded compositional language experiments.

The latent sensory scene is sampled before any sentence is rendered.  Semantic
families are split before paraphrasing, so a held-out channel pair cannot leak
through a different wording or through the text-only portion of the corpus.
All surface forms use the fixed 100-word lexicon plus punctuation.

The stimulus-to-meaning rules are authored experimental conventions.  They are
not claims about what pain, danger, food, or language mean to a biological fly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json

import numpy as np

from flystudy.language_lexicon import (
    Event,
    Scene,
    LEXICON,
    PUNCTUATION,
    LOCATIONS,
    KINDS,
    LEVELS,
    DURATIONS,
    SPEEDS,
    surface_text,
)


SPLITS = ("train", "validation_id", "validation_composition", "test_composition")
REPORT_FAMILIES = ("report.self", "report.fly")
QA_FAMILIES = (
    "qa.feel",
    "qa.detect",
    "qa.first",
    "qa.after",
    "qa.first_place",
    "qa.last_place",
    "qa.strength",
    "qa.start",
    "qa.stop",
    "qa.same_strength",
    "qa.same_side",
    "qa.same_leg",
    "qa.count",
    "qa.before",
    "qa.same_kind",
    "qa.meaning",
    "qa.action",
    "qa.reason",
)
TEXT_FAMILIES = (
    "text.temporal",
    "text.signal",
    "text.meaning",
    "text.reason",
    "text.contrast",
    "text.while",
    "text.count_three",
    "text.count_four",
    "text.pulse",
    "text.side",
    "text.dialogue",
)
COMPOSITION_QA_FAMILIES = ("qa.feel", "qa.detect", "qa.first", "qa.after")


def _canonical_pair(a: int, b: int) -> tuple[int, int]:
    if not 0 <= a < 24 or not 0 <= b < 24 or a == b:
        raise ValueError("a channel pair requires two distinct channels in [0, 24)")
    return (a, b) if a < b else (b, a)


VALIDATION_PAIRS = tuple(sorted({_canonical_pair(i, (i + 1) % 24) for i in range(24)}))
TEST_PAIRS = tuple(sorted({_canonical_pair(i, (i + 5) % 24) for i in range(24)}))
_ALL_PAIRS = {_canonical_pair(a, b) for a in range(24) for b in range(a + 1, 24)}
TRAIN_PAIRS = tuple(sorted(_ALL_PAIRS - set(VALIDATION_PAIRS) - set(TEST_PAIRS)))
PARTITION_PAIRS = {
    "train": TRAIN_PAIRS,
    "validation": VALIDATION_PAIRS,
    "test": TEST_PAIRS,
}
_PAIR_TO_PARTITION = {
    pair: partition for partition, pairs in PARTITION_PAIRS.items() for pair in pairs
}

if len(TRAIN_PAIRS) != 228 or len(VALIDATION_PAIRS) != 24 or len(TEST_PAIRS) != 24:
    raise RuntimeError("invalid channel-pair partition sizes")


def event_channel(event: Event) -> int:
    return event.location * len(KINDS) + event.kind


def pair_family(first: Event, second: Event) -> tuple[int, int]:
    return _canonical_pair(event_channel(first), event_channel(second))


def pair_partition(family: tuple[int, int]) -> str:
    return _PAIR_TO_PARTITION[_canonical_pair(*family)]


def scene_pair_families(scene: Scene) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({pair_family(a, b) for a, b in combinations(scene.events, 2)}))


@dataclass(frozen=True)
class CorpusRecord:
    record_id: int
    split: str
    mode: str
    template_family: str
    variant: int
    scene: Scene
    prompt_words: tuple[str, ...]
    answer_words: tuple[str, ...]

    @property
    def pair_families(self) -> tuple[tuple[int, int], ...]:
        return scene_pair_families(self.scene)

    @property
    def prompt(self) -> str:
        return surface_text(self.prompt_words)

    @property
    def answer(self) -> str:
        return surface_text(self.answer_words)


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


def _validate(words: tuple[str, ...], *, question_or_text: bool) -> None:
    allowed = set(LEXICON) | set(PUNCTUATION)
    unknown = set(words) - allowed
    if unknown:
        raise RuntimeError(f"tokens outside the fixed vocabulary: {sorted(unknown)}")
    if not words or words[-1] not in ({"?", "."} if question_or_text else {"."}):
        raise RuntimeError("invalid terminal punctuation")
    if any(a in PUNCTUATION and b in PUNCTUATION for a, b in zip(words, words[1:])):
        raise RuntimeError("adjacent punctuation is not allowed")
    banned = (
        ("a", "pressure"),
        ("a", "weak", "pressure"),
        ("a", "strong", "pressure"),
        ("one", "events"),
        ("two", "event"),
        ("three", "event"),
        ("four", "event"),
        ("did", "happened"),
        ("did", "detected"),
        ("did", "felt"),
    )
    if any(tuple(words[i:i + len(phrase)]) == phrase
           for phrase in banned for i in range(len(words) - len(phrase) + 1)):
        raise RuntimeError("known unnatural phrase")


def _values(event: Event) -> tuple[str, str, str, str, str, str]:
    side, part = LOCATIONS[event.location]
    return side, part, KINDS[event.kind], LEVELS[event.level], DURATIONS[event.duration], SPEEDS[event.speed]


def _felt(event: Event, owner: str) -> str:
    side, part, kind, level, _, _ = _values(event)
    article = "" if kind == "pressure" else "a "
    return f"{article}{level} {kind} on {owner} {side} {part} leg"


def _definite(event: Event) -> str:
    side, part, kind, level, _, _ = _values(event)
    return f"the {level} {kind} on the {side} {part} leg"


def _timing(event: Event, subject: str) -> str:
    _, _, _, _, duration, speed = _values(event)
    return f"{subject} started {speed} and stopped after a {duration} time ."


def _meaning(kind: int) -> str:
    return (
        "a signal from the body",
        "food",
        "a change in the ground",
        "pain and danger",
    )[kind]


def _action(kind: int, *, past: bool) -> str:
    base = (
        ("follow", "followed", "the signal"),
        ("approach", "approached", "the food"),
        ("follow", "followed", "the movement"),
        ("avoid", "avoided", "that place"),
    )[kind]
    return f"{base[1] if past else base[0]} {base[2]}"


def _event_report(event: Event, *, reporter: str, timing_subject: str) -> str:
    if reporter == "self":
        clause = f"i felt {_felt(event, 'my')} ."
    elif reporter == "fly":
        clause = f"the fly detected {_felt(event, 'its')} ."
    else:
        raise ValueError("unknown reporter")
    return f"{clause} {_timing(event, timing_subject)}"


def _render_report(scene: Scene, reporter: str, variant: int) -> tuple[str, str]:
    if len(scene.events) not in (1, 2):
        raise ValueError("reports require one or two events")
    first = scene.events[0]
    owner = "my" if reporter == "self" else "its"
    actor = "i" if reporter == "self" else "the fly"
    verb = "felt" if reporter == "self" else "detected"
    question = "what did i feel ?" if reporter == "self" else "what did the fly detect ?"
    if len(scene.events) == 1:
        return question, _event_report(first, reporter=reporter, timing_subject="the sensation")
    last = scene.events[-1]
    _, _, _, _, duration1, speed1 = _values(first)
    _, _, _, _, duration2, speed2 = _values(last)
    first_object, last_object = _felt(first, owner), _felt(last, owner)
    first_np, last_np = _definite(first), _definite(last)
    forms = (
        (f"first , {actor} {verb} {first_object} . {_timing(first, 'the sensation')} "
         f"then , {actor} {verb} {last_object} . {_timing(last, 'the sensation')}"),
        (f"{actor} {verb} {first_object} first . "
         f"it started {speed1} and stopped after a {duration1} time . "
         f"after that , {actor} {verb} {last_object} . "
         f"it started {speed2} and stopped after a {duration2} time ."),
        (f"{first_np} happened before {last_np} . "
         f"the first signal started {speed1} and stopped after a {duration1} time . "
         f"the last signal started {speed2} and stopped after a {duration2} time ."),
        (f"before {last_np} happened , {actor} {verb} {first_object} . "
         f"the first sensation started {speed1} and stopped after a {duration1} time . "
         f"then , {actor} {verb} {last_object} . "
         f"the last sensation started {speed2} and stopped after a {duration2} time ."),
    )
    return question, forms[variant % len(forms)]


def _yes_no(value: bool, yes: str, no: str) -> str:
    return f"yes , {yes} ." if value else f"no , {no} ."


def _render_qa(scene: Scene, family: str, variant: int) -> tuple[str, str]:
    if not scene.events:
        raise ValueError("QA requires a scene")
    first, last = scene.events[0], scene.events[-1]
    side1, part1, kind1, level1, duration1, speed1 = _values(first)
    side2, part2, kind2, level2, _, _ = _values(last)
    first_np, last_np = _definite(first), _definite(last)
    count_word = ("one", "two", "three", "four")[len(scene.events) - 1]
    mapping = {
        "qa.feel": _render_report(scene, "self", variant),
        "qa.detect": _render_report(scene, "fly", variant),
        "qa.first": ("which event happened first ?", f"{first_np} happened before {last_np} ."),
        "qa.after": ("what happened after the first event ?", f"{last_np} happened after {first_np} ."),
        "qa.first_place": ("where did the first event happen ?", f"it happened on the {side1} {part1} leg ."),
        "qa.last_place": ("where did the last event happen ?", f"it happened on the {side2} {part2} leg ."),
        "qa.strength": ("how strong was the first sensation ?", f"the first sensation was {level1} ."),
        "qa.start": ("how did the first signal start ?", f"it started {speed1} ."),
        "qa.stop": ("when did the first signal stop ?", f"it stopped after a {duration1} time ."),
        "qa.same_strength": (
            "are the first and last events the same strength ?",
            (_yes_no(True, f"both were {level1}", "") if level1 == level2 else
             _yes_no(False, "", f"the first was {level1} but the last was {level2}")),
        ),
        "qa.same_side": (
            "did the first and last events happen on the same side ?",
            (_yes_no(True, f"both happened on the {side1} side", "") if side1 == side2 else
             _yes_no(False, "", f"the first happened on the {side1} side but the last happened on the {side2} side")),
        ),
        "qa.same_leg": (
            "did the first and last events happen on the same leg ?",
            _yes_no(first.location == last.location, "they happened on the same leg", "they happened on different legs"),
        ),
        "qa.count": ("how many events happened ?", f"{count_word} {'event' if len(scene.events) == 1 else 'events'} happened ."),
        "qa.before": _named_order_question(first, last),
        "qa.same_kind": (
            "did the first and last events mean the same ?",
            _yes_no(first.kind == last.kind, "they meant the same", "they did not mean the same"),
        ),
        "qa.meaning": (f"what did the {kind1} mean ?", f"it meant {_meaning(first.kind)} ."),
        "qa.action": (f"what did the fly do because of the {kind1} ?", f"it {_action(first.kind, past=True)} ."),
        "qa.reason": (f"why did the fly {_action(first.kind, past=False)} ?", f"because the {kind1} meant {_meaning(first.kind)} ."),
    }
    try:
        return mapping[family]
    except KeyError as error:
        raise ValueError(f"unknown QA family: {family}") from error


def _render_text(scene: Scene, family: str, variant: int) -> tuple[str, str]:
    if not scene.events:
        raise ValueError("text continuation requires a scene")
    first, last = scene.events[0], scene.events[-1]
    side1, part1, kind1, level1, duration1, speed1 = _values(first)
    side2, _, _, level2, duration2, speed2 = _values(last)
    first_np, last_np = _definite(first), _definite(last)
    same_side = side1 == side2
    mapping = {
        "text.temporal": (f"{first_np} happened before {last_np} .", f"{last_np} happened after {first_np} ."),
        "text.signal": (
            f"the signal from the {side1} {part1} leg was a {duration1} pulse . it started {speed1} . did it start slowly or quickly ?",
            f"it started {speed1} and stopped after a {duration1} time .",
        ),
        "text.meaning": (
            f"{first_np} meant {_meaning(first.kind)} .",
            f"so the fly {_action(first.kind, past=True)} .",
        ),
        "text.reason": (
            f"the fly detected {_felt(first, 'its')} . why did it {_action(first.kind, past=False)} ?",
            f"because the {kind1} meant {_meaning(first.kind)} .",
        ),
        "text.contrast": (
            f"{first_np} happened first , but {last_np} happened last .",
            ("there was no change in strength ." if level1 == level2 else f"there was a change in strength from {level1} to {level2} ."),
        ),
        "text.while": (
            f"while the fly felt {_felt(first, 'its')} , it detected {_felt(last, 'its')} . the last sensation started {speed2} and was a {duration2} signal .",
            f"the last sensation stopped after a {duration2} time .",
        ),
        "text.count_three": (
            f"there were three events . one event happened first . it was {first_np} . the last event was {last_np} .",
            "another event followed the first event . the last event happened after both .",
        ),
        "text.count_four": (
            f"there were four events . the first event was {first_np} . the last event was {last_np} .",
            "the last event happened after the first three events .",
        ),
        "text.pulse": (
            f"the {kind1} on the {side1} {part1} leg is a {duration1} pulse . it started {speed1} . did it start slowly or quickly ?",
            f"it started {speed1} and stopped after a {duration1} time .",
        ),
        "text.side": (
            f"the first event happened on the {side1} side and the last event happened on the {side2} side .",
            (_yes_no(True, f"both happened on the {side1} side", "") if same_side else
             _yes_no(False, "", "they did not happen on the same side")),
        ),
    }
    if family == "text.dialogue":
        question, answer = _render_qa(scene, QA_FAMILIES[(variant // 4) % len(QA_FAMILIES)], variant % 4)
        _, context = _render_report(scene, "self" if variant % 2 else "fly", variant % 4)
        return f"{context} {question}", answer
    try:
        return mapping[family]
    except KeyError as error:
        raise ValueError(f"unknown text family: {family}") from error


def render_record(scene: Scene, family: str, variant: int = 0) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if family == "report.self":
        prompt, answer = _render_report(scene, "self", variant)
    elif family == "report.fly":
        prompt, answer = _render_report(scene, "fly", variant)
    elif family in QA_FAMILIES:
        prompt, answer = _render_qa(scene, family, variant)
    elif family in TEXT_FAMILIES:
        prompt, answer = _render_text(scene, family, variant)
    else:
        raise ValueError(f"unknown language template family: {family}")
    prompt_words, answer_words = _words(prompt), _words(answer)
    _validate(prompt_words, question_or_text=True)
    _validate(answer_words, question_or_text=False)
    return prompt_words, answer_words


def _rng(record_id: int, split: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, SPLITS.index(split), record_id]))


def _named_order_question(first: Event, last: Event) -> tuple[str, str]:
    a, b = sorted((first, last), key=event_channel)
    question = f"did {_definite(a)} happen before {_definite(b)} ?"
    answer = "yes , it happened before that event ." if a == first else "no , it happened after that event ."
    return question, answer


def _reserved(value) -> bool:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % 5 == 0


def scene_reserved(scene: Scene) -> bool:
    # Both event orders and every paraphrase share one ID split.
    return _reserved(sorted((e.location, e.kind, e.level, e.duration, e.speed) for e in scene.events))


def prompt_reserved(words: tuple[str, ...]) -> bool:
    return _reserved(words)


def _event(channel: int, rng: np.random.Generator) -> Event:
    return Event(channel // len(KINDS), channel % len(KINDS), int(rng.integers(2)),
                 int(rng.integers(2)), int(rng.integers(2)))


def _sample_scene(rng: np.random.Generator, *, count: int, partition: str) -> Scene:
    if count == 1:
        return Scene((_event(int(rng.integers(24)), rng),))
    if count == 2:
        a, b = PARTITION_PAIRS[partition][int(rng.integers(len(PARTITION_PAIRS[partition])))]
        if bool(rng.integers(2)):
            a, b = b, a
        return Scene((_event(a, rng), _event(b, rng)))
    if partition != "train" or count not in (3, 4):
        raise ValueError("long text scenes are restricted to training families")
    for _ in range(10_000):
        channels = tuple(int(x) for x in rng.choice(24, count, replace=False))
        if all(pair_partition(_canonical_pair(a, b)) == "train" for a, b in combinations(channels, 2)):
            return Scene(tuple(_event(channel, rng) for channel in channels))
    raise RuntimeError("could not sample a train-only multi-event scene")


def make_record(record_id: int, *, split: str = "train", seed: int = 20260921) -> CorpusRecord:
    if record_id < 0 or split not in SPLITS:
        raise ValueError("invalid record id or split")
    rng = _rng(record_id, split, seed)
    if split in ("validation_composition", "test_composition"):
        partition = "validation" if split == "validation_composition" else "test"
        if record_id % 2 == 0:
            mode = "report"
            family = REPORT_FAMILIES[(record_id // 2) % len(REPORT_FAMILIES)]
        else:
            mode = "qa"
            family = COMPOSITION_QA_FAMILIES[(record_id // 2) % len(COMPOSITION_QA_FAMILIES)]
        count = 2
    else:
        partition = "train"
        bucket = record_id % 10
        if bucket < 6:
            mode = "report"
            family = REPORT_FAMILIES[(record_id // 10) % len(REPORT_FAMILIES)]
            count = 1 if (record_id // 10 + bucket) % 4 == 0 else 2
        elif bucket < 8:
            mode = "qa"
            family = QA_FAMILIES[(record_id // 10) % len(QA_FAMILIES)]
            occurrence = 2 * ((record_id // 10) // len(QA_FAMILIES)) + bucket - 6
            count = 1 + occurrence % 2 if family == "qa.count" else 2
        else:
            mode = "text"
            family = TEXT_FAMILIES[(record_id // 10) % len(TEXT_FAMILIES)]
            count = 3 if family == "text.count_three" else 4 if family == "text.count_four" else 2
    variant = int(rng.integers(4 * len(QA_FAMILIES) if family == "text.dialogue" else 4))
    for _ in range(20_000):
        scene = _sample_scene(rng, count=count, partition=partition)
        if split in ("train", "validation_id"):
            if scene_reserved(scene) != (split == "validation_id"):
                continue
            if mode == "qa" and family in ("qa.same_strength", "qa.same_side", "qa.same_leg", "qa.same_kind", "qa.before"):
                a, b = scene.events
                truth = {"qa.same_strength": a.level == b.level,
                         "qa.same_side": a.location // 3 == b.location // 3,
                         "qa.same_leg": a.location == b.location,
                         "qa.same_kind": a.kind == b.kind,
                         "qa.before": event_channel(a) < event_channel(b)}[family]
                if truth != (occurrence % 2 == 0):
                    continue
        prompt, answer = render_record(scene, family, variant)
        if mode == "text" and prompt_reserved(prompt) != (split == "validation_id"):
            continue
        break
    else:
        raise RuntimeError(f"could not satisfy split/label constraints: {split}/{family}")
    record = CorpusRecord(record_id, split, mode, family, variant, scene, prompt, answer)
    if any(pair_partition(pair) != partition for pair in record.pair_families):
        raise RuntimeError("semantic family leaked across a split")
    return record


def swap_record(record: CorpusRecord) -> CorpusRecord:
    if len(record.scene.events) != 2 or record.template_family not in (
        set(REPORT_FAMILIES) | set(COMPOSITION_QA_FAMILIES)
    ):
        raise ValueError("record does not define an order-sensitive two-event task")
    scene = Scene(tuple(reversed(record.scene.events)))
    prompt, answer = render_record(scene, record.template_family, record.variant)
    swapped = CorpusRecord(record.record_id, record.split, record.mode, record.template_family,
                           record.variant, scene, prompt, answer)
    if swapped.answer_words == record.answer_words:
        raise RuntimeError("event-order swap did not change the answer")
    return swapped


def audit_corpus(*, examples: int = 30_000, seed: int = 20260921) -> dict:
    if examples < 1:
        raise ValueError("examples must be positive")
    words = Counter()
    target_words = Counter()
    loss_words = Counter()
    modes = Counter()
    templates = Counter()
    channels = Counter()
    pairs: set[tuple[int, int]] = set()
    seen_text: dict[tuple[str, ...], tuple[str, ...]] = {}
    text_conflicts = 0
    holdout_leaks = 0
    exact_records = Counter()
    surface_rows = Counter()
    lengths = []
    unknown = 0
    for record_id in range(examples):
        row = make_record(record_id, seed=seed)
        modes[row.mode] += 1
        templates[row.template_family] += 1
        words.update(word for word in row.prompt_words + row.answer_words if word in LEXICON)
        target_words.update(word for word in row.answer_words if word in LEXICON)
        supervised = row.prompt_words + row.answer_words if row.mode == "text" else row.answer_words
        loss_words.update(word for word in supervised if word in LEXICON)
        unknown += sum(word not in LEXICON and word not in PUNCTUATION
                       for word in row.prompt_words + row.answer_words)
        channels.update(event_channel(event) for event in row.scene.events)
        pairs.update(row.pair_families)
        holdout_leaks += sum(pair_partition(pair) != "train" for pair in row.pair_families)
        surface_rows[(row.prompt_words, row.answer_words)] += 1
        scene_key = tuple((event.location, event.kind, event.level, event.duration, event.speed)
                          for event in row.scene.events) if row.mode != "text" else ()
        exact_records[(scene_key, row.prompt_words, row.answer_words)] += 1
        lengths.append(len(row.prompt_words) + len(row.answer_words))
        if row.mode == "text":
            previous = seen_text.setdefault(row.prompt_words, row.answer_words)
            text_conflicts += previous != row.answer_words
    missing = sorted(set(LEXICON) - set(words))
    duplicates = sum(count - 1 for count in exact_records.values())
    surface_repeats = sum(count - 1 for count in surface_rows.values())
    return {
        "schema": 7,
        "revision": 2,
        "outputAnswerWordsObserved": len(target_words),
        "lossBearingWordsObserved": len(loss_words),
        "missingLossBearingWords": sorted(set(LEXICON) - set(loss_words)),
        "lossWordCounts": dict(sorted(loss_words.items())),
        "examples": examples,
        "grain": "one conditional prompt-answer exposure",
        "lexicalWordsExpected": len(LEXICON),
        "lexicalWordsObserved": len(words),
        "missingLexicalWords": missing,
        "minimumLexicalWordCount": min(words.values(), default=0),
        "maximumLexicalWordCount": max(words.values(), default=0),
        "wordCounts": dict(sorted(words.items())),
        "modeCounts": dict(sorted(modes.items())),
        "templateCounts": dict(sorted(templates.items())),
        "primitiveChannelsObserved": len(channels),
        "primitiveChannelCounts": dict(sorted(channels.items())),
        "trainingPairFamiliesObserved": len(pairs),
        "pairPartitionSizes": {name: len(group) for name, group in PARTITION_PAIRS.items()},
        "unknownTokenCount": unknown,
        "textPromptConflicts": int(text_conflicts),
        "holdoutFamilyLeaks": int(holdout_leaks),
        "uniqueConditionalRecords": len(exact_records),
        "uniqueSurfacePairs": len(surface_rows),
        "exactDuplicateRows": duplicates,
        "exactDuplicateRate": duplicates / examples,
        "surfacePairRepeatRate": surface_repeats / examples,
        "minimumCombinedTokens": min(lengths),
        "maximumCombinedTokens": max(lengths),
        "meanCombinedTokens": float(np.mean(lengths)),
        "claimScope": "authored 100-word grounded controlled English; not unrestricted English",
        "authoredMeaningRule": "touch=body signal, taste=food, movement=ground change, pressure=pain/danger",
    }
