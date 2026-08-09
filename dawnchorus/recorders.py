"""
Recorder profiles: what a given box does to the audio and to the filenames.

A *site* is a place. A *recorder* is the hardware listening there. Those are separate
facts, and conflating them silently corrupts phenology: mic sensitivity, self-noise,
gain and sample rate all shift BirdNET's confidence scores, and confidence is what
`morning_summary` thresholds to decide when a species *started* singing. Two boxes on
the same post will not report the same onset minute. So every detection carries the
recorder that produced it, and comparisons are made recorder-aware rather than pooled.

Two concrete things a profile pins down, both of which are silent-failure modes:

  * **Filename timestamp convention.** The batch path reconstructs wall-clock time as
    `<start parsed from the filename> + <offset within file>`. Different makers stamp
    names differently; an unparseable name drops the whole recording.
  * **Clock zone.** AudioMoth stamps filenames in **UTC** by default, Song Meters in
    whatever you set (ours is station-local EDT). Guess wrong and solar time is off by
    the entire UTC offset, which looks like a plausible-but-wrong onset rather than an
    error.

Unknown hardware doesn't need a code change: `sniff()` infers the convention from the
filenames themselves, and `clock="unknown"` profiles are resolved empirically (see
`tools/compare_recorders.py`, which checks a candidate clock against solar time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

# --------------------------------------------------------------------------------------
# Filename timestamp conventions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Convention:
    """How a recorder writes the recording-start time into the filename."""

    id: str
    regex: str
    fmt: str                      # strptime format, or "hexepoch" for AudioMoth-legacy
    example: str

    def parse(self, text: str) -> datetime | None:
        """Return the recording start encoded in `text`, or None if it isn't there."""
        m = re.search(self.regex, str(text))
        if not m:
            return None
        tok = m.group(0)
        if self.fmt == "hexepoch":
            try:
                # Legacy AudioMoth: filename is the UTC unix epoch in hex. Return it
                # naive-UTC so it lines up with every other convention (which are naive
                # too); the recorder's `clock` decides whether a tz conversion follows.
                return datetime.fromtimestamp(int(tok, 16), tz=timezone.utc).replace(tzinfo=None)
            except (ValueError, OSError, OverflowError):
                return None
        try:
            return datetime.strptime(tok, self.fmt)
        except ValueError:
            return None


# Ordered most- to least-specific: `sniff` tries them in order and the first that parses
# a filename wins, so a longer/stricter pattern must precede one it contains.
CONVENTIONS: tuple[Convention, ...] = (
    # Owl Sense: date and time separated by BOTH "_" and "T". Must precede `iso-dash`,
    # which allows only a single separator character and would not match this at all.
    Convention("iso-dash-t", r"\d{4}-\d{2}-\d{2}_T\d{2}-\d{2}-\d{2}", "%Y-%m-%d_T%H-%M-%S",
               "e74d3_2026-08-06_T05-43-03.wav"),
    Convention("iso-dash", r"\d{4}-\d{2}-\d{2}[_T]\d{2}-\d{2}-\d{2}", "%Y-%m-%d_%H-%M-%S",
               "2026-05-17_04-30-00.wav"),
    Convention("iso-colon", r"\d{4}-\d{2}-\d{2}[_T]\d{2}:\d{2}:\d{2}", "%Y-%m-%d_%H:%M:%S",
               "2026-05-17T04:30:00.wav"),
    Convention("compact", r"\d{8}_\d{6}", "%Y%m%d_%H%M%S",
               "2MM43813_20260517_043000.wav"),
    Convention("compact-t", r"\d{8}T\d{6}", "%Y%m%dT%H%M%S",
               "20260517T043000.wav"),
    Convention("compact-min", r"\d{8}_\d{4}(?!\d)", "%Y%m%d_%H%M",
               "20260517_0430.wav"),
    Convention("hex-epoch", r"(?<![0-9A-Fa-f])[0-9A-F]{8}(?![0-9A-F])", "hexepoch",
               "5E0C1F80.WAV"),
)

CONVENTIONS_BY_ID = {c.id: c for c in CONVENTIONS}

# The historical default (Song Meter / modern AudioMoth), kept as the fallback so
# existing callers that pass no recorder behave exactly as before.
DEFAULT_CONVENTION = CONVENTIONS_BY_ID["compact"]


# --------------------------------------------------------------------------------------
# Recorder profiles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Recorder:
    """A model of recorder, plus how to read what it wrote.

    `clock` is one of:
      local   - filenames are already station-local; no conversion (file_tz None)
      utc     - filenames are UTC; must be converted to the station tz
      unknown - not yet established; treated as local but flagged by `needs_clock_check`
    """

    id: str
    name: str
    maker: str
    convention: str | None            # Convention.id, or None -> sniff from filenames
    clock: str                        # local | utc | unknown
    sample_rate: int | None = None    # Hz, as configured for this project
    channels: int | None = None
    notes: str = ""

    @property
    def file_tz(self) -> str | None:
        """Value to pass as `file_tz`; None means filenames are already station-local."""
        return "UTC" if self.clock == "utc" else None

    @property
    def needs_clock_check(self) -> bool:
        return self.clock == "unknown"

    @property
    def nyquist(self) -> float | None:
        """Highest frequency this recorder can represent, in Hz.

        BirdNET works at 48 kHz internally, so a recorder sampling below 96 kHz cannot
        show the model anything above its own Nyquist — high-frequency species (kinglets,
        waxwings, Blackpoll) are attenuated or absent. A real source of species-list
        differences between two boxes at the same spot.
        """
        return None if self.sample_rate is None else self.sample_rate / 2

    def resolve(self, filenames=()) -> "Recorder":
        """Return a copy with `convention` filled in, sniffing the names if needed."""
        if self.convention is not None:
            return self
        found = sniff(filenames)
        return replace(self, convention=(found.id if found else DEFAULT_CONVENTION.id))

    def timestamp_spec(self) -> tuple[str, str]:
        """(ts_regex, ts_format) for `load_birdnet_analyzer`."""
        c = CONVENTIONS_BY_ID.get(self.convention or "", DEFAULT_CONVENTION)
        return c.regex, c.fmt


REGISTRY: dict[str, Recorder] = {
    "song-meter-micro-2": Recorder(
        id="song-meter-micro-2",
        name="Song Meter Micro 2",
        maker="Wildlife Acoustics",
        convention="compact",
        clock="local",
        sample_rate=24000,
        channels=1,
        notes=("Names files <UNIT>_YYYYMMDD_HHMMSS.wav (unit 2MM43813 here). Clock is set "
               "to station-local EDT, confirmed against civil dawn on 2026-07-25/26 — do "
               "NOT pass --file-tz for this recorder."),
    ),
    "song-meter-generic": Recorder(
        id="song-meter-generic",
        name="Song Meter (SM4/Mini/Micro)",
        maker="Wildlife Acoustics",
        convention="compact",
        clock="local",
        notes="Family default for other Wildlife Acoustics units.",
    ),
    "audiomoth": Recorder(
        id="audiomoth",
        name="AudioMoth",
        maker="Open Acoustic Devices",
        convention="compact",
        clock="utc",
        notes=("Stamps filenames in UTC by default — pass the station tz so they convert, "
               "or every solar time is wrong by the whole UTC offset."),
    ),
    "audiomoth-legacy": Recorder(
        id="audiomoth-legacy",
        name="AudioMoth (hex-epoch firmware)",
        maker="Open Acoustic Devices",
        convention="hex-epoch",
        clock="utc",
        notes="Older firmware names files with the UTC unix epoch in hex, e.g. 5E0C1F80.WAV.",
    ),
    "owl-sense": Recorder(
        id="owl-sense",
        name="Owl Sense",
        maker="Owl Sense",
        convention="iso-dash-t",  # confirmed from unit e74d3's cards, 2026-08-07
        clock="local",            # CONFIRMED 2026-08-07 against the Song Meter (see notes)
        sample_rate=24000,
        channels=1,
        notes=("Added 2026-08-07 for the paired Montague trial against the Song Meter Micro 2. "
               "Names files <unit>_YYYY-MM-DD_THH-MM-SS.wav (unit e74d3 here) in 2-hour blocks; "
               "24 kHz/16-bit mono, i.e. the SAME audio format as the Song Meter Micro 2, so a "
               "species-list difference between them is NOT a bandwidth artefact. Clock CONFIRMED "
               "station-local on the paired 2026-08-06/07 mornings: cross-correlation lag 0 min "
               "(sharpness 14.4x) over mutual coverage, corroborated by a peak-minute delta of "
               "exactly 0.0 on all 11 paired species-mornings. Do NOT pass --file-tz for this box."),
    ),
    "generic": Recorder(
        id="generic",
        name="Generic recorder",
        maker="",
        convention=None,
        clock="unknown",
        notes="Unknown hardware: sniff the filename convention, assume a local clock.",
    ),
}


def get(recorder_id: str | None) -> Recorder | None:
    """Look up a profile by id. None/empty returns None (caller keeps its own defaults)."""
    if not recorder_id:
        return None
    try:
        return REGISTRY[recorder_id]
    except KeyError:
        raise ValueError(
            f"unknown recorder {recorder_id!r}. Known: {', '.join(sorted(REGISTRY))}. "
            "Add a profile to dawnchorus/recorders.REGISTRY, or use 'generic'."
        ) from None


def sniff(filenames) -> Convention | None:
    """Pick the convention that parses the most of `filenames`; None if none do.

    Lets an unregistered recorder work without a code change, and gives `resolve()` its
    answer for profiles that declare `convention=None`.
    """
    best, best_n = None, 0
    for c in CONVENTIONS:
        n = sum(1 for f in filenames if c.parse(str(f)) is not None)
        if n > best_n:
            best, best_n = c, n
    return best


def describe(recorder_id: str | None, filenames=()) -> str:
    """One-line human summary, used by the CLIs when they echo what they're about to do."""
    r = get(recorder_id)
    if r is None:
        return "recorder: unspecified (default filename convention, local clock)"
    r = r.resolve(filenames)
    # Don't say "Owl Sense Owl Sense" when the maker is already in the model name.
    title = r.name if (not r.maker or r.maker.lower() in r.name.lower()) else f"{r.maker} {r.name}"
    bits = [title, f"names={r.convention}", f"clock={r.clock}"]
    if r.sample_rate:
        bits.append(f"{r.sample_rate / 1000:g} kHz")
    if r.channels:
        bits.append("mono" if r.channels == 1 else f"{r.channels}ch")
    return "recorder: " + ", ".join(bits)
