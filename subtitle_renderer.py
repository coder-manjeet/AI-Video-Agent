"""FFmpeg capability probing and burn-in filter construction for subtitles.

Capability detection never assumes a binary or font exists: FFmpeg filter
support comes from ``ffmpeg -hide_banner -filters`` and fonts come from an
actual directory scan. Commands use argument lists only; no shell is involved.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from subtitle_models import SubtitleGenerationError, SubtitleStyle, ass_color

logger = logging.getLogger(__name__)

CAPABILITY_TIMEOUT_S = 30.0
FILTER_TIMEOUT_S = 120.0
_FILTER_SPECIALS = frozenset(",\\[]")
# FFmpeg filtergraph parsing needs two levels of escaping for a drive colon and
# three for a quote; commas and brackets only need one (verified with FFmpeg).
_COLON_ESCAPE = "\\" * 2 + ":"
_QUOTE_ESCAPE = "\\" * 3 + "'"
_STYLE_SPECIALS = frozenset("\\':,[]")

# Candidate families are verified from filenames; none is assumed installed.
DEVANAGARI_FONT_FAMILIES: tuple[str, ...] = (
    "Nirmala UI",
    "Mangal",
    "Noto Sans Devanagari",
    "Noto Serif Devanagari",
    "Lohit Devanagari",
    "Samanata",
    "Kohinoor Devanagari",
    "Devanagari Sangam MN",
)
_FONT_FILENAMES: dict[str, frozenset[str]] = {
    "Nirmala UI": frozenset(
        {"nirmala.ttf", "nirmalab.ttf", "nirmalai.ttf", "nirmalabi.ttf", "nirmalaui.ttf"}
    ),
    "Mangal": frozenset({"mangal.ttf", "mangalb.ttf", "mangal.otf"}),
    "Noto Sans Devanagari": frozenset(
        {
            "notosansdevanagari-regular.ttf",
            "notosansdevanagari-bold.ttf",
            "notosansdevanagari-medium.ttf",
            "notosansdevanagari.ttf",
        }
    ),
    "Noto Serif Devanagari": frozenset(
        {
            "notoserifdevanagari-regular.ttf",
            "notoserifdevanagari-bold.ttf",
            "notoserifdevanagari.ttf",
        }
    ),
    "Lohit Devanagari": frozenset({"lohitdevanagari.ttf", "lohit-devanagari.ttf"}),
    "Samanata": frozenset({"samanata.ttf", "samanatab.ttf"}),
    "Kohinoor Devanagari": frozenset(
        {"kohinoordevanagari.ttf", "kohinoordevanagaribold.ttf"}
    ),
    "Devanagari Sangam MN": frozenset({"devanagarisangammn.ttc"}),
}
_ALIGNMENTS = {"top": 8, "middle": 5, "bottom": 2}


class CommandOutput(Protocol):
    """The read-only part of a command result used for capability detection."""

    @property
    def stdout(self) -> str: ...

    @property
    def returncode(self) -> int: ...


CommandRunner = Callable[[Sequence[str], float], CommandOutput]


@dataclass(frozen=True)
class _CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _default_command_runner(command: Sequence[str], timeout_s: float) -> CommandOutput:
    """Run an argument-list command without a shell (the production runner)."""
    values = [str(part) for part in command]
    if not values:
        raise ValueError("command must not be empty")
    completed = subprocess.run(
        values,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return _CommandResult(
        tuple(values),
        completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


@dataclass(frozen=True)
class SubtitleCapabilities:
    """Verified FFmpeg filter and font support for subtitle burn-in."""

    ffmpeg_bin: str
    ffprobe_bin: str
    libass: bool
    burn_in_supported: bool
    fonts: tuple[str, ...]
    devanagari_font: str | None
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("ffmpeg_bin", "ffprobe_bin"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("libass", "burn_in_supported"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(self, "fonts", tuple(self.fonts))
        for font in self.fonts:
            if not isinstance(font, str):
                raise TypeError("fonts must contain strings")
        if self.devanagari_font is not None and not isinstance(self.devanagari_font, str):
            raise TypeError("devanagari_font must be a string or None")
        object.__setattr__(self, "notes", tuple(self.notes))
        for note in self.notes:
            if not isinstance(note, str):
                raise TypeError("notes must contain strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "ffmpeg_bin": self.ffmpeg_bin,
            "ffprobe_bin": self.ffprobe_bin,
            "libass": self.libass,
            "burn_in_supported": self.burn_in_supported,
            "fonts": list(self.fonts),
            "devanagari_font": self.devanagari_font,
            "notes": list(self.notes),
        }


def _font_directories(font_dirs: Sequence[str | Path] | None) -> list[Path]:
    candidates: list[Path] = []
    if font_dirs is not None:
        candidates.extend(Path(entry).expanduser() for entry in font_dirs)
    extra = os.environ.get("RENDER_SUBTITLE_FONT_DIRS", "")
    if extra.strip():
        candidates.extend(
            Path(entry.strip()).expanduser() for entry in extra.split(os.pathsep) if entry.strip()
        )
    windir = os.environ.get("WINDIR")
    candidates.append(Path(windir) / "Fonts" if windir else Path("C:/Windows/Fonts"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _candidate_filenames(font_family: str) -> frozenset[str]:
    known = set(_FONT_FILENAMES.get(font_family, frozenset()))
    compact = re.sub(r"\s+", "", font_family).lower()
    known.update({f"{compact}.ttf", f"{compact}.otf", f"{compact}.ttc"})
    return frozenset(known)


def find_font_path(
    font_family: str, font_dirs: Sequence[str | Path] | None = None
) -> Path | None:
    """Return a real font file for *font_family*, or None when not found."""
    if not isinstance(font_family, str) or not font_family.strip():
        raise ValueError("font_family must be a non-empty string")
    candidates = _candidate_filenames(font_family)
    for directory in _font_directories(font_dirs):
        if not directory.is_dir():
            continue
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name.lower())
        except OSError as exc:
            logger.debug("Could not list font directory %s: %s", directory, exc)
            continue
        for entry in entries:
            if entry.name.lower() in candidates and entry.is_file():
                return entry
    return None


def _parse_filter_names(stdout: str) -> set[str]:
    """Extract filter names from ``ffmpeg -filters`` output."""
    names: set[str] = set()
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[A-Z.]{1,4}", parts[0]):
            names.add(parts[1].lower())
    return names


def check_capabilities(
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    command_runner: CommandRunner | None = None,
    font_dirs: Sequence[str | Path] | None = None,
) -> SubtitleCapabilities:
    """Probe FFmpeg filters and scan font directories; report honest results."""
    notes: list[str] = []
    runner = command_runner if command_runner is not None else _default_command_runner
    names: set[str] = set()
    try:
        result = runner([ffmpeg_bin, "-hide_banner", "-filters"], CAPABILITY_TIMEOUT_S)
        returncode = int(getattr(result, "returncode", 0))
        stdout = str(getattr(result, "stdout", "") or "")
        if returncode != 0:
            notes.append(f"FFmpeg filter probe exited with code {returncode}")
        names = _parse_filter_names(stdout)
    except Exception as exc:  # noqa: BLE001 - probing must never abort generation
        logger.debug("FFmpeg filter probe failed: %s", exc)
        notes.append(f"FFmpeg filter probe failed: {exc}")

    libass = "ass" in names or "subtitles" in names
    burn_in_supported = "subtitles" in names or "ass" in names
    if "subtitles" not in names:
        notes.append("FFmpeg 'subtitles' filter was not found")
    if "ass" not in names:
        notes.append("FFmpeg 'ass' filter was not found")

    fonts: list[str] = []
    for family in DEVANAGARI_FONT_FAMILIES:
        if find_font_path(family, font_dirs) is not None:
            fonts.append(family)
    devanagari_font = fonts[0] if fonts else None
    if devanagari_font is None:
        notes.append("No Devanagari-capable font found in scanned directories")

    return SubtitleCapabilities(
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        libass=libass,
        burn_in_supported=burn_in_supported,
        fonts=tuple(fonts),
        devanagari_font=devanagari_font,
        notes=tuple(notes),
    )


def escape_filter_path(path: str | Path) -> str:
    """Posix-normalize a Windows path and escape FFmpeg filtergraph characters."""
    normalized = str(path).replace("\\", "/")
    pieces: list[str] = []
    for char in normalized:
        if char == ":":
            pieces.append(_COLON_ESCAPE)
        elif char == "'":
            pieces.append(_QUOTE_ESCAPE)
        elif char in _FILTER_SPECIALS:
            pieces.append(f"\\{char}")
        else:
            pieces.append(char)
    return "".join(pieces)


def _escape_style_value(value: object) -> str:
    text = str(value)
    return "".join(f"\\{char}" if char in _STYLE_SPECIALS else char for char in text)


def _ass_color_with_opacity(hex_color: str, opacity_percent: int) -> str:
    opaque = ass_color(hex_color)
    alpha = 255 - round(255 * opacity_percent / 100)
    return f"&H{alpha:02X}{opaque[4:]}"


def build_burn_in_filter(subtitle_path: str | Path, style: SubtitleStyle) -> str:
    """Build an FFmpeg ``subtitles`` filter with a forced ASS style."""
    if not isinstance(style, SubtitleStyle):
        raise TypeError("style must be a SubtitleStyle")
    alignment = _ALIGNMENTS.get(style.position, _ALIGNMENTS["bottom"])
    border_style = 3 if style.background else 1
    parts = (
        f"FontName={_escape_style_value(style.font_family)}",
        f"FontSize={style.font_size}",
        f"PrimaryColour={ass_color(style.font_color)}",
        f"OutlineColour={ass_color(style.outline_color)}",
        f"BackColour={_ass_color_with_opacity(style.background_color, style.background_opacity)}",
        f"BorderStyle={border_style}",
        f"Outline={style.outline}",
        f"Shadow={style.shadow}",
        f"Bold={-1 if style.bold else 0}",
        f"Alignment={alignment}",
        f"MarginL={style.margin_l}",
        f"MarginR={style.margin_r}",
        f"MarginV={style.margin_v}",
    )
    return (
        f"subtitles=filename={escape_filter_path(subtitle_path)}:"
        f"force_style='{','.join(parts)}'"
    )


def write_subtitle_file(path: str | Path, content: str) -> Path:
    """Atomically write UTF-8 subtitle content, refusing to replace a file."""
    target = Path(path)
    if target.exists():
        raise SubtitleGenerationError(
            "subtitle.file.collision",
            f"Subtitle file already exists: {target}",
            str(target),
        )
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    # SRT benefits from a BOM for legacy Windows players; WebVTT must not have one.
    encoding = "utf-8-sig" if target.suffix.lower() == ".srt" else "utf-8"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise SubtitleGenerationError(
                "subtitle.file.collision",
                f"Subtitle file already exists: {target}",
                str(target),
            )
        temp_path.replace(target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    logger.info("Wrote subtitle file %s (%d bytes)", target, len(content.encode("utf-8")))
    return target
