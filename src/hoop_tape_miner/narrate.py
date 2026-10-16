"""Turn a spoken script (plain paragraphs written for the ear) into an MP3, with a pause between paragraphs.

usage:
  hoop-tape-miner narrate workspace/teams/<opponent>/<date> [--lang zh en]
      report narration: listen.zh.md / listen.en.md -> listen.zh.mp3 / listen.en.mp3 in that folder
  hoop-tape-miner narrate --script cues.zh.md [--gap 6 --chime --loud --rate -5%]
      any single script -> cues.zh.mp3 next to it (language from the .zh/.en suffix). --chime plays a short
      two-tone bell before every paragraph and --gap leaves silence after it, which makes a loop you can
      listen to on court; --loud normalises to -14 LUFS so it cuts through gym noise.

Uses edge-tts (Microsoft neural voices, needs internet: the script text is sent to Microsoft's TTS
service) and ffmpeg ($FFMPEG overrides the binary). Lines starting with '#' are notes for the writer and are
not read. Voices: optional `narration.voices` in game.json (folder or its parent), e.g.
  "narration": {"voices": {"en": "en-GB-RyanNeural", "zh": {"voice": "zh-CN-YunxiNeural", "rate": "-5%"}}}
otherwise the defaults in VOICES.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from .game_config import load_game

VOICES = {"zh": ("zh-CN-YunjianNeural", "+0%"), "en": ("en-US-AndrewNeural", "+0%")}
PAUSE_S = 0.7
MP3 = ["-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "48k"]


def paragraphs(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return [" ".join(p.split()) for p in "\n".join(lines).split("\n\n") if p.strip()]


def voices(game: dict) -> dict[str, tuple[str, str]]:
    """VOICES overridden by game.json narration.voices ({lang: "voice"} or {lang: {"voice", "rate"}})."""
    out = dict(VOICES)
    for lang, v in ((game.get("narration") or {}).get("voices") or {}).items():
        if lang.startswith("_"):
            continue
        if isinstance(v, str):
            out[lang] = (v, out.get(lang, (v, "+0%"))[1])
        elif isinstance(v, dict) and v.get("voice"):
            out[lang] = (v["voice"], v.get("rate", "+0%"))
    return out


async def speak(paras: list[str], voice: str, rate: str, tmp: Path) -> list[Path]:
    import edge_tts

    parts = []
    for i, p in enumerate(paras):
        out = tmp / f"p{i:03d}.mp3"
        await edge_tts.Communicate(p, voice, rate=rate).save(str(out))
        parts.append(out)
    return parts


def ffmpeg(*args: str):
    subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-y", *args], check=True)


def silence(seconds: float, out: Path) -> Path:
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(seconds), *MP3, str(out))
    return out


def chime(out: Path) -> Path:
    """Two soft rising tones (A5 -> E6), then a short breath before the voice."""
    ffmpeg("-filter_complex",
           "sine=f=880:d=0.14,afade=t=in:d=0.01,afade=t=out:st=0.09:d=0.05[a];"
           "sine=f=1319:d=0.45,afade=t=in:d=0.01,afade=t=out:st=0.05:d=0.4[b];"
           "[a][b]concat=n=2:v=0:a=1,volume=0.35,apad=pad_dur=0.35",
           *MP3, str(out))
    return out


def stitch(parts: list[Path], out: Path, tmp: Path, gap: float = PAUSE_S, bell: bool = False, loud: bool = False):
    pause = silence(gap, tmp / "pause.mp3")
    ding = chime(tmp / "chime.mp3") if bell else None
    seq = []
    for p in parts:
        seq += ([ding] if ding else []) + [p, pause]
    if not bell:
        seq = seq[:-1]  # a narration ends on the last word; a cue loop keeps its gap so the repeat breathes
    listing = tmp / "list.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in seq), encoding="utf8")
    norm = ["-af", "loudnorm=I=-14:TP=-1.5:LRA=7"] if loud else []
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), *norm, *MP3, str(out))


def render(src: Path, out: Path, lang: str, gap: float, bell: bool, loud: bool, rate: str | None,
           voice_table: dict | None = None):
    paras = paragraphs(src.read_text(encoding="utf8"))
    voice, default_rate = (voice_table or VOICES)[lang]
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        parts = asyncio.run(speak(paras, voice, rate or default_rate, tmp))
        stitch(parts, out, tmp, gap, bell, loud)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB,", len(paras), "paragraphs")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", help="report folder with listen.<lang>.md")
    ap.add_argument("--lang", nargs="+", default=list(VOICES), choices=list(VOICES))
    ap.add_argument("--script", type=Path, help="a single script file instead of a report folder")
    ap.add_argument("--out", type=Path, help="output mp3 for --script (default: next to it)")
    ap.add_argument("--gap", type=float, default=PAUSE_S, help="seconds of silence after each paragraph")
    ap.add_argument("--chime", action="store_true", help="short bell before each paragraph")
    ap.add_argument("--loud", action="store_true", help="normalise loudness to -14 LUFS")
    ap.add_argument("--rate", help='speaking rate, e.g. "-5%%"')
    a = ap.parse_args(argv)
    if a.script:
        vt = voices(load_game(a.script.parent))
        lang = next((c for c in vt if a.script.name.endswith(f".{c}.md")), a.lang[0])
        render(a.script, a.out or a.script.with_suffix(".mp3"), lang, a.gap, a.chime, a.loud, a.rate, vt)
        return
    if not a.folder:
        ap.error("give a report folder or --script")
    folder = Path(a.folder)
    vt = voices(load_game(folder))
    for lang in a.lang:
        render(folder / f"listen.{lang}.md", folder / f"listen.{lang}.mp3", lang, a.gap, a.chime, a.loud, a.rate, vt)


if __name__ == "__main__":
    main()
