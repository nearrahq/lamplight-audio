# Usage (from ~/Developer/lamplight): ~/Developer/tts/.venv/bin/python scripts/voice-chapters.py [--only web|kjv] [--books JHN,PSA] [--limit N]
#   Writes chapter narration to audio-chapters/<tr>/<book n>/<chapter>.m4a and verse start times to
#   audio-chapters/<tr>/<book n>/t.json {"<chapter>": [seconds where verse 1, 2, ... begins]}.
#   Same voice and mastering as voice-bible.py (it imports that file). Re-runnable: finished chapters are kept.
#   audio-chapters/ is NOT in the app bundle: it is deployed on its own (audio-site/, `npm run deploy:audio`).
"""Whole-chapter narration for the reader, the chapter sheet and the plans.

Each verse is said on its own (so a verse never runs into the next and its start time is exact), joined with a
short breath, then the chapter is mastered once (one global gain, so the times hold)."""

import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("vb", ROOT / "scripts" / "voice-bible.py")
vb = importlib.util.module_from_spec(spec); spec.loader.exec_module(vb)

OUT = Path(os.environ.get("CHAPTERS_OUT", ROOT / "audio-chapters"))
if os.environ.get("KOKORO_MODEL"):
    vb.MODEL, vb.VOICES = Path(os.environ["KOKORO_MODEL"]), Path(os.environ["KOKORO_VOICES"])
GAP = 0.28           # breath between verses, on top of each verse's own edges
vb.BITRATE = "40k"   # speech, mono: whole chapters stay small


def render(tts, verses, voice, lang, speed):
    parts, starts, t = [], [], 0.0
    for v in verses:
        if not v:
            starts.append(round(t, 2)); continue
        a = vb.speak(tts, vb.spoken(v), voice, lang, speed)
        starts.append(round(t, 2))
        parts += [a, vb.silence(GAP)]
        t += (len(a) + round(GAP * vb.SR)) / vb.SR
    return np.concatenate(parts[:-1]), starts


def groups(n):
    """Books split into n shares of about equal verse count (largest first, each to the lightest share)."""
    shares = [[0, []] for _ in range(n)]
    for bk in sorted(vb.BOOKS, key=lambda b: -sum(b["ch"])):
        s = min(shares, key=lambda x: x[0]); s[0] += sum(bk["ch"]); s[1].append(bk)
    return [sorted(s[1], key=lambda b: b["n"]) for s in shares]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=vb.TRS)
    ap.add_argument("--books", default="", help="book ids in order, e.g. JHN,PSA (default: whole Bible)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--group", default="", help="i/n: render share i of n (books split evenly by verse count)")
    args = ap.parse_args()
    from kokoro_onnx import Kokoro
    tts = Kokoro(str(vb.MODEL), str(vb.VOICES))
    voice, lang, speed = vb.VOICE, vb.LANG, vb.SPEED
    aac = vb.aac_options()
    order = [vb.BY_ID[b] for b in args.books.split(",") if b] or vb.BOOKS
    if args.group:
        i, n = map(int, args.group.split("/"))
        order = groups(n)[i]
        print("books:", " ".join(b["id"] for b in order), flush=True)
    made = 0
    for tr in vb.TRS:
        if args.only and tr != args.only: continue
        for bk in order:
            d = OUT / tr / str(bk["n"]); d.mkdir(parents=True, exist_ok=True)
            tpath = d / "t.json"
            times = json.loads(tpath.read_text()) if tpath.exists() else {}
            for c, verses in enumerate(vb.book(tr, bk["n"]), 1):
                out = d / f"{c}.m4a"
                if out.exists() and str(c) in times: continue
                if args.limit and made >= args.limit: return 0
                t0 = time.time()
                audio, starts = render(tts, verses, voice, lang, speed)
                vb.master(audio, out, aac)
                times[str(c)] = starts
                tpath.write_text(json.dumps(times, separators=(",", ":")))
                made += 1
                print(f"{tr} {bk['id']} {c}: {len(audio) / vb.SR / 60:.1f} min audio in {time.time() - t0:.0f} s, "
                      f"{out.stat().st_size // 1024} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
