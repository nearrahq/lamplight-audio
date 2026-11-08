# Usage (from ~/Developer/lamplight): ~/Developer/tts/.venv/bin/python scripts/voice-bible.py
#   Writes public/audio/{web,kjv}/<key>.m4a for every For you verse (src/verses.json) and every collection passage
#   (src/topics.json), plus public/audio/manifest.json {"voice", "web": {key: seconds}, "kjv": {...}}.
#   Re-runnable: a file that already exists is kept (its length comes from the old manifest or ffprobe).
#   --only web|kjv renders one translation, --limit N stops after N new files, --force re-renders everything.
"""Narration for Lamplight's verses with Kokoro-82M (kokoro-onnx on CPU, no network, no keys).

key = String(n) for one verse, `${n}-${to}` for a passage (store.ts saveKey), n being the global verse number.
A passage in both the deck and a collection is rendered once. Deck text is src/verses.json as shown on the cards;
collection text is the Bible files' verses joined with a space and tidied the same way build-verses.py tidies a
card (a psalm's title line goes, unpaired quote marks go). For speech: curly quotes straightened, LORD/GOD said
as words, brackets and parentheses dropped, dashes become a comma pause. Only the verse text is spoken.

Same mastering as ~/Developer/guides/scripts/voice.py: trim to EDGE of silence, a smooth lookahead limiter,
two-pass loudnorm to -16 LUFS with linear gain, mono AAC, the limit tightening until true peak <= TP_MAX.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
TTS = Path.home() / "Developer" / "tts"
MODEL = TTS / "models" / "kokoro-v1.0.fp16.onnx"
VOICES = TTS / "models" / "voices-v1.0.bin"
FFMPEG = shutil.which("ffmpeg") or str(Path.home() / ".local" / "bin" / "ffmpeg")
FFPROBE = shutil.which("ffprobe") or str(Path.home() / ".local" / "bin" / "ffprobe")
OUT = ROOT / "public" / "audio"

SR = 24000
VOICE, LANG, SPEED = "am_michael", "en-us", 0.88
FALLBACK = ("bm_george", "en-gb")
BITRATE = "48k"
EDGE = 0.12
HOLD = 0.02
PLR = 12.5
TP_MAX = -1.7
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
PCM = ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0"]
TRS = ("web", "kjv")

BOOKS = json.loads((ROOT / "src" / "books.json").read_text(encoding="utf-8"))
BY_ID = {b["id"]: b for b in BOOKS}


# --- passages -------------------------------------------------------------------------------------------------

def glob(b, c, v):
    bk = BOOKS[b - 1]
    return bk["first"] + sum(bk["ch"][: c - 1]) + v - 1


def save_key(n, to=None):
    return f"{n}-{to}" if to and to > n else str(n)


HEADING = re.compile(r"^(A |For the |To |By |Of |According to )[^.]*\.\s*")


def card(t, psalm_first):
    """build-verses.py's tidy for a verse shown on its own."""
    if psalm_first:
        while HEADING.match(t) and HEADING.sub("", t, 1):
            t = HEADING.sub("", t, 1)
    for o, c in (("“", "”"), ("‘", "’")):
        if t.count(o) < t.count(c) and t.rstrip().endswith(c):
            t = t.rstrip()[:-1]
        if t.count(o) > t.count(c) and t.startswith(o):
            t = t[1:]
    t = t.strip()
    t = t[0].upper() + t[1:]
    return re.sub(r"[,;:]$", ".", t)


_books = {}


def book(tr, b):
    if (tr, b) not in _books:
        _books[tr, b] = json.loads((ROOT / "public" / "b" / tr / f"{b}.json").read_text(encoding="utf-8"))
    return _books[tr, b]


def passages():
    """{key: {"ref": str, "web": text, "kjv": text}} in deck order, then collection order."""
    out = {}
    for x in json.loads((ROOT / "src" / "verses.json").read_text(encoding="utf-8")):
        out[save_key(x["n"], x.get("to"))] = {"ref": x["ref"], "web": x["web"], "kjv": x["kjv"]}
    for topic in json.loads((ROOT / "src" / "topics.json").read_text(encoding="utf-8")):
        for ref in topic["refs"]:
            code, cv = ref.split(" ")
            c, v = cv.split(":")
            a, _, z = v.partition("-")
            bk, c, a = BY_ID[code], int(c), int(a)
            z = int(z or a)
            key = save_key(glob(bk["n"], c, a), glob(bk["n"], c, z))
            if key in out:
                continue
            p = {"ref": ref}
            for tr in TRS:
                text = " ".join(t for t in book(tr, bk["n"])[c - 1][a - 1 : z] if t)
                p[tr] = card(text, code == "PSA" and a == 1)
            out[key] = p
    return out


QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "Æ": "Ae", "æ": "ae"})


def spoken(text):
    """The verse as it should be said."""
    t = text.translate(QUOTES)
    t = re.sub(r"\bLORD\b", "Lord", t)
    t = re.sub(r"\bGOD\b", "God", t)
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"[\[\]()]", "", t)
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    t = re.sub(r"([,;:])\s*([,;:])", r"\1", t)
    t = re.sub(r"[,;:]$", ".", " ".join(t.split()))  # "(... sight:)" ends in a colon once the brackets go
    return t if t[-1] in ".!?\"'" else t + "."


# --- audio (as in guides/scripts/voice.py) --------------------------------------------------------------------

def silence(seconds):
    return np.zeros(round(seconds * SR), np.float32)


def frames_db(x):
    n = len(x) // (SR // 100)
    frames = x[: n * (SR // 100)].reshape(n, -1).astype(np.float64)
    return 20 * np.log10(np.sqrt((frames**2).mean(1)) + 1e-9)


def trim(x):
    db = frames_db(x)
    loud = np.flatnonzero(db > db.max() - 45)
    f = SR // 100
    y = x[max(0, (loud[0] - 1) * f) : (loud[-1] + 3) * f].astype(np.float32)
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))).astype(np.float32)
    y[:f] *= ramp
    y[-f:] *= ramp[::-1]
    return y


def speak(tts, text, voice, lang, speed):
    audio = trim(tts.create(text, voice=voice, speed=speed, lang=lang)[0])
    return np.concatenate([silence(EDGE - 0.01), audio, silence(EDGE - 0.02)])


def peak_hold(x):
    n, h = len(x), round(HOLD * SR)
    tp = np.abs(np.fft.irfft(np.fft.rfft(x), 4 * n) * 4).reshape(n, 4).max(1)
    return sliding_window_view(np.pad(tp, h), 2 * h + 1).max(1)


def limit(x, held, ceiling_db):
    gain = np.minimum(1.0, 10 ** (ceiling_db / 20) / np.maximum(held, 1e-9))
    h = round(HOLD * SR)
    box = np.ones(h // 2) / (h // 2)
    for _ in range(2):
        gain = np.convolve(np.pad(gain, h, mode="edge"), box, mode="same")[h:-h]
    return (x * gain).astype(np.float32), float(20 * np.log10(gain.min()))


def ffmpeg(args, data=None):
    io = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
    run = subprocess.run([FFMPEG, "-hide_banner", "-nostats", *args], capture_output=True, **io)
    if run.returncode:
        raise RuntimeError(run.stderr.decode(errors="replace")[-1500:])
    return run


def report_json(stderr):
    text = stderr.decode(errors="replace")
    return json.loads(text[text.rindex("{") : text.rindex("}") + 1])


def loudness(args, data=None):
    return report_json(ffmpeg([*args, "-af", f"{LOUDNORM}:print_format=json", "-f", "null", "-"], data).stderr)


def aac_options():
    codec = "aac_at" if re.search(r"\baac_at\b", ffmpeg(["-encoders"]).stdout.decode()) else "aac"
    return ["-ac", "1", "-c:a", codec, "-b:a", BITRATE, "-movflags", "+faststart", "-map_metadata", "-1"]


def encode(audio, out, aac):
    raw = audio.astype("<f4").tobytes()
    m = loudness(PCM, raw)
    af = (
        f"{LOUDNORM}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
        f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true:print_format=json"
        f",aresample={SR}"
    )
    return report_json(ffmpeg([*PCM, "-af", af, *aac, "-y", str(out)], raw).stderr)["normalization_type"]


def check(out, n):
    pcm = np.frombuffer(ffmpeg(["-i", str(out), "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"]).stdout, "<f4")
    if len(pcm) < n - SR // 100:
        raise RuntimeError(f"{out} decodes to {len(pcm)} samples, expected {n}")
    pcm = pcm[:n]
    m = loudness(["-i", str(out)])
    db = frames_db(pcm)
    loud = np.flatnonzero(db > db.max() - 45)
    return {
        "ms": round(1000 * n / SR),
        "lufs": float(m["input_i"]),
        "tp": float(m["input_tp"]),
        "lead": 10 * int(loud[0]),
        "trail": 10 * int(len(db) - 1 - loud[-1]),
    }


def master(audio, out, aac):
    tmp = out.with_suffix(".tmp.m4a")
    loud = float(loudness(PCM, audio.astype("<f4").tobytes())["input_i"])
    held, plr = peak_hold(audio), PLR
    best = None
    for _ in range(12):
        limited, cut = limit(audio, held, loud + plr)
        mode = encode(limited, tmp, aac)
        c = check(tmp, len(limited))
        if mode == "linear" and c["tp"] <= TP_MAX:
            tmp.replace(out)  # only a finished, checked file gets the real name, so a rerun never keeps a half
            return c, cut
        if mode == "linear" and c["tp"] <= -1.5 and (best is None or c["tp"] < best[0]["tp"]):
            best = (c, cut, tmp.read_bytes())  # inside the -1.5 dBTP limit, just not the 0.2 dB margin
        plr -= max(0.3, c["tp"] - TP_MAX + 0.2)
    tmp.unlink(missing_ok=True)
    if best:  # a short line's AAC overshoot can wobble around the margin; take the best try inside the limit
        out.write_bytes(best[2])
        return best[0], best[1]
    raise RuntimeError(f"true peak {c['tp']} dBTP ({mode} gain) after 12 tries")


def probe_seconds(path):
    run = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    return round(float(run.stdout.strip()), 2)


def key_order(k):
    a, _, z = k.partition("-")
    return int(a), int(z or a)


def main():
    ap = argparse.ArgumentParser(description="Render Lamplight's verse narration with Kokoro-82M.")
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--lang", default=LANG)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--only", choices=TRS)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new files")
    ap.add_argument("--force", action="store_true", help="re-render files that already exist")
    args = ap.parse_args()

    tts = Kokoro(str(MODEL), str(VOICES))
    voice, lang = args.voice, args.lang
    try:
        tts.create("Peace.", voice=voice, speed=args.speed, lang=lang)
    except Exception as e:  # noqa: BLE001
        print(f"{voice} failed ({e}); using {FALLBACK[0]}")
        voice, lang = FALLBACK

    mpath = OUT / "manifest.json"
    old = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    if old.get("voice") and old["voice"] != voice and not args.force:
        print(f"note: existing files were said by {old['voice']}; pass --force to re-render them all as {voice}")
    manifest = {"voice": old.get("voice") or voice, "web": dict(old.get("web", {})), "kjv": dict(old.get("kjv", {}))}
    if args.force:
        manifest["voice"] = voice

    def save():
        for tr in TRS:
            manifest[tr] = {k: manifest[tr][k] for k in sorted(manifest[tr], key=key_order)}
        mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    aac = aac_options()
    items = passages()
    print(f"{len(items)} passages, encoder {aac[3]} {BITRATE}, voice {voice} ({lang}), speed {args.speed}")
    print(f"{'tr':4}{'key':13}{'ref':18}{'words':>6}{'s':>7}{'LUFS':>7}{'dBTP':>6}{'cut':>6}{'lead':>5}{'trail':>6}")
    failed, problems, made = [], [], 0
    for tr in TRS:
        if args.only and tr != args.only:
            continue
        (OUT / tr).mkdir(parents=True, exist_ok=True)
        for key, p in items.items():
            out = OUT / tr / f"{key}.m4a"
            if out.exists() and not args.force:
                if key not in manifest[tr]:
                    manifest[tr][key] = probe_seconds(out)
                continue
            if args.limit and made >= args.limit:
                break
            text = spoken(p[tr])
            try:
                c, cut = master(speak(tts, text, voice, lang, args.speed), out, aac)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{tr}/{key} {p['ref']}: {e}")
                print(f"FAILED {tr}/{key}: {e}")
                continue
            made += 1
            manifest[tr][key] = round(c["ms"] / 1000, 2)
            save()
            print(f"{tr:4}{key:13}{p['ref'][:17]:18}{len(text.split()):6d}{c['ms'] / 1000:7.2f}{c['lufs']:7.1f}"
                  f"{c['tp']:6.1f}{cut:6.1f}{c['lead']:5d}{c['trail']:6d}", flush=True)
            if abs(c["lufs"] + 16) > 1.5 or c["tp"] > -1.5:
                problems.append(f"{tr}/{key}: {c['lufs']} LUFS, {c['tp']} dBTP")
            per_word = c["ms"] / 1000 / len(text.split())
            if not 0.25 <= per_word <= 0.75 or not 60 <= c["lead"] <= 250 or not 60 <= c["trail"] <= 250:
                problems.append(f"{tr}/{key}: {per_word:.3f} s a word, edges {c['lead']}/{c['trail']} ms")
    save()

    for tr in TRS:
        files = list((OUT / tr).glob("*.m4a")) if (OUT / tr).exists() else []
        d = list(manifest[tr].values())
        size = sum(f.stat().st_size for f in files)
        if d:
            print(f"{tr}: {len(files)} files, {size / 1e6:.1f} MB, {sum(d) / 60:.1f} min, "
                  f"min {min(d):.2f} s, max {max(d):.2f} s, mean {sum(d) / len(d):.2f} s")
    print(f"{made} rendered this run")
    print("\n".join(failed + problems) or "all checks passed")
    return 1 if failed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
