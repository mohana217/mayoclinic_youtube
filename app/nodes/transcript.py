"""
transcript.py
---------------------
Fetches transcripts using a 3-tier fallback:
  1. YouTube CC / auto-captions  (youtube-transcript-api v1.x — instance API)
  2. Whisper via yt-dlp
  3. YouTube video metadata (title + description)
"""

import re
import os
import tempfile

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
    YTAPI_AVAILABLE = True
except ImportError:
    YTAPI_AVAILABLE = False
    print("[transcript_fetcher] youtube-transcript-api not installed — tier 1 disabled")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("[transcript_fetcher] yt-dlp not installed — tier 2 disabled")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[transcript_fetcher] openai-whisper not installed — tier 2 disabled")

try:
    from googleapiclient.discovery import build as yt_build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

MIN_TRANSCRIPT_WORDS = 60

def _clean(text: str) -> str:
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = text.split('. ')
    seen, deduped = set(), []
    for s in sentences:
        key = s.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return '. '.join(deduped)

def _wc(text: str) -> int:
    return len(text.split())

def _usable(text: str) -> bool:
    return _wc(_clean(text)) >= MIN_TRANSCRIPT_WORDS


# ── Tier 1: YouTube CC (v1.x instance API) ────────────────────────────────────
def _fetch_cc(video_id: str) -> dict | None:
    if not YTAPI_AVAILABLE:
        return None
    api = YouTubeTranscriptApi()  # v1.x: instantiate first
    try:
        tlist = api.list(video_id)
    except TranscriptsDisabled:
        print(f"[CC] Transcripts disabled for {video_id}")
        return None
    except NoTranscriptFound:
        print(f"[CC] No transcript for {video_id}")
        return None
    except Exception as e:
        print(f"[CC] list() error for {video_id}: {type(e).__name__}: {e}")
        return None

    transcript_obj = None
    source = "cc_auto"
    try:
        transcript_obj = tlist.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        source = "cc_manual"
    except Exception:
        pass

    if transcript_obj is None:
        try:
            transcript_obj = tlist.find_generated_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            pass

    if transcript_obj is None:
        try:
            available_langs = [t.language_code for t in tlist]
            transcript_obj = tlist.find_transcript(available_langs)
            transcript_obj = transcript_obj.translate('en')
            source = "cc_translated"
        except Exception as e:
            print(f"[CC] No usable transcript for {video_id}: {e}")
            return None

    try:
        fetched = transcript_obj.fetch()
        # v1.x: FetchedTranscript is iterable; each snippet has .text
        raw_text = ' '.join(snippet.text for snippet in fetched)
    except Exception as e:
        print(f"[CC] fetch() failed for {video_id}: {type(e).__name__}: {e}")
        return None

    cleaned = _clean(raw_text)
    if not _usable(cleaned):
        print(f"[CC] Too short ({_wc(cleaned)} words) for {video_id} — trying fallback")
        return None

    print(f"[CC] OK — source={source}, words={_wc(cleaned)}, video={video_id}")
    return {"text": cleaned, "source": source, "word_count": _wc(cleaned)}


# ── Tier 2: Whisper ───────────────────────────────────────────────────────────
_whisper_model_cache = {}

def _get_whisper_model():
    size = os.environ.get("WHISPER_MODEL_SIZE", "base")
    if size not in _whisper_model_cache:
        print(f"[Whisper] Loading model '{size}'…")
        _whisper_model_cache[size] = whisper.load_model(size)
    return _whisper_model_cache[size]

def _fetch_whisper(video_id: str) -> dict | None:
    if not YT_DLP_AVAILABLE or not WHISPER_AVAILABLE:
        print(f"[Whisper] Skipped — yt_dlp={YT_DLP_AVAILABLE}, whisper={WHISPER_AVAILABLE}")
        return None
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"[Whisper] yt-dlp failed for {video_id}: {type(e).__name__}: {e}")
            return None

        audio_path = None
        for fname in os.listdir(tmpdir):
            if fname.startswith("audio"):
                audio_path = os.path.join(tmpdir, fname)
                break

        if not audio_path:
            print(f"[Whisper] No audio file found after download for {video_id}")
            return None

        try:
            model = _get_whisper_model()
            result = model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                fp16=False,
                initial_prompt=(
                    "Patient speaking about their experience at Mayo Clinic. "
                    "Medical terminology, hospital, doctor, treatment, surgery, cancer, care."
                ),
            )
            cleaned = _clean(result.get("text", ""))
            if not _usable(cleaned):
                print(f"[Whisper] Too short ({_wc(cleaned)} words) for {video_id}")
                return None
            print(f"[Whisper] OK — words={_wc(cleaned)}, video={video_id}")
            return {"text": cleaned, "source": "whisper", "word_count": _wc(cleaned)}
        except Exception as e:
            print(f"[Whisper] Transcription failed for {video_id}: {type(e).__name__}: {e}")
            return None


# ── Tier 3: YouTube metadata ──────────────────────────────────────────────────
def _fetch_metadata(video_id: str) -> dict | None:
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key or not GOOGLE_API_AVAILABLE:
        print(f"[Metadata] Skipped — no API key or google-api-python-client missing")
        return None
    try:
        yt = yt_build("youtube", "v3", developerKey=api_key)
        resp = yt.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return None
        snippet = items[0]["snippet"]
        combined = f"Video Title: {snippet.get('title','')}\n\nDescription: {snippet.get('description','')[:2000]}"
        print(f"[Metadata] OK — words={_wc(combined)}, video={video_id}")
        return {"text": combined, "source": "metadata_only", "word_count": _wc(combined)}
    except Exception as e:
        print(f"[Metadata] Failed for {video_id}: {type(e).__name__}: {e}")
        return None


# ── LangGraph node ────────────────────────────────────────────────────────────
def fetch_transcripts(state: dict) -> dict:
    videos = state.get("videos", [])
    transcripts = []

    for video in videos:
        vid = video.get("video_id", "").strip()
        if not vid:
            continue

        print(f"\n=== Fetching transcript for: {vid} ===")
        entry = {"video_id": vid, "text": "", "source": "failed", "word_count": 0, "usable": False}

        result = _fetch_cc(vid)
        if result is None:
            result = _fetch_whisper(vid)
        if result is None:
            result = _fetch_metadata(vid)

        if result:
            entry.update(result)
            entry["usable"] = _usable(result["text"])
        else:
            print(f"[fetch_transcripts] ALL tiers failed for {vid}")

        transcripts.append(entry)

    return {**state, "transcripts": transcripts}