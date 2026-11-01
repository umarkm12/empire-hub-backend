"""Empire Hub Backend — video link to scene-by-scene recreation prompt.

Paste a video link (YouTube, TikTok, Instagram, Facebook, or a direct
MP4 link) or upload a video file. The app sends the video to Google's
Gemini model, which watches it and returns a timestamped scene-by-scene
breakdown plus recreation prompts you can paste into any AI video tool
(Veo, Runway, Kling, Sora, Pika, Luma).

The Gemini API key is read ONLY from the GEMINI_API_KEY environment
variable (on Hugging Face Spaces: a Space secret with that name).
No key is stored in this file. If the key is missing, the app shows a
clear message instead of crashing.
"""

import os
import re
import tempfile
import time
from urllib.parse import urlparse

import gradio as gr
import yt_dlp
from google import genai
from google.genai import types

# Model used for video understanding. Can be overridden with the
# GEMINI_MODEL environment variable, but the default is fine.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# The prompt generator is asked for at most this many scenes, so the
# output UI (one copy box per scene) stays simple.
MAX_SCENES = 12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_youtube_url(url):
    """True for YouTube watch/shorts/embed/live links (incl. youtu.be).

    Uses real URL parsing (not substring matching) so look-alike domains
    such as notyoutube.com do not count.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    host = (parsed.netloc or "").lower().split("@")[-1].split(":")[0]
    if host == "youtu.be":
        return bool(parsed.path.strip("/"))
    if host in ("youtube.com", "www.youtube.com", "music.youtube.com"):
        if parsed.path == "/watch":
            return True
        return any(
            parsed.path.startswith(prefix)
            for prefix in ("/shorts/", "/embed/", "/live/", "/v/")
        )
    return False


def get_client():
    """Build a Gemini client from the environment key, or raise a clear error."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Gemini API key is not set. Add GEMINI_API_KEY as a secret "
            "(on Hugging Face Spaces: Settings > Variables and secrets), "
            "then try again. No key is stored in this app."
        )
    return genai.Client(api_key=key)


def build_analysis_prompt():
    """The instruction sent to Gemini along with the video."""
    return (
        "You are a video analyst. Watch this video carefully from start to end.\n"
        "Return a scene-by-scene breakdown and recreation prompts. "
        "Use this exact format:\n\n"
        "MASTER PROMPT:\n"
        "[One paragraph that recreates the whole video, in English, for AI video "
        "tools: overall style, the main subject kept identical in every scene, "
        "setting, story flow, mood. No timestamps here.]\n\n"
        "NEGATIVE PROMPT:\n"
        "[Comma-separated things to avoid, adapted to this video, e.g. blurry, "
        "distorted face, extra fingers, changing clothes, watermark, flickering.]\n\n"
        "SCENE 1 (00:00-00:08):\n"
        "Visual: [What is visible: subjects, what they do, setting, camera "
        "angle and movement, lighting, mood. English, one detailed paragraph.]\n"
        "Dialogue: [Exact spoken lines in this scene, in the video's original "
        "language. If there is no dialogue, write: None.]\n"
        "Recreation prompt: [One detailed English prompt that recreates ONLY "
        "this scene, ready to paste into Veo, Runway, Kling, Sora, Pika or Luma. "
        "Keep the main subject identical to the master prompt.]\n\n"
        "SCENE 2 (00:08-00:15):\n"
        "[Same three lines: Visual, Dialogue, Recreation prompt.]\n\n"
        "Rules:\n"
        "- Use only what is really in the video. Never invent scenes, dialogue, "
        "or details that are not there.\n"
        "- If a scene has no dialogue, write None for Dialogue.\n"
        "- Divide the video into at most 12 scenes, merging short shots.\n"
        "- Timestamps must cover the video from start to end without gaps.\n"
        "- Prompts in English. Dialogue lines in the original language."
    )


def download_video(url, tmpdir):
    """Download a non-YouTube link with yt-dlp. Returns the local file path.

    Raises a friendly error if the platform blocks the download.
    """
    try:
        import imageio_ffmpeg
        _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _ffmpeg = None
    ydl_opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "ffmpeg_location": _ffmpeg,
        "outtmpl": os.path.join(tmpdir, "video.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:  # yt-dlp DownloadError and friends
        detail = str(exc).strip().split("\n")[0][:200]
        raise RuntimeError(
            "Could not download this video. The platform may require login "
            "or may be blocking downloads right now. Please download the video "
            "yourself and use the file upload below instead. "
            "Details: " + detail
        )
    # Find the downloaded file (merge can change the extension).
    candidates = []
    for name in os.listdir(tmpdir):
        if name.endswith((".part", ".ytdl", ".temp")):
            continue
        path = os.path.join(tmpdir, name)
        if os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        raise RuntimeError(
            "Download finished but no video file was found. Please try the "
            "file upload below instead."
        )
    path = max(candidates, key=os.path.getsize)
    if os.path.getsize(path) > 2 * 1024 * 1024 * 1024:
        raise RuntimeError(
            "This video is larger than 2 GB, which the Gemini API does not "
            "accept. Please use a shorter or smaller clip."
        )
    return path


def upload_and_wait(client, path):
    """Upload a local file to Gemini and wait until it is ready to use."""
    uploaded = client.files.upload(file=path)
    deadline = time.time() + 300  # up to 5 minutes for large files
    while True:
        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError(
                "Gemini could not process this video file. "
                "Try a shorter clip or a different format (MP4 works best)."
            )
        if time.time() > deadline:
            raise RuntimeError(
                "Gemini is taking too long to process this video. "
                "Please try a shorter clip."
            )
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)


def call_gemini(client, video, prompt):
    """Send the video (File object or YouTube URI part) to Gemini."""
    response = client.models.generate_content(
        model=MODEL, contents=[video, prompt]
    )
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError(
            "Gemini returned no text for this video. It may be blocked or "
            "unsupported. Please try a different video."
        )
    return text.strip()


# ---------------------------------------------------------------------------
# Output parsing (so each scene gets its own copy box)
# ---------------------------------------------------------------------------

SCENE_HEADER_RE = re.compile(
    r"(?im)^SCENE\s+(\d+)\s*\(([^)]*)\)\s*:?\s*$"
)


def parse_scenes(full_text):
    """Split Gemini's output into [(label, scene_text), ...].

    Returns an empty list if the output does not follow the scene format.
    """
    matches = list(SCENE_HEADER_RE.finditer(full_text))
    if not matches:
        return []
    scenes = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        chunk = full_text[start:end].strip()
        label = "Scene %s (%s)" % (match.group(1), match.group(2))
        scenes.append((label, chunk))
    return scenes[:MAX_SCENES]


def extract_master(full_text):
    """Extract the MASTER PROMPT section, or return the full text as fallback."""
    match = re.search(
        r"MASTER PROMPT:\s*(.*?)(?=\nNEGATIVE PROMPT:|\nSCENE\s+\d+|\Z)",
        full_text,
        re.DOTALL,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return full_text


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def generate(link, upload):
    """Run the whole workflow. Returns UI outputs; never raises to the user."""
    link = (link or "").strip()
    if not link and not upload:
        return (
            "Paste a video link or upload a video file, then press Generate.",
            "",
            None,
            *[gr.update(value="", visible=False) for _ in range(MAX_SCENES)],
        )

    try:
        client = get_client()
    except RuntimeError as exc:
        return (
            str(exc),
            "",
            None,
            *[gr.update(value="", visible=False) for _ in range(MAX_SCENES)],
        )

    tmpdir = tempfile.mkdtemp(prefix="empirehub-")
    uploaded_remote = None
    try:
        prompt = build_analysis_prompt()
        if link and is_youtube_url(link):
            # YouTube URLs go straight to Gemini — no download needed.
            # Only public videos work this way.
            video_part = types.Part.from_uri(
                file_uri=link, mime_type="video/mp4"
            )
            full_text = call_gemini(client, video_part, prompt)
        else:
            if link:
                local_path = download_video(link, tmpdir)
            else:
                local_path = upload  # Gradio gives a local temp path
            uploaded_remote = upload_and_wait(client, local_path)
            try:
                full_text = call_gemini(client, uploaded_remote, prompt)
            finally:
                try:
                    client.files.delete(name=uploaded_remote.name)
                except Exception:
                    pass
                uploaded_remote = None
    except RuntimeError as exc:
        return (
            str(exc),
            "",
            None,
            *[gr.update(value="", visible=False) for _ in range(MAX_SCENES)],
        )
    except Exception as exc:  # any Gemini / network error, kept short
        detail = str(exc).strip().split("\n")[0][:300]
        return (
            "Something went wrong: " + detail,
            "",
            None,
            *[gr.update(value="", visible=False) for _ in range(MAX_SCENES)],
        )

    # Build outputs: master prompt box, one box per scene, .txt download.
    master = extract_master(full_text)
    scenes = parse_scenes(full_text)
    if not scenes:
        status = (
            "Done. The model did not follow the scene format, so the full "
            "result is shown below and in the download."
        )
        master_box_text = full_text
    else:
        status = "Done — %d scenes. Copy any prompt, or download the full text." % len(
            scenes
        )
        master_box_text = master

    txt_path = os.path.join(tmpdir, "empire-hub-prompt.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(full_text)

    scene_updates = []
    for i in range(MAX_SCENES):
        if i < len(scenes):
            label, text = scenes[i]
            scene_updates.append(gr.update(value=text, visible=True, label=label))
        else:
            scene_updates.append(gr.update(value="", visible=False))

    return (status, master_box_text, txt_path, *scene_updates)


# ---------------------------------------------------------------------------
# UI (single column, mobile-friendly, no login, no notes required)
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(title="Empire Hub — Video Link to Prompt") as demo:
        gr.Markdown(
            "# Empire Hub\n"
            "Paste any video link. The AI watches the video and writes the "
            "scene-by-scene prompt to remake it."
        )
        link = gr.Textbox(
            label="Video link",
            placeholder="Paste a YouTube, TikTok, Instagram, Facebook or MP4 link…",
        )
        upload = gr.File(
            label="Or upload a video file (use this if a link download is blocked)",
            file_types=["video"],
        )
        run = gr.Button("Generate prompt", variant="primary")
        status = gr.Markdown("")
        master = gr.Textbox(
            label="Master prompt",
            lines=10,
            buttons=["copy"],
        )
        scene_boxes = [
            gr.Textbox(
                label="Scene %d" % (i + 1),
                lines=6,
                visible=False,
                buttons=["copy"],
            )
            for i in range(MAX_SCENES)
        ]
        download = gr.DownloadButton("Download full prompt (.txt)", visible=False)
        path_state = gr.State(None)

        def on_generate(link_value, upload_value):
            result = generate(link_value, upload_value)
            status_text, master_text, txt_path = result[0], result[1], result[2]
            scene_updates = result[3:]
            return (
                status_text,
                master_text,
                txt_path,
                gr.update(visible=bool(txt_path)),
                *scene_updates,
            )

        run.click(
            on_generate,
            inputs=[link, upload],
            outputs=[status, master, path_state, download, *scene_boxes],
        )
        # The download button serves the .txt file made by Generate.
        download.click(
            lambda path: path,
            inputs=[path_state],
            outputs=[download],
        )
        gr.Markdown(
            "Notes: YouTube links work best and need no download. Instagram and "
            "Facebook links are often blocked by login walls — use the file "
            "upload instead. If you give both a link and a file, the link is "
            "used. Very long or large videos may fail on the AI side; "
            "try a shorter clip. This app needs a Gemini API key set as a secret."
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    # Render (and similar hosts) provide the port via $PORT; default 7860 for local/HF use.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
