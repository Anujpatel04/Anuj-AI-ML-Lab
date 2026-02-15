#!/usr/bin/env python3
"""
OpenAI Sora short video generator (20–30 seconds).
Uses OPENAI_API_KEY from project .env. Generates 2–3 clips and concatenates with ffmpeg.
Uses REST API directly so it works with any openai package version.
"""

import json
import os
import sys
import time
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.openai.com/v1"

# Load .env from project root (parent of AI_AGENTS)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

def load_env():
    """Load OPENAI_API_KEY from .env without adding project to path."""
    if not DOTENV_PATH.is_file():
        raise FileNotFoundError(f".env not found at {DOTENV_PATH}")
    env = {}
    with open(DOTENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1].replace('\\"', '"')
            elif v.startswith("'") and v.endswith("'"):
                v = v[1:-1].replace("\\'", "'")
            env[k] = v
    return env

def get_api_key():
    env = load_env()
    key = env.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return key

def _api_call(req: urllib.request.Request) -> bytes:
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body)
            msg = err.get("error", {}).get("message", body) if isinstance(err.get("error"), dict) else body
        except Exception:
            msg = body or str(e)
        raise RuntimeError(f"API error {e.code}: {msg}") from e

def create_video(api_key: str, prompt: str, seconds: str = "12", model: str = "sora-2", size: str = "1280x720") -> dict:
    """Start a video generation job via POST /v1/videos (multipart/form-data)."""
    boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n{prompt}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="seconds"\r\n\r\n{seconds}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="size"\r\n\r\n{size}\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(f"{BASE_URL}/videos", data=body, headers=headers, method="POST")
    return json.loads(_api_call(req).decode())

def get_video_status(api_key: str, video_id: str) -> dict:
    """GET /v1/videos/{video_id}"""
    req = urllib.request.Request(
        f"{BASE_URL}/videos/{video_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return json.loads(_api_call(req).decode())

def wait_for_video(api_key: str, video_id: str, poll_interval: int = 15) -> dict:
    """Poll until job is completed or failed."""
    while True:
        video = get_video_status(api_key, video_id)
        status = video.get("status", "")
        if status == "completed":
            return video
        if status == "failed":
            err = video.get("error") or {}
            msg = err.get("message", str(err))
            raise RuntimeError(f"Video generation failed: {msg}")
        progress = video.get("progress", 0)
        print(f"  Status: {status}, progress: {progress}%")
        time.sleep(poll_interval)

def download_video(api_key: str, video_id: str) -> bytes:
    """Download MP4: GET /v1/videos/{video_id}/content"""
    req = urllib.request.Request(
        f"{BASE_URL}/videos/{video_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return _api_call(req)

def concatenate_mp4s(paths: list, out_path: str) -> bool:
    """Concatenate MP4 files using ffmpeg. Returns True on success."""
    if not paths:
        return False
    if len(paths) == 1:
        import shutil
        shutil.copy(paths[0], out_path)
        return True
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        for p in paths:
            list_file.write(f"file '{os.path.abspath(p)}'\n")
        list_file.close()
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file.name, "-c", "copy", out_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        try:
            os.unlink(list_file.name)
        except Exception:
            pass

def generate_short_video(
    prompt: str,
    output_path: str = "output.mp4",
    duration_seconds: int = 24,
    model: str = "sora-2",
    size: str = "1280x720",
    poll_interval: int = 15,
):
    """
    Generate a short video (20–30 seconds) from a text prompt.

    duration_seconds: target total length (20–30). We use 12s clips and optionally 8s; default 24.
    """
    if duration_seconds < 20:
        duration_seconds = 20
    if duration_seconds > 30:
        duration_seconds = 30

    # Clip lengths: use 12s as much as possible, then 8s
    clips = []
    remaining = duration_seconds
    while remaining >= 12:
        clips.append("12")
        remaining -= 12
    if remaining >= 8:
        clips.append("8")
        remaining -= 8
    elif remaining >= 4:
        clips.append("4")
        remaining -= 4
    if not clips:
        clips = ["12"]

    api_key = get_api_key()
    total_clips = len(clips)
    print(f"Generating {total_clips} clip(s): {', '.join(clips)}s → ~{sum(int(c) for c in clips)}s total")
    print(f"Model: {model}, size: {size}\n")

    part_prompts = []
    if total_clips == 1:
        part_prompts = [prompt]
    else:
        for i in range(total_clips):
            part_prompts.append(
                f"[Part {i+1}/{total_clips}] {prompt}"
                if total_clips > 1 else prompt
            )

    temp_dir = tempfile.mkdtemp()
    clip_paths = []
    try:
        for i, (sec, part_prompt) in enumerate(zip(clips, part_prompts)):
            print(f"Clip {i+1}/{total_clips} ({sec}s)...")
            job = create_video(api_key, part_prompt, seconds=sec, model=model, size=size)
            video_id = job["id"]
            wait_for_video(api_key, video_id, poll_interval=poll_interval)
            data = download_video(api_key, video_id)
            path = os.path.join(temp_dir, f"clip_{i}.mp4")
            with open(path, "wb") as f:
                f.write(data)
            clip_paths.append(path)
            print(f"  Saved clip {i+1}.\n")

        out_abs = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
        if concatenate_mp4s(clip_paths, out_abs):
            print(f"Done. Video saved to: {out_abs}")
        else:
            # Fallback: copy first clip if ffmpeg missing
            with open(clip_paths[0], "rb") as f:
                with open(out_abs, "wb") as out:
                    out.write(f.read())
            print(f"ffmpeg not found; saved single clip to: {out_abs}")
    finally:
        for p in clip_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python short_video_generator.py <prompt> [output.mp4]")
        print("Example: python short_video_generator.py 'A calico cat playing piano on stage, soft lighting' my_video.mp4")
        sys.exit(1)
    prompt = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    generate_short_video(prompt, output_path=output)

if __name__ == "__main__":
    main()
