import asyncio
import atexit
import base64
import logging
import os
import re
import shlex
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json
import shutil

DEFAULT_ALLOWED_FRONTEND_ORIGIN = "https://you-tubevideos-downloader.vercel.app"

logger = logging.getLogger("ytdl")
logger.setLevel(logging.INFO)


def normalize_origin(value: str) -> str:
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def parse_allowed_frontend_origins() -> list[str]:
    raw = os.getenv("ALLOWED_FRONTEND_ORIGINS", DEFAULT_ALLOWED_FRONTEND_ORIGIN)
    origins = []
    for item in raw.split(","):
        origin = normalize_origin(item)
        if origin and origin not in origins:
            origins.append(origin)
    return origins or [DEFAULT_ALLOWED_FRONTEND_ORIGIN]


ALLOWED_FRONTEND_ORIGINS = parse_allowed_frontend_origins()
ALLOW_HEALTH_WITHOUT_ORIGIN = os.getenv("ALLOW_HEALTH_WITHOUT_ORIGIN", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

app = FastAPI(title="YT Downloader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
)


def request_has_allowed_frontend(request: Request) -> bool:
    origin = normalize_origin(request.headers.get("origin", ""))
    if origin in ALLOWED_FRONTEND_ORIGINS:
        return True

    referer = normalize_origin(request.headers.get("referer", ""))
    return referer in ALLOWED_FRONTEND_ORIGINS


@app.middleware("http")
async def require_allowed_frontend(request: Request, call_next):
    if ALLOW_HEALTH_WITHOUT_ORIGIN and request.url.path == "/health":
        return await call_next(request)

    if request_has_allowed_frontend(request):
        return await call_next(request)

    return JSONResponse(
        status_code=403,
        content={
            "detail": "Forbidden: this API only accepts requests from the allowed frontend origin.",
        },
    )

@app.exception_handler(HTTPException)
async def production_http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "Request failed. Please try again."
    if exc.status_code >= 500 and exc.status_code != 503:
        message = "Something went wrong while processing the request. Please try again later."
    request_id = uuid.uuid4().hex[:12]
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": message,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": message,
                "request_id": request_id,
            },
        },
    )


@app.exception_handler(Exception)
async def production_unhandled_exception_handler(request: Request, exc: Exception):
    request_id = uuid.uuid4().hex[:12]
    logger.exception("Unhandled request error %s", request_id)
    message = "Something went wrong while processing the request. Please try again later."
    return JSONResponse(
        status_code=500,
        content={
            "detail": message,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": message,
                "request_id": request_id,
            },
        },
    )


TEMP_DIR = Path(tempfile.gettempdir()) / "ytdl_temp"
TEMP_DIR.mkdir(exist_ok=True)


def get_env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), minimum)
    except ValueError:
        return default


def get_env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(float(raw), minimum)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


YT_DLP = shutil.which("yt-dlp") or "yt-dlp"


def cleanup_runtime_secret_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.debug("Could not remove runtime secret file %s", path)


def write_runtime_secret_file(name: str, content: str) -> str | None:
    cleaned = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return None

    fd, temp_path = tempfile.mkstemp(prefix=f"ytdl_{name}_", suffix=".txt", dir=TEMP_DIR)
    path = Path(temp_path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(cleaned)
        handle.write("\n")

    atexit.register(cleanup_runtime_secret_file, path)
    return str(path)


def decode_cookie_env_content() -> str | None:
    raw_b64 = os.getenv("YTDL_COOKIES_B64", "").strip()
    if raw_b64:
        try:
            return base64.b64decode(raw_b64.encode("utf-8"), validate=True).decode("utf-8")
        except Exception:
            logger.warning("YTDL_COOKIES_B64 is set but could not be decoded as UTF-8 base64.")
            return None

    raw = os.getenv("YTDL_COOKIES", "").strip()
    if raw:
        return raw.replace("\\n", "\n")
    return None


def resolve_optional_cookies_file() -> str | None:
    raw_file = os.getenv("YTDL_COOKIES_FILE", "").strip()
    if raw_file:
        candidate = Path(raw_file).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    cookie_content = decode_cookie_env_content()
    if cookie_content:
        return write_runtime_secret_file("cookies", cookie_content)

    return None


def resolve_ffmpeg() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = resolve_ffmpeg()
BROWSER_COOKIE_SOURCES = ("chrome", "edge", "firefox", "brave")

# FIX 2: Cache now stores (expiry, payload, winning_profile) — the winning
# client profile is persisted so /download can prefer it over all others.
FORMAT_CACHE: dict[str, tuple[float, dict, str | None]] = {}

INFLIGHT_FORMAT_REQUESTS: dict[str, "asyncio.Task[dict]"] = {}
FORMAT_CACHE_TTL_SECONDS = 1200


def normalize_po_token(raw: str | None, default_binding: str) -> str | None:
    if not raw:
        return None

    normalized_tokens: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        normalized_tokens.append(token if "+" in token else f"{default_binding}+{token}")

    return ",".join(normalized_tokens) or None


YTDL_PLAYER_CLIENTS = os.getenv("YTDL_PLAYER_CLIENTS", "").strip() or None
YTDL_YOUTUBE_VISITOR_DATA = os.getenv("YTDL_YOUTUBE_VISITOR_DATA", "").strip() or None
YTDL_YOUTUBE_PO_TOKEN_BINDING = os.getenv("YTDL_YOUTUBE_PO_TOKEN_BINDING", "mweb.gvs").strip() or "mweb.gvs"
YTDL_YOUTUBE_PO_TOKEN_RAW = os.getenv("YTDL_YOUTUBE_PO_TOKEN", "").strip() or None
YTDL_YOUTUBE_PO_TOKEN = normalize_po_token(YTDL_YOUTUBE_PO_TOKEN_RAW, YTDL_YOUTUBE_PO_TOKEN_BINDING)


def build_client_profiles() -> tuple[str | None, ...]:
    # Keep a stable order and dedupe while allowing env overrides.
    profiles: list[str | None] = [None]
    if YTDL_PLAYER_CLIENTS:
        profiles.append(YTDL_PLAYER_CLIENTS)
    if YTDL_YOUTUBE_PO_TOKEN:
        # mweb is currently the most practical client when PO tokens are available.
        profiles.append("mweb,web_safari,android_vr,tv")
    profiles.extend(
        [
            "tv,android_vr,web_safari,mweb,web_creator,web",
            "web,android",
            "ios",
            "tv_embedded,web",
        ]
    )

    ordered_unique: list[str | None] = []
    for profile in profiles:
        if profile not in ordered_unique:
            ordered_unique.append(profile)
    return tuple(ordered_unique)


YOUTUBE_CLIENT_PROFILES = build_client_profiles()
YTDL_CONCURRENT_FRAGMENTS = str(get_env_int("YTDL_CONCURRENT_FRAGMENTS", 16, 1))
YTDL_HTTP_CHUNK_SIZE = os.getenv("YTDL_HTTP_CHUNK_SIZE", "10M").strip() or None

YTDL_IMPERSONATE = os.getenv("YTDL_IMPERSONATE", "chrome").strip() or None
YTDL_USER_AGENT = os.getenv("YTDL_USER_AGENT", "").strip() or None
YTDL_SOURCE_ADDRESS = os.getenv("YTDL_SOURCE_ADDRESS", "").strip() or None
YTDL_POT_PROVIDER_URL = os.getenv("YTDL_POT_PROVIDER_URL", "").strip().rstrip("/") or None
YTDL_POT_PROVIDER_SCRIPT_HOME = os.getenv("YTDL_POT_PROVIDER_SCRIPT_HOME", "").strip() or None
YTDL_POT_PROVIDER_SCRIPT_PATH = os.getenv("YTDL_POT_PROVIDER_SCRIPT_PATH", "").strip() or None
YTDL_POT_TOKEN_TTL_HOURS = str(get_env_int("YTDL_POT_TOKEN_TTL_HOURS", get_env_int("TOKEN_TTL", 6, 1), 1))
os.environ.setdefault("TOKEN_TTL", YTDL_POT_TOKEN_TTL_HOURS)
YTDL_FORCE_IPV4 = get_env_bool("YTDL_FORCE_IPV4", True)
YTDL_EXTRACT_SLEEP_SECONDS = get_env_float("YTDL_EXTRACT_SLEEP_SECONDS", 0.2, 0.0)
YTDL_ENABLE_BROWSER_COOKIE_FALLBACK = get_env_bool("YTDL_ENABLE_BROWSER_COOKIE_FALLBACK", False)

YTDL_FORMAT_MAX_PROFILE_ATTEMPTS = get_env_int("YTDL_FORMAT_MAX_PROFILE_ATTEMPTS", 4, 1)
YTDL_DOWNLOAD_MAX_PROFILE_ATTEMPTS = get_env_int("YTDL_DOWNLOAD_MAX_PROFILE_ATTEMPTS", 8, 1)
YTDL_FORMAT_TIMEOUT_SEC = get_env_int("YTDL_FORMAT_TIMEOUT_SEC", 28, 5)
YTDL_DOWNLOAD_TIMEOUT_SEC = get_env_int("YTDL_DOWNLOAD_TIMEOUT_SEC", 300, 60)
YTDL_PLAYLIST_INFO_TIMEOUT_SEC = get_env_int("YTDL_PLAYLIST_INFO_TIMEOUT_SEC", 60, 10)
YTDL_PLAYLIST_DOWNLOAD_TIMEOUT_SEC = get_env_int("YTDL_PLAYLIST_DOWNLOAD_TIMEOUT_SEC", 3600, 300)
YTDL_MIX_PLAYLIST_MAX_ITEMS = get_env_int("YTDL_MIX_PLAYLIST_MAX_ITEMS", 50, 1)

YTDL_MAX_CONCURRENT_PROCESSES = get_env_int("YTDL_MAX_CONCURRENT_PROCESSES", 5, 1)
YTDL_MIN_START_INTERVAL_SECONDS = get_env_float("YTDL_MIN_START_INTERVAL_SECONDS", 0.2, 0.0)
YTDLP_SEMAPHORE = asyncio.Semaphore(YTDL_MAX_CONCURRENT_PROCESSES)
YTDLP_START_LOCK = asyncio.Lock()
LAST_YTDLP_START_MONOTONIC = 0.0

YTDL_PROXY_URL = os.getenv("YTDL_PROXY_URL", "").strip() or None
YTDL_COOKIES_FILE_CONFIGURED = bool(os.getenv("YTDL_COOKIES_FILE", "").strip())
YTDL_COOKIES_ENV_CONFIGURED = bool(os.getenv("YTDL_COOKIES", "").strip() or os.getenv("YTDL_COOKIES_B64", "").strip())
YTDL_COOKIES_CONFIGURED = YTDL_COOKIES_FILE_CONFIGURED or YTDL_COOKIES_ENV_CONFIGURED
YTDL_COOKIES_FILE = resolve_optional_cookies_file()

if YTDL_COOKIES_CONFIGURED and not YTDL_COOKIES_FILE:
    logger.warning("YouTube cookies are configured but no readable cookies file could be prepared.")

if YTDL_YOUTUBE_PO_TOKEN and not YTDL_PLAYER_CLIENTS:
    logger.info("PO token configured with binding %s. Consider setting YTDL_PLAYER_CLIENTS=mweb,web_safari for best reliability.", YTDL_YOUTUBE_PO_TOKEN_BINDING)


def safe_decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace") if data else ""


def content_disposition_filename(filename: str) -> str:
    ascii_fallback = re.sub(r'[^A-Za-z0-9._ -]', '_', filename).strip() or "download"
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quote(filename)}"


def safe_download_basename(name: str, fallback: str) -> str:
    return re.sub(r'[^\w\s-]', '', name)[:80].strip() or fallback


YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}


def normalize_youtube_host(host: str) -> str:
    host = host.lower()
    if host.startswith("www."):
        return host[4:]
    return host


def parse_http_url(url: str):
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    return parsed


def is_supported_youtube_host(host: str) -> bool:
    normalized = normalize_youtube_host(host)
    return normalized in YOUTUBE_HOSTS or normalized == "youtu.be"


def get_youtube_list_id(url: str) -> str | None:
    parsed = parse_http_url(url)
    if not parsed or not is_supported_youtube_host(parsed.netloc):
        return None
    playlist_ids = [item for item in parse_qs(parsed.query).get("list", []) if item.strip()]
    return playlist_ids[0].strip() if playlist_ids else None


def is_generated_youtube_mix(url: str) -> bool:
    list_id = get_youtube_list_id(url)
    if not list_id:
        return False
    # RD* is YouTube Mix/Radio. It is generated and can be very large or unstable.
    return list_id.upper().startswith("RD")


def playlist_item_limit_for_url(url: str) -> int | None:
    if is_generated_youtube_mix(url):
        return YTDL_MIX_PLAYLIST_MAX_ITEMS
    return None


def append_playlist_limit_flags(cmd: list[str], url: str) -> None:
    item_limit = playlist_item_limit_for_url(url)
    if item_limit:
        cmd += ["--playlist-end", str(item_limit)]


def append_network_auth_flags(cmd: list[str]) -> None:
    if YTDL_PROXY_URL:
        cmd += ["--proxy", YTDL_PROXY_URL]
    if YTDL_COOKIES_FILE:
        cmd += ["--cookies", YTDL_COOKIES_FILE]


def append_common_network_flags(cmd: list[str]) -> None:
    if YTDL_IMPERSONATE:
        cmd += ["--impersonate", YTDL_IMPERSONATE]
    if YTDL_FORCE_IPV4:
        cmd += ["--force-ipv4"]
    if YTDL_USER_AGENT:
        cmd += ["--add-headers", f"User-Agent:{YTDL_USER_AGENT}"]
    if YTDL_SOURCE_ADDRESS:
        cmd += ["--source-address", YTDL_SOURCE_ADDRESS]


def append_pot_provider_extractor_args(cmd: list[str]) -> None:
    if YTDL_POT_PROVIDER_URL:
        cmd += ["--extractor-args", f"youtubepot-bgutilhttp:base_url={YTDL_POT_PROVIDER_URL}"]

    if YTDL_POT_PROVIDER_SCRIPT_HOME:
        cmd += ["--extractor-args", f"youtubepot-bgutilscript:server_home={YTDL_POT_PROVIDER_SCRIPT_HOME}"]
    elif YTDL_POT_PROVIDER_SCRIPT_PATH:
        cmd += ["--extractor-args", f"youtubepot-bgutilscript:script_path={YTDL_POT_PROVIDER_SCRIPT_PATH}"]


def build_extractor_args(
    client_profile: str | None,
    *,
    skip_webpage: bool,
    include_auth_args: bool,
) -> str | None:
    parts: list[str] = []
    if client_profile:
        parts.append(f"player_client={client_profile}")
    if skip_webpage:
        parts.append("player_skip=webpage,configs")
    if include_auth_args and YTDL_YOUTUBE_VISITOR_DATA:
        parts.append(f"visitor_data={YTDL_YOUTUBE_VISITOR_DATA}")
    if include_auth_args and YTDL_YOUTUBE_PO_TOKEN:
        parts.append(f"po_token={YTDL_YOUTUBE_PO_TOKEN}")
    if not parts:
        return None
    return "youtube:" + ";".join(parts)


def is_yt_bot_check(err: str) -> bool:
    e = err.lower()
    return any(
        marker in e
        for marker in (
            "too many requests",
            "not a bot",
            "confirm you're not a bot",
            "sign in to confirm",
            "unusual traffic",
            "automated requests",
            "http error 429",
            "error 429",
        )
    )


def is_cookie_source_error(err: str) -> bool:
    e = err.lower()
    if "dpapi" in e or "failed to decrypt" in e:
        return True
    return (
        ("cookies" in e or "cookie" in e)
        and (
            "could not copy" in e
            or "could not find" in e
            or "failed to decrypt" in e
            or "database is locked" in e
            or "permission denied" in e
            or "sqlite" in e
        )
    )


def classify_yt_error(err: str) -> tuple[int, str]:
    e = err.lower()
    if "requested format is not available" in e or "requested format not available" in e:
        return 409, "Selected quality is not currently available. Please refresh formats and choose another quality."
    if "video unavailable" in e or "this video is unavailable" in e or "private video" in e:
        return 404, "This video is unavailable or private."
    if is_yt_bot_check(err):
        return (
            503,
            "The backend could not pass YouTube verification for this video. "
            "The automatic PO-token provider, cookies, or outbound proxy may be missing or unhealthy. Please try again shortly.",
        )
    return 400, "Could not fetch video information. Please check the URL and try again."


async def run_yt_dlp(args: list[str], timeout_sec: int) -> tuple[int, str, str]:
    global LAST_YTDLP_START_MONOTONIC

    async with YTDLP_SEMAPHORE:
        async with YTDLP_START_LOCK:
            now = time.monotonic()
            wait_for = YTDL_MIN_START_INTERVAL_SECONDS - (now - LAST_YTDLP_START_MONOTONIC)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            LAST_YTDLP_START_MONOTONIC = time.monotonic()

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        return proc.returncode, safe_decode(stdout), safe_decode(stderr)


async def run_yt_dlp_with_cookie_fallback(
    base_cmd: list[str],
    timeout_sec: int,
    *,
    context: str,
) -> tuple[str, str]:
    logger.info("yt-dlp attempt [%s] %s", context, shlex.join(base_cmd))
    code, out, err = await run_yt_dlp(base_cmd, timeout_sec)
    
    base_err = (err or out or "").strip()
    
    # If code is 0 (success) and there is no bot check in stderr, we're good.
    # We must check for bot checks even on code 0 because `--ignore-errors`
    # (used for playlists) causes yt-dlp to exit with 0 even if all videos fail.
    if code == 0 and not is_yt_bot_check(base_err):
        return out, err

    base_err = (err or out or "").strip()
    if base_err:
        logger.warning("yt-dlp failed [%s] %s", context, base_err.splitlines()[0][:300])

    # For non bot-check errors, fail fast on the base attempt.
    if not is_yt_bot_check(base_err):
        status, msg = classify_yt_error(base_err)
        raise HTTPException(status_code=status, detail=msg)

    # Base attempt triggered a bot-check; try browser cookie fallbacks.
    last_err = base_err

    # If browser-cookie probing is disabled or an explicit cookies file is configured,
    # avoid expensive local-browser scans and return classified error immediately.
    if (not YTDL_ENABLE_BROWSER_COOKIE_FALLBACK) or YTDL_COOKIES_FILE:
        status, msg = classify_yt_error(last_err)
        raise HTTPException(status_code=status, detail=msg)

    for browser in BROWSER_COOKIE_SOURCES:
        cookie_cmd = base_cmd[:-1] + ["--cookies-from-browser", browser, base_cmd[-1]]
        logger.info("yt-dlp cookie attempt [%s] browser=%s %s", context, browser, shlex.join(cookie_cmd))
        code, out, err = await run_yt_dlp(cookie_cmd, timeout_sec)
        attempt_err = (err or out or "").strip()

        if code == 0 and not is_yt_bot_check(attempt_err):
            logger.info("yt-dlp recovered with cookie fallback [%s] browser=%s", context, browser)
            return out, err

        attempt_err = (err or out or "").strip()
        if attempt_err:
            logger.warning("yt-dlp cookie fail [%s] browser=%s %s", context, browser, attempt_err.splitlines()[0][:300])

        # Cookie extraction errors are local machine issues; keep trying other browsers.
        if is_cookie_source_error(attempt_err):
            continue

        last_err = attempt_err or last_err
        # Continue cookie attempts while YouTube is still returning bot-check responses.
        if is_yt_bot_check(last_err):
            continue

        # Non bot-check and non cookie-source error: stop and classify.
        break

    status, msg = classify_yt_error(last_err)
    raise HTTPException(status_code=status, detail=msg)


def build_download_command(
    req: "DownloadRequest",
    output_template: str,
    *,
    client_profile: str | None,
    skip_webpage: bool,
) -> list[str]:
    cmd = [
        YT_DLP,
        "--no-playlist",
        "--concurrent-fragments",
        YTDL_CONCURRENT_FRAGMENTS,
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--extractor-retries",
        "2",
        "--retry-sleep",
        "extractor:linear=1::1",
        "--retry-sleep",
        "http:exp=1:4",
        # FIX 3: Removed --no-warnings so real yt-dlp error messages are
        # captured in stderr and surfaced through classify_yt_error for
        # accurate error reporting. Warnings are not fatal and rarely relevant.
        "--js-runtimes",
        "node",
        "-o",
        output_template,
    ]

    extractor_args = build_extractor_args(
        client_profile,
        skip_webpage=skip_webpage,
        include_auth_args=True,
    )
    if extractor_args:
        cmd += ["--extractor-args", extractor_args]
    append_pot_provider_extractor_args(cmd)

    if YTDL_EXTRACT_SLEEP_SECONDS > 0:
        cmd += ["--sleep-requests", str(YTDL_EXTRACT_SLEEP_SECONDS)]

    if YTDL_HTTP_CHUNK_SIZE:
        cmd += ["--http-chunk-size", YTDL_HTTP_CHUNK_SIZE]

    if req.audio_format_id:
        # Prefer AAC/m4a-compatible audio first for broader playback support,
        # then fall back to requested/any available audio.
        format_selector = (
            f"{req.format_id}+bestaudio[ext=m4a]"
            f"/{req.format_id}+bestaudio[acodec*=mp4a]"
            f"/{req.format_id}+{req.audio_format_id}"
            f"/{req.format_id}+bestaudio"
            f"/{req.format_id}"
        )
        cmd += [
            "-f",
            format_selector,
            "--merge-output-format",
            "mp4",
        ]
    else:
        cmd += ["-f", req.format_id]

    if FFMPEG:
        cmd += ["--ffmpeg-location", FFMPEG]

    append_common_network_flags(cmd)
    append_network_auth_flags(cmd)

    cmd.append(req.url.strip())
    return cmd


def build_playlist_info_command(url: str, *, client_profile: str | None) -> list[str]:
    cmd = [
        YT_DLP,
        "--dump-single-json",
        "--flat-playlist",
        "--yes-playlist",
        "--js-runtimes",
        "node",
        "--extractor-retries",
        "2",
        "--retry-sleep",
        "extractor:linear=1::1",
    ]

    extractor_args = build_extractor_args(
        client_profile,
        skip_webpage=False,
        include_auth_args=True,
    )
    if extractor_args:
        cmd += ["--extractor-args", extractor_args]
    append_pot_provider_extractor_args(cmd)

    if YTDL_EXTRACT_SLEEP_SECONDS > 0:
        cmd += ["--sleep-requests", str(YTDL_EXTRACT_SLEEP_SECONDS)]

    append_playlist_limit_flags(cmd, url)
    append_common_network_flags(cmd)
    append_network_auth_flags(cmd)
    cmd.append(url)
    return cmd


def build_playlist_format_selector(quality: str, ffmpeg_available: bool) -> tuple[str, bool]:
    normalized = (quality or "best").strip().lower()
    height_by_quality = {
        "1080p": 1080,
        "720p": 720,
        "480p": 480,
        "360p": 360,
    }

    if normalized == "audio":
        return "bestaudio[ext=m4a]/bestaudio", False

    if normalized not in {"best", *height_by_quality.keys()}:
        raise HTTPException(status_code=400, detail="Invalid playlist quality selected")

    if not ffmpeg_available:
        if normalized == "best":
            return "best[ext=mp4]/best", False
        height = height_by_quality[normalized]
        return f"best[height<={height}][ext=mp4]/best[height<={height}]/best", False

    if normalized == "best":
        return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best", True

    height = height_by_quality[normalized]
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={height}]+bestaudio"
        f"/best[height<={height}][ext=mp4]"
        f"/best[height<={height}]",
        True,
    )


def build_playlist_download_command(
    req: "PlaylistDownloadRequest",
    output_template: str,
    *,
    client_profile: str | None,
    skip_webpage: bool,
) -> list[str]:
    cmd = [
        YT_DLP,
        "--yes-playlist",
        "--ignore-errors",
        "--windows-filenames",
        "--concurrent-fragments",
        YTDL_CONCURRENT_FRAGMENTS,
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--extractor-retries",
        "2",
        "--retry-sleep",
        "extractor:linear=1::1",
        "--retry-sleep",
        "http:exp=1:4",
        "--js-runtimes",
        "node",
        "-o",
        output_template,
    ]

    extractor_args = build_extractor_args(
        client_profile,
        skip_webpage=skip_webpage,
        include_auth_args=True,
    )
    if extractor_args:
        cmd += ["--extractor-args", extractor_args]
    append_pot_provider_extractor_args(cmd)

    if YTDL_EXTRACT_SLEEP_SECONDS > 0:
        cmd += ["--sleep-requests", str(YTDL_EXTRACT_SLEEP_SECONDS)]

    if YTDL_HTTP_CHUNK_SIZE:
        cmd += ["--http-chunk-size", YTDL_HTTP_CHUNK_SIZE]

    format_selector, needs_merge = build_playlist_format_selector(req.quality, bool(FFMPEG))
    cmd += ["-f", format_selector]

    if FFMPEG and needs_merge:
        cmd += [
            "--merge-output-format",
            "mp4",
            "--ffmpeg-location",
            FFMPEG,
        ]
    elif FFMPEG:
        cmd += ["--ffmpeg-location", FFMPEG]

    append_playlist_limit_flags(cmd, req.url.strip())
    append_common_network_flags(cmd)
    append_network_auth_flags(cmd)
    cmd.append(req.url.strip())
    return cmd


def format_quality_key(formats: list[dict]) -> tuple[int, int, int]:
    downloadable_video_heights = sorted(
        {
            f.get("height") or 0
            for f in formats
            if f.get("has_video") and f.get("downloadable")
        },
        reverse=True,
    )
    top_downloadable_height = downloadable_video_heights[0] if downloadable_video_heights else 0
    downloadable_count = sum(1 for f in formats if f.get("downloadable"))
    return top_downloadable_height, downloadable_count, len(formats)


async def fetch_video_json(url: str, client_profile: str | None) -> tuple[dict, int]:
    last_error: HTTPException | None = None
    last_stderr = ""

    for skip_webpage in (False, True):
        # `player_skip=webpage,configs` without visitor_data is often noisy and
        # less reliable for default clients; keep it for explicit profiles.
        if skip_webpage and not YTDL_YOUTUBE_VISITOR_DATA and client_profile is None:
            continue

        cmd = [
            YT_DLP,
            "--dump-json",
            "--no-playlist",
            "--js-runtimes", "node",
            "--extractor-retries",
            "2",
            "--retry-sleep",
            "extractor:linear=1::1",
        ]

        extractor_args = build_extractor_args(
            client_profile,
            skip_webpage=skip_webpage,
            include_auth_args=True,
        )
        if extractor_args:
            cmd += ["--extractor-args", extractor_args]
        append_pot_provider_extractor_args(cmd)

        if YTDL_EXTRACT_SLEEP_SECONDS > 0:
            cmd += ["--sleep-requests", str(YTDL_EXTRACT_SLEEP_SECONDS)]

        append_common_network_flags(cmd)
        append_network_auth_flags(cmd)
        cmd.append(url)

        try:
            stdout_text, stderr_text = await run_yt_dlp_with_cookie_fallback(
                cmd,
                timeout_sec=YTDL_FORMAT_TIMEOUT_SEC,
                context=f"formats profile={client_profile or 'default'} skip_webpage={skip_webpage}",
            )
            last_stderr = stderr_text
            try:
                info = json.loads(stdout_text)
            except json.JSONDecodeError:
                raise HTTPException(status_code=502, detail=f"Could not parse yt-dlp output. {stderr_text[:200]}")
            return info, len(info.get("formats", []))
        except HTTPException as ex:
            last_error = ex
            continue

    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=400, detail=f"Failed to fetch video info. {last_stderr[:200]}")


def build_formats_payload(info: dict) -> dict:
    formats = []

    audio_only_formats = [
        f
        for f in info.get("formats", [])
        if f.get("vcodec", "none") == "none" and f.get("acodec", "none") != "none"
    ]

    def _audio_any_score(f: dict) -> tuple[float, float]:
        return ((f.get("abr") or 0), (f.get("tbr") or 0))

    def _audio_compat_score(f: dict) -> tuple[int, int, float, float]:
        acodec = str(f.get("acodec") or "").lower()
        ext = str(f.get("ext") or "").lower()
        aac_like = int("mp4a" in acodec or acodec == "aac")
        mp4_container = int(ext in ("m4a", "mp4"))
        return (aac_like, mp4_container, (f.get("abr") or 0), (f.get("tbr") or 0))

    best_audio_any = max(audio_only_formats, key=_audio_any_score, default=None)
    best_audio_compat = max(audio_only_formats, key=_audio_compat_score, default=None)
    best_audio_id = (
        (best_audio_compat or {}).get("format_id")
        or (best_audio_any or {}).get("format_id")
    )
    best_audio_fallback_id = (best_audio_any or {}).get("format_id")

    for f in info.get("formats", []):
        fid = f.get("format_id", "")
        vcodec = f.get("vcodec", "none")
        acodec = f.get("acodec", "none")
        ext = f.get("ext", "")
        height = f.get("height")
        width = f.get("width")
        filesize = f.get("filesize") or f.get("filesize_approx")
        tbr = f.get("tbr")
        abr = f.get("abr")
        fps = f.get("fps")

        has_video = vcodec != "none" and vcodec is not None
        has_audio = acodec != "none" and acodec is not None
        needs_merge = bool(has_video and not has_audio)
        downloadable = bool(has_audio or (needs_merge and best_audio_id and FFMPEG))
        acodec_lower = str(acodec or "").lower()
        compat_audio = bool("mp4a" in acodec_lower or acodec_lower == "aac")

        # Hide storyboard/mhtml entries from UI as requested.
        if "storyboard" in fid or ext == "mhtml":
            continue

        if has_video:
            label = f"{height}p" if height else "video"
            if fps and fps > 30:
                label += f" {int(fps)}fps"
            if f.get("format_note") and f.get("format_note") not in label:
                label += f" {f.get('format_note')}"
            kind = "video+audio" if has_audio else "video only"
        elif has_audio:
            label = "audio only"
            kind = "audio"
            if abr:
                label += f" {int(abr)}kbps"
        else:
            label = f.get("format_note") or ext or fid or "other"
            kind = "other"

        size_str = None
        if filesize:
            mb = filesize / (1024 * 1024)
            size_str = f"{mb:.1f} MB"
        elif tbr:
            size_str = f"~{int(tbr)}kbps"

        formats.append({
            "format_id": fid,
            "label": label,
            "ext": ext,
            "kind": kind,
            "has_video": has_video,
            "has_audio": has_audio,
            "needs_merge": needs_merge,
            "downloadable": downloadable,
            "compat_audio": compat_audio,
            "height": height,
            "width": width,
            "filesize": filesize,
            "filesize_str": size_str,
            "tbr": tbr,
            "fps": fps,
            "vcodec": vcodec,
            "acodec": acodec,
            "format_note": f.get("format_note"),
            "format_id_display": fid,
        })

    def sort_key(f):
        playable_rank = 1 if f["downloadable"] else 0
        has_video_rank = 1 if f["has_video"] else 0
        has_audio_rank = 1 if f["has_audio"] else 0
        compat_audio_rank = 1 if f.get("compat_audio") else 0
        progressive_rank = 1 if f["has_video"] and f["has_audio"] else 0
        container_rank = 1 if f["ext"] == "mp4" else 0
        height_rank = f["height"] or 0
        fps_rank = f["fps"] or 0
        audio_rank = f.get("tbr") or 0
        return (
            playable_rank,
            has_video_rank,
            progressive_rank,
            has_audio_rank,
            compat_audio_rank,
            container_rank,
            height_rank,
            fps_rank,
            audio_rank,
        )

    formats.sort(key=sort_key, reverse=True)

    # Mark one best format so the frontend can put the most relevant option first/selected.
    recommended_idx = next(
        (
            i
            for i, fmt in enumerate(formats)
            if fmt["downloadable"] and fmt["has_video"] and fmt["has_audio"] and fmt["ext"] == "mp4" and fmt.get("compat_audio")
        ),
        next(
            (
                i
                for i, fmt in enumerate(formats)
                if fmt["downloadable"] and fmt["has_video"] and fmt["has_audio"]
            ),
            next(
                (
                    i
                    for i, fmt in enumerate(formats)
                    if fmt["downloadable"] and fmt["has_video"]
                ),
                0,
            ),
        ),
    )
    if formats:
        formats[recommended_idx]["recommended"] = True

    return {
        "title": info.get("title", "Unknown"),
        "thumbnail": info.get("thumbnail"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "formats": formats,
        "best_audio_id": best_audio_id,
        "best_audio_fallback_id": best_audio_fallback_id,
        "ffmpeg_available": bool(FFMPEG),
    }


def is_valid_youtube_url(url: str) -> bool:
    parsed = parse_http_url(url)
    if not parsed or not is_supported_youtube_host(parsed.netloc):
        return False

    host = normalize_youtube_host(parsed.netloc)
    path = parsed.path.strip("/")
    query = parse_qs(parsed.query)

    if host == "youtu.be":
        return bool(path)

    if query.get("v"):
        return True

    # Let yt-dlp handle common direct video URL shapes beyond /watch.
    return bool(
        path.startswith("shorts/")
        or path.startswith("live/")
        or path.startswith("embed/")
        or path.startswith("v/")
        or path.startswith("clip/")
    )


def is_valid_youtube_playlist_url(url: str) -> bool:
    return get_youtube_list_id(url) is not None


class URLRequest(BaseModel):
    url: str


class DownloadRequest(BaseModel):
    url: str
    format_id: str
    audio_format_id: str | None = None
    title: str = "video"


class PlaylistDownloadRequest(BaseModel):
    url: str
    title: str = "playlist"
    quality: str = "best"


@app.post("/formats")
async def get_formats(req: URLRequest):
    if not is_valid_youtube_url(req.url.strip()):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    cache_key = req.url.strip()
    now = asyncio.get_running_loop().time()
    cached = FORMAT_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    inflight_task = INFLIGHT_FORMAT_REQUESTS.get(cache_key)
    if inflight_task:
        return await inflight_task

    async def fetch_best_payload() -> dict[str, Any]:
        best_payload = None
        best_score = (-1, -1, -1)
        best_profile: str | None = None  # FIX 2: track which profile won
        last_error = None

        for idx, client_profile in enumerate(YOUTUBE_CLIENT_PROFILES):
            if idx >= YTDL_FORMAT_MAX_PROFILE_ATTEMPTS:
                break
            try:
                info, raw_count = await fetch_video_json(req.url.strip(), client_profile)
                payload = build_formats_payload(info)
                score = format_quality_key(payload["formats"])

                if score > best_score:
                    best_score = score
                    best_payload = payload
                    best_profile = client_profile  # FIX 2: remember the winner

                # Fast path: first successful payload is usually enough for UI selection.
                if payload["formats"] and score[1] >= 1:
                    break
            except HTTPException as ex:
                last_error = ex
                continue

        if best_payload is not None:
            # FIX 2: cache includes winning profile so /download can use it
            FORMAT_CACHE[cache_key] = (
                asyncio.get_running_loop().time() + FORMAT_CACHE_TTL_SECONDS,
                best_payload,
                best_profile,
            )
            return best_payload

        if last_error is not None:
            raise last_error

        raise HTTPException(status_code=500, detail="Failed to fetch video info")

    try:
        task = asyncio.create_task(fetch_best_payload())
        INFLIGHT_FORMAT_REQUESTS[cache_key] = task
        return await task

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out fetching video info")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected backend error")
        raise HTTPException(status_code=500, detail="Something went wrong while processing the request. Please try again later.")
    finally:
        INFLIGHT_FORMAT_REQUESTS.pop(cache_key, None)


async def fetch_playlist_json(url: str, client_profile: str | None) -> dict:
    cmd = build_playlist_info_command(url, client_profile=client_profile)
    stdout_text, stderr_text = await run_yt_dlp_with_cookie_fallback(
        cmd,
        timeout_sec=YTDL_PLAYLIST_INFO_TIMEOUT_SEC,
        context=f"playlist-info profile={client_profile or 'default'}",
    )
    try:
        return json.loads(stdout_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Could not parse playlist info. {stderr_text[:200]}")


def build_playlist_payload(info: dict, source_url: str) -> dict:
    entries = [entry for entry in info.get("entries", []) if entry]
    videos = []
    first_thumbnail = None
    item_limit = playlist_item_limit_for_url(source_url)
    is_limited = bool(item_limit and len(entries) >= item_limit)

    for entry in entries:
        thumbnails = entry.get("thumbnails") or []
        thumbnail = entry.get("thumbnail")
        if not thumbnail and thumbnails:
            thumbnail = thumbnails[-1].get("url")
        if thumbnail and not first_thumbnail:
            first_thumbnail = thumbnail
        videos.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title") or "Untitled video",
                "duration": entry.get("duration"),
                "uploader": entry.get("uploader"),
                "thumbnail": thumbnail,
                "url": entry.get("url"),
            }
        )

    return {
        "title": info.get("title") or "YouTube playlist",
        "uploader": info.get("uploader") or info.get("channel"),
        "playlist_count": info.get("playlist_count") or len(videos),
        "download_count": len(videos),
        "download_limited": is_limited,
        "playlist_limit": item_limit,
        "playlist_kind": "mix" if is_generated_youtube_mix(source_url) else "playlist",
        "thumbnail": info.get("thumbnail") or first_thumbnail,
        "videos": videos,
    }


@app.post("/playlist-info")
async def get_playlist_info(req: URLRequest):
    url = req.url.strip()
    if not is_valid_youtube_playlist_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube playlist URL")

    last_error: HTTPException | None = None
    for idx, client_profile in enumerate(YOUTUBE_CLIENT_PROFILES):
        if idx >= YTDL_FORMAT_MAX_PROFILE_ATTEMPTS:
            break
        try:
            info = await fetch_playlist_json(url, client_profile)
            return build_playlist_payload(info, url)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out fetching playlist info")
        except HTTPException as ex:
            last_error = ex
            continue

    if last_error is not None:
        raise last_error
    raise HTTPException(status_code=500, detail="Failed to fetch playlist info")


# Build ordered client_profile attempts for /download so it mirrors every
# profile that /formats can use, with the cached winning profile tried first.
def _build_download_attempt_specs(
    preferred_profile: str | None,
) -> list[tuple[str | None, bool]]:
    """
    Return ordered (client_profile, skip_webpage) attempts for /download.

    Strategy:
     1. Try the preferred_profile (the one that won at /formats time) first.
     2. Then fall through ALL remaining YOUTUBE_CLIENT_PROFILES in order,
         so we never miss the profile that
       produced the format IDs listed in the UI.
    """
    profiles_in_order: list[str | None] = []

    # Preferred profile goes first (may be None)
    profiles_in_order.append(preferred_profile)

    # Append remaining profiles from the canonical list, preserving order
    for p in YOUTUBE_CLIENT_PROFILES:
        if p not in profiles_in_order:
            profiles_in_order.append(p)

    specs: list[tuple[str | None, bool]] = []
    for profile in profiles_in_order:
        specs.append((profile, False))
        if YTDL_YOUTUBE_VISITOR_DATA or profile is not None:
            specs.append((profile, True))

    return specs[:YTDL_DOWNLOAD_MAX_PROFILE_ATTEMPTS]


async def stream_video_download(req: DownloadRequest):
    if not is_valid_youtube_url(req.url.strip()):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    try:
        safe_title = re.sub(r'[^\w\s-]', '', req.title)[:60].strip() or "video"
        output_template = str(session_dir / f"{safe_title}.%(ext)s")

        if req.audio_format_id and not FFMPEG:
            raise HTTPException(
                status_code=503,
                detail="FFmpeg is required to download HD video-only formats. Install ffmpeg or use a video+audio format.",
            )

        # FIX 1+2: look up the winning profile from /formats cache and use it
        # as the preferred starting point, then fall back through all profiles.
        cache_key = req.url.strip()
        now = asyncio.get_running_loop().time()
        cached = FORMAT_CACHE.get(cache_key)
        preferred_profile: str | None = None
        if cached and cached[0] > now:
            # cached[2] is the winning profile stored by /formats (FIX 2)
            preferred_profile = cached[2]

        attempt_specs = _build_download_attempt_specs(
            preferred_profile,
        )

        download_errors: list[HTTPException] = []
        download_succeeded = False

        for client_profile, skip_webpage in attempt_specs:
            cmd = build_download_command(
                req,
                output_template,
                client_profile=client_profile,
                skip_webpage=skip_webpage,
            )
            try:
                await run_yt_dlp_with_cookie_fallback(
                    cmd,
                    timeout_sec=YTDL_DOWNLOAD_TIMEOUT_SEC,
                    context=(
                        f"download profile={client_profile or 'default'} "
                        f"skip_webpage={skip_webpage} fmt={req.format_id} aud={req.audio_format_id or '-'}"
                    ),
                )
                download_succeeded = True
                break
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Download timed out")
            except HTTPException as ex:
                download_errors.append(ex)
                if ex.status_code in (401, 403):
                    break

        if not download_succeeded:
            verification_error = next((e for e in download_errors if e.status_code == 503), None)
            format_unavailable_error = next((e for e in download_errors if e.status_code == 409), None)
            if verification_error is not None:
                raise verification_error
            if format_unavailable_error is not None:
                raise HTTPException(
                    status_code=409,
                    detail="Selected quality stream is temporarily unavailable. Refetch formats and try again.",
                )
            if download_errors:
                raise download_errors[-1]
            raise HTTPException(status_code=500, detail="Download failed due to unknown error")

        # Find the output file
        files = list(session_dir.iterdir())
        if not files:
            raise HTTPException(status_code=500, detail="No output file found")

        output_file = files[0]
        ext = output_file.suffix.lstrip(".")
        mime_map = {
            "mp4": "video/mp4",
            "webm": "video/webm",
            "mkv": "video/x-matroska",
            "m4a": "audio/mp4",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
        }
        media_type = mime_map.get(ext, "application/octet-stream")
        dl_filename = f"{safe_download_basename(req.title, 'video')}.{ext}"

        async def file_generator():
            try:
                with open(output_file, "rb") as f:
                    while chunk := f.read(1024 * 256):  # 256KB chunks
                        yield chunk
            finally:
                shutil.rmtree(session_dir, ignore_errors=True)

        return StreamingResponse(
            file_generator(),
            media_type=media_type,
            headers={
                "Content-Disposition": content_disposition_filename(dl_filename),
                "Content-Length": str(output_file.stat().st_size),
            },
        )

    except HTTPException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.exception("Unexpected backend error")
        raise HTTPException(status_code=500, detail="Something went wrong while processing the request. Please try again later.")


@app.post("/download")
async def download_video(req: DownloadRequest):
    return await stream_video_download(req)


@app.get("/download")
async def download_video_get(
    url: str,
    format_id: str,
    audio_format_id: str | None = None,
    title: str = "video",
):
    return await stream_video_download(
        DownloadRequest(
            url=url,
            format_id=format_id,
            audio_format_id=audio_format_id,
            title=title,
        )
    )


async def stream_playlist_download(req: PlaylistDownloadRequest):
    if not is_valid_youtube_playlist_url(req.url.strip()):
        raise HTTPException(status_code=400, detail="Invalid YouTube playlist URL")

    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    downloads_dir = session_dir / "playlist"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    try:
        safe_title = safe_download_basename(req.title, "playlist")
        safe_quality = safe_download_basename(req.quality, "best").lower()
        output_template = str(downloads_dir / "%(playlist_autonumber)03d - %(title).80s.%(ext)s")

        download_errors: list[HTTPException] = []
        download_succeeded = False

        for client_profile, skip_webpage in _build_download_attempt_specs(None):
            cmd = build_playlist_download_command(
                req,
                output_template,
                client_profile=client_profile,
                skip_webpage=skip_webpage,
            )
            try:
                await run_yt_dlp_with_cookie_fallback(
                    cmd,
                    timeout_sec=YTDL_PLAYLIST_DOWNLOAD_TIMEOUT_SEC,
                    context=(
                        f"playlist-download profile={client_profile or 'default'} "
                        f"skip_webpage={skip_webpage} quality={req.quality}"
                    ),
                )
                download_succeeded = True
                break
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Playlist download timed out")
            except HTTPException as ex:
                download_errors.append(ex)
                if ex.status_code in (401, 403):
                    break
        if not download_succeeded:
            verification_error = next((e for e in download_errors if e.status_code == 503), None)
            if verification_error is not None:
                raise verification_error
            if download_errors:
                raise download_errors[-1]
            raise HTTPException(status_code=500, detail="Playlist download failed due to unknown error")

        downloaded_files = [path for path in downloads_dir.rglob("*") if path.is_file()]
        if not downloaded_files:
            raise HTTPException(status_code=500, detail="No playlist videos were downloaded")

        zip_path = session_dir / f"{safe_title}-{safe_quality}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in downloaded_files:
                archive.write(path, arcname=path.relative_to(downloads_dir))

        async def file_generator():
            try:
                with open(zip_path, "rb") as f:
                    while chunk := f.read(1024 * 256):
                        yield chunk
            finally:
                shutil.rmtree(session_dir, ignore_errors=True)

        return StreamingResponse(
            file_generator(),
            media_type="application/zip",
            headers={
                "Content-Disposition": content_disposition_filename(f"{safe_title}-{safe_quality}.zip"),
                "Content-Length": str(zip_path.stat().st_size),
            },
        )

    except HTTPException:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.exception("Unexpected backend error")
        raise HTTPException(status_code=500, detail="Something went wrong while processing the request. Please try again later.")


@app.post("/download-playlist")
async def download_playlist(req: PlaylistDownloadRequest):
    return await stream_playlist_download(req)


@app.get("/download-playlist")
async def download_playlist_get(
    url: str,
    title: str = "playlist",
    quality: str = "best",
):
    return await stream_playlist_download(
        PlaylistDownloadRequest(url=url, title=title, quality=quality)
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "features": {
            "playlist_info": True,
            "playlist_download": True,
            "download_get": True,
            "playlist_quality": True,
        },
        "yt_dlp": bool(shutil.which("yt-dlp")),
        "ffmpeg": bool(FFMPEG),
        "concurrent_fragments": YTDL_CONCURRENT_FRAGMENTS,
        "http_chunk_size": YTDL_HTTP_CHUNK_SIZE,
        "max_concurrent_processes": YTDL_MAX_CONCURRENT_PROCESSES,
        "min_start_interval_seconds": YTDL_MIN_START_INTERVAL_SECONDS,
        "extract_sleep_seconds": YTDL_EXTRACT_SLEEP_SECONDS,
        "browser_cookie_fallback_enabled": YTDL_ENABLE_BROWSER_COOKIE_FALLBACK,
        "format_max_profile_attempts": YTDL_FORMAT_MAX_PROFILE_ATTEMPTS,
        "download_max_profile_attempts": YTDL_DOWNLOAD_MAX_PROFILE_ATTEMPTS,
        "format_timeout_sec": YTDL_FORMAT_TIMEOUT_SEC,
        "download_timeout_sec": YTDL_DOWNLOAD_TIMEOUT_SEC,
        "playlist_info_timeout_sec": YTDL_PLAYLIST_INFO_TIMEOUT_SEC,
        "playlist_download_timeout_sec": YTDL_PLAYLIST_DOWNLOAD_TIMEOUT_SEC,
        "mix_playlist_max_items": YTDL_MIX_PLAYLIST_MAX_ITEMS,
        "proxy_configured": bool(YTDL_PROXY_URL),
        "impersonate_configured": bool(YTDL_IMPERSONATE),
        "user_agent_configured": bool(YTDL_USER_AGENT),
        "source_address_configured": bool(YTDL_SOURCE_ADDRESS),
        "pot_provider_url_configured": bool(YTDL_POT_PROVIDER_URL),
        "pot_provider_script_configured": bool(YTDL_POT_PROVIDER_SCRIPT_HOME or YTDL_POT_PROVIDER_SCRIPT_PATH),
        "pot_token_ttl_hours": YTDL_POT_TOKEN_TTL_HOURS,
        "force_ipv4": YTDL_FORCE_IPV4,
        "player_clients_override": YTDL_PLAYER_CLIENTS,
        "visitor_data_configured": bool(YTDL_YOUTUBE_VISITOR_DATA),
        "po_token_configured": bool(YTDL_YOUTUBE_PO_TOKEN),
        "po_token_binding": YTDL_YOUTUBE_PO_TOKEN_BINDING if YTDL_YOUTUBE_PO_TOKEN else None,
        "cookies_file_configured": YTDL_COOKIES_FILE_CONFIGURED,
        "cookies_env_configured": YTDL_COOKIES_ENV_CONFIGURED,
        "cookies_configured": YTDL_COOKIES_CONFIGURED,
        "cookies_file_available": bool(YTDL_COOKIES_FILE),
    }
