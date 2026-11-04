"""Lightweight smoke tests — no GPU, no model load required.

Validates module structure, helpers, API routing, and request parsing
so CI can catch regressions without any CUDA hardware.

The voice registry is loaded at session start from the committed preset
bundle.  No network and no qwen_tts are needed — missing qwen_tts is
tolerated and produces voice records with ``prompt_item=None`` (sufficient
for membership and routing tests).
"""
from __future__ import annotations

import asyncio
import importlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_server():
    """Import server module without triggering model loading.

    lifespan/startup only runs when the ASGI app receives its first lifespan
    event, which TestClient / direct import do not trigger.
    """
    return importlib.import_module("server")


@pytest.fixture(scope="session", autouse=True)
def _load_registry():
    """Populate the registry from the committed preset bundle once per run.

    Runs for every test transparently (autouse) so voice-resolution tests
    don't each need to remember to load it.  No model is bound — preset
    records still carry valid IDs, their ``prompt_item`` is ``None`` when
    qwen_tts is unavailable (fine for smoke tests).
    """
    server = _import_server()
    server._voice_registry.load()
    yield


# -----------------------------------------------------------------------
# Module-level smoke
# -----------------------------------------------------------------------

def test_module_imports():
    server = _import_server()
    assert hasattr(server, "app")
    assert hasattr(server, "MODEL_ID")
    assert server.MODEL_ID.startswith("Qwen/Qwen3-TTS")


def test_default_model_is_base():
    """Voice cloning path requires -Base, not -CustomVoice."""
    server = _import_server()
    # Accept either the exact default or a user-set env override that still
    # points at a -Base-compatible checkpoint.
    assert "CustomVoice" not in server.MODEL_ID, (
        "Default model must not be -CustomVoice — voice cloning requires -Base."
    )


# -----------------------------------------------------------------------
# Sentence splitter
# -----------------------------------------------------------------------

def test_split_sentences_basic():
    server = _import_server()
    out = server._split_sentences("Hello world. This is a test! Right?")
    assert out == ["Hello world.", "This is a test!", "Right?"]


def test_split_sentences_strips_whitespace():
    server = _import_server()
    out = server._split_sentences("   One.    Two.    ")
    assert out == ["One.", "Two."]


def test_split_sentences_empty():
    server = _import_server()
    assert server._split_sentences("") == []
    assert server._split_sentences("   ") == []


def test_split_sentences_no_terminator():
    server = _import_server()
    # Single chunk with no sentence terminator should still pass through.
    assert server._split_sentences("hello world") == ["hello world"]


def test_split_speech_chunks_bounds_long_sentences(monkeypatch):
    server = _import_server()
    monkeypatch.setattr(server, "CHUNK_STRATEGY", "characters")
    chunks = server._split_speech_chunks(
        "First sentence. " + ("longword " * 120) + ". Final sentence!"
    )

    assert len(chunks) > 3
    assert all(0 < len(chunk) <= server.MAX_SPEECH_CHUNK_CHARS for chunk in chunks)
    assert chunks[0] == "First sentence."
    assert chunks[-1].endswith("Final sentence!")


def test_split_speech_chunks_pack_adjacent_sentences():
    server = _import_server()
    chunks = server._split_speech_chunks("First sentence. Second sentence. Third sentence.")

    assert chunks == ["First sentence. Second sentence. Third sentence."]


def test_chunk_settings_default_and_validation(monkeypatch):
    server = _import_server()
    enable_name = "QWEN3_TTS_ENABLE_CHUNKED_SPEECH"
    seconds_name = "QWEN3_TTS_MAX_CHUNK_SECONDS"
    chars_name = "QWEN3_TTS_MAX_CHUNK_CHARS"
    monkeypatch.delenv(enable_name, raising=False)
    monkeypatch.delenv(seconds_name, raising=False)
    monkeypatch.delenv(chars_name, raising=False)

    assert server._bool_env(enable_name, True) is True
    assert server._positive_int_env(seconds_name, 60) == 60
    assert server._positive_int_env(chars_name, 500) == 500

    monkeypatch.setenv(enable_name, "off")
    monkeypatch.setenv(seconds_name, "23")
    monkeypatch.setenv(chars_name, "250")
    assert server._bool_env(enable_name, True) is False
    assert server._positive_int_env(seconds_name, 60) == 23
    assert server._positive_int_env(chars_name, 500) == 250

    for invalid in ("", "maybe"):
        monkeypatch.setenv(enable_name, invalid)
        with pytest.raises(RuntimeError, match=enable_name):
            server._bool_env(enable_name, True)
    for name in (seconds_name, chars_name):
        for invalid in ("0", "-2", "1.5", "invalid"):
            monkeypatch.setenv(name, invalid)
            with pytest.raises(RuntimeError, match=name):
                server._positive_int_env(name, 60)


def test_optional_cuda_memory_fraction_validation(monkeypatch):
    server = _import_server()
    name = "QWEN3_TTS_CUDA_MEMORY_FRACTION"
    monkeypatch.delenv(name, raising=False)
    assert server._optional_cuda_memory_fraction_env(name) is None

    for value in ("0.25", "1"):
        monkeypatch.setenv(name, value)
        assert server._optional_cuda_memory_fraction_env(name) == float(value)

    for value in ("0", "-0.1", "1.01", "invalid"):
        monkeypatch.setenv(name, value)
        with pytest.raises(RuntimeError, match=name):
            server._optional_cuda_memory_fraction_env(name)


def test_speech_chunk_size_is_configurable(monkeypatch):
    server = _import_server()
    monkeypatch.setattr(server, "CHUNK_STRATEGY", "characters")
    monkeypatch.setattr(server, "MAX_SPEECH_CHUNK_CHARS", 250)
    chunks = server._split_speech_chunks("word " * 200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 250 for chunk in chunks)


def test_openai_voice_alias_overrides_are_case_insensitive():
    server = _import_server()
    assert server._parse_voice_alias_overrides('{"ALLOY":"custom_voice"}') == {
        "alloy": "custom_voice"
    }


def test_qwen_pronunciation_overrides_are_whole_word(monkeypatch):
    import re

    server = _import_server()
    monkeypatch.setattr(server, "_QWEN_PRONUNCIATIONS", {"zorbo": "ZOR-bo"})
    monkeypatch.setattr(
        server,
        "_QWEN_PRONUNCIATION_PATTERN",
        re.compile(r"(?<!\w)(?:zorbo)(?!\w)", re.IGNORECASE),
    )

    assert server._apply_qwen_pronunciations("Zorbo met Zorbology.") == "ZOR-bo met Zorbology."


def test_preset_embedding_dimensions_match_model_families():
    from voice_registry import _expected_voice_embedding_dim

    assert _expected_voice_embedding_dim("qwen3-tts-0.6b-base") == 1024
    assert _expected_voice_embedding_dim("Qwen3-TTS-12Hz-1.7B-Base") == 2048
    assert _expected_voice_embedding_dim("unknown") is None


def test_generate_audio_caps_duration_and_generator_tokens(monkeypatch):
    import numpy as np

    server = _import_server()
    captured = {}
    record = SimpleNamespace(ref_audio_path=None, prompt_item=object())

    class _FakeModel:
        def generate_voice_clone(self, **kwargs):
            captured.update(kwargs)
            return [np.zeros(61 * 24000, dtype=np.float32)], 24000

    monkeypatch.setattr(server, "_voice_registry", SimpleNamespace(get=lambda _voice: record))
    monkeypatch.setattr(server, "_model", _FakeModel(), raising=False)

    wav, sample_rate = server._generate_audio_sync(
        "A capped segment.",
        "ryan",
        None,
        "English",
        server.MAX_SPEECH_CHUNK_TOKENS,
        server.MAX_SPEECH_CHUNK_SECONDS,
    )

    assert sample_rate == 24000
    assert len(wav) == 60 * sample_rate
    assert captured["max_new_tokens"] == server.MAX_SPEECH_CHUNK_TOKENS


# -----------------------------------------------------------------------
# Voice alias mapping + registry resolution
# -----------------------------------------------------------------------

def test_resolve_voice_openai_aliases():
    server = _import_server()
    assert server._resolve_voice("alloy") == "ryan"
    assert server._resolve_voice("echo") == "aiden"
    assert server._resolve_voice("fable") == "dylan"
    assert server._resolve_voice("onyx") == "eric"
    assert server._resolve_voice("nova") == "serena"
    assert server._resolve_voice("shimmer") == "vivian"


def test_resolve_voice_passthrough():
    server = _import_server()
    assert server._resolve_voice("ryan") == "ryan"
    assert server._resolve_voice("sohee") == "sohee"
    assert server._resolve_voice("uncle_fu") == "uncle_fu"


def test_resolve_voice_unknown_raises_400():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._resolve_voice("not_a_real_voice")
    assert exc_info.value.status_code == 400


def test_resolve_voice_case_insensitive():
    server = _import_server()
    # Uppercase alias should still resolve.
    assert server._resolve_voice("ALLOY") == "ryan"


def test_registry_has_nine_presets():
    server = _import_server()
    presets = server._voice_registry.list(kind="preset")
    ids = {p["id"] for p in presets}
    assert ids == {
        "aiden", "dylan", "eric", "ono_anna", "ryan",
        "serena", "sohee", "uncle_fu", "vivian",
    }


# -----------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------

def test_validate_text_empty_raises():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._validate_text("")
    assert exc_info.value.status_code == 400


def test_validate_text_none_raises():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._validate_text(None)
    assert exc_info.value.status_code == 400


def test_validate_text_too_long_raises():
    from fastapi import HTTPException
    server = _import_server()
    with pytest.raises(HTTPException) as exc_info:
        server._validate_text("x" * (server.MAX_TEXT_LENGTH + 1))
    assert exc_info.value.status_code == 413


def test_validate_text_ok():
    server = _import_server()
    assert server._validate_text("Hello") == "Hello"
    assert server._validate_text("  Hello  ") == "Hello"  # strips whitespace


# -----------------------------------------------------------------------
# Route registration
# -----------------------------------------------------------------------

def test_routes_registered():
    server = _import_server()
    paths = {route.path for route in server.app.routes}
    expected = {
        "/v1/audio/speech",
        "/v1/audio/speech/stream",
        "/v1/audio/speech/pcm-stream",
        "/v1/models",
        "/v1/models/{model_id}",
        "/v1/voices",
        "/v1/voices/{voice_id}",
        "/health",
        "/voices",  # legacy alias
    }
    missing = expected - paths
    assert not missing, f"Missing routes: {missing}"


def test_standard_speech_synthesizes_all_bounded_chunks(monkeypatch):
    from fastapi.testclient import TestClient
    import numpy as np

    server = _import_server()
    monkeypatch.setattr(server, "CHUNK_STRATEGY", "characters")
    calls = []

    def fake_generate(
        text, voice_id, instruct, language, max_new_tokens=None, max_audio_seconds=None
    ):
        calls.append((text, max_new_tokens, max_audio_seconds))
        return np.array([len(calls)], dtype=np.float32), 24000

    monkeypatch.setattr(server, "_generate_audio_sync", fake_generate)
    monkeypatch.setattr(server, "_wav_to_bytes_sync", lambda wav, _sr, _fmt: wav.tobytes())
    monkeypatch.setattr(server, "_model_ready", True)
    monkeypatch.setattr(server, "_API_KEY", "")
    monkeypatch.setattr(server, "_INFER_SEM", asyncio.Semaphore(1), raising=False)
    monkeypatch.setattr(server, "ENABLE_CHUNKED_SPEECH", True)

    response = TestClient(server.app).post(
        "/v1/audio/speech",
        json={
            "input": "First sentence. " + ("longword " * 120) + ". Final sentence!",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert len(calls) > 3
    assert all(len(text) <= server.MAX_SPEECH_CHUNK_CHARS for text, _, _ in calls)
    assert all(tokens == server.MAX_SPEECH_CHUNK_TOKENS for _, tokens, _ in calls)
    assert all(seconds == server.MAX_SPEECH_CHUNK_SECONDS for _, _, seconds in calls)
    assert np.frombuffer(response.content, dtype=np.float32).tolist() == list(
        range(1, len(calls) + 1)
    )


def test_standard_speech_can_restore_single_shot_behavior(monkeypatch):
    from fastapi.testclient import TestClient
    import numpy as np

    server = _import_server()
    calls = []

    def fake_generate(
        text, voice_id, instruct, language, max_new_tokens=None, max_audio_seconds=None
    ):
        calls.append((text, max_new_tokens, max_audio_seconds))
        return np.array([1], dtype=np.float32), 24000

    monkeypatch.setattr(server, "_generate_audio_sync", fake_generate)
    monkeypatch.setattr(server, "_wav_to_bytes_sync", lambda wav, _sr, _fmt: wav.tobytes())
    monkeypatch.setattr(server, "_model_ready", True)
    monkeypatch.setattr(server, "_API_KEY", "")
    monkeypatch.setattr(server, "_INFER_SEM", asyncio.Semaphore(1), raising=False)
    monkeypatch.setattr(server, "ENABLE_CHUNKED_SPEECH", False)
    text = "First sentence. " + ("longword " * 120) + ". Final sentence!"

    response = TestClient(server.app).post(
        "/v1/audio/speech",
        json={"input": text, "response_format": "wav"},
    )

    assert response.status_code == 200
    assert calls == [(text, None, None)]


def test_sentence_stream_caps_each_chunk(monkeypatch):
    from fastapi.testclient import TestClient
    import numpy as np

    server = _import_server()
    monkeypatch.setattr(server, "CHUNK_STRATEGY", "characters")
    calls = []

    def fake_generate(
        text, voice_id, instruct, language, max_new_tokens=None, max_audio_seconds=None
    ):
        calls.append((text, max_new_tokens, max_audio_seconds))
        return np.array([0], dtype=np.float32), 24000

    monkeypatch.setattr(server, "_generate_audio_sync", fake_generate)
    monkeypatch.setattr(server, "_wav_to_bytes_sync", lambda *_args: b"audio")
    monkeypatch.setattr(server, "_model_ready", True)
    monkeypatch.setattr(server, "_API_KEY", "")
    monkeypatch.setattr(server, "_INFER_SEM", asyncio.Semaphore(1), raising=False)
    monkeypatch.setattr(server, "ENABLE_CHUNKED_SPEECH", True)
    text = "First sentence. " + ("longword " * 120) + ". Final sentence!"

    response = TestClient(server.app).post(
        "/v1/audio/speech/stream",
        json={"input": text, "response_format": "wav"},
    )

    assert response.status_code == 200
    assert response.headers["x-tts-sentences"] == str(len(server._split_sentences(text)))
    assert response.headers["x-tts-chunks"] == str(len(calls))
    assert all(tokens == server.MAX_SPEECH_CHUNK_TOKENS for _, tokens, _ in calls)
    assert all(seconds == server.MAX_SPEECH_CHUNK_SECONDS for _, _, seconds in calls)
    assert len(response.content) == len(calls) * (4 + len(b"audio"))


def test_sentence_stream_can_restore_original_chunking(monkeypatch):
    from fastapi.testclient import TestClient
    import numpy as np

    server = _import_server()
    calls = []

    def fake_generate(
        text, voice_id, instruct, language, max_new_tokens=None, max_audio_seconds=None
    ):
        calls.append((text, max_new_tokens, max_audio_seconds))
        return np.array([0], dtype=np.float32), 24000

    monkeypatch.setattr(server, "_generate_audio_sync", fake_generate)
    monkeypatch.setattr(server, "_wav_to_bytes_sync", lambda *_args: b"audio")
    monkeypatch.setattr(server, "_model_ready", True)
    monkeypatch.setattr(server, "_API_KEY", "")
    monkeypatch.setattr(server, "_INFER_SEM", asyncio.Semaphore(1), raising=False)
    monkeypatch.setattr(server, "ENABLE_CHUNKED_SPEECH", False)

    response = TestClient(server.app).post(
        "/v1/audio/speech/stream",
        json={"input": "First. Second.", "response_format": "wav"},
    )

    assert response.status_code == 200
    assert [text for text, _, _ in calls] == ["First.", "Second."]
    assert all(tokens is None and seconds is None for _, tokens, seconds in calls)
    assert "x-tts-chunks" not in response.headers


# -----------------------------------------------------------------------
# Health endpoint (no model loaded)
# -----------------------------------------------------------------------

def test_health_payload_when_loading():
    server = _import_server()
    payload = asyncio.run(server.health())
    assert "model_ready" in payload
    assert "model_id" in payload
    assert "backend" in payload
    assert "device" in payload
    assert payload["model_ready"] is False
    assert payload["status"] == "loading"
    # Voice counts should surface even before the model is loaded.
    assert payload["voices"]["preset"] == 9
    assert payload["voices"]["custom"] >= 0


# -----------------------------------------------------------------------
# Voice listing endpoint
# -----------------------------------------------------------------------

def test_list_voices_returns_all_presets():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.get("/v1/voices")
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        ids = {v["id"] for v in body["data"]}
        assert {"aiden", "ryan", "vivian"}.issubset(ids)
        kinds = {v["kind"] for v in body["data"]}
        assert "preset" in kinds
        assert "openai_voice_aliases" in body
    finally:
        server._API_KEY = original


def test_legacy_voices_endpoint_still_works():
    """Back-compat: GET /voices must still return a data-shaped payload."""
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.get("/voices")
        assert r.status_code == 200
        assert "data" in r.json()
    finally:
        server._API_KEY = original


def test_delete_preset_is_forbidden():
    """Presets must not be deletable via the public API."""
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.delete("/v1/voices/ryan")
        assert r.status_code == 403
    finally:
        server._API_KEY = original


def test_delete_unknown_voice_returns_404():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.delete("/v1/voices/vc_doesnotexist")
        assert r.status_code == 404
    finally:
        server._API_KEY = original


# -----------------------------------------------------------------------
# Voice registration (custom clones) — model is mocked
# -----------------------------------------------------------------------

def test_register_voice_round_trip(tmp_path):
    """POST /v1/voices with a fake audio file registers a custom voice.

    The actual model call is mocked (we inject a fake model onto the
    registry) so no GPU or qwen_tts install is required.  We verify:
      - new voice appears in the registry with a vc_* ID
      - sidecar files land in the configured custom dir
      - GET /v1/voices lists it
      - DELETE removes it and the files
    """
    from fastapi.testclient import TestClient
    import torch

    server = _import_server()
    reg = server._voice_registry

    # Redirect the registry to a temp dir so we don't touch /data/voices.
    original_dir = reg.custom_dir
    reg.custom_dir = tmp_path
    original_api_key = server._API_KEY
    server._API_KEY = ""

    # Stub model that returns a fabricated VoiceClonePromptItem-like object.
    class _FakeItem:
        ref_spk_embedding = torch.zeros(2048, dtype=torch.float32)
        ref_code = None
        x_vector_only_mode = True
        icl_mode = False
        ref_text = None

    class _FakeModel:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            # Ensure the temp file was written as expected.
            assert Path(ref_audio).exists()
            return [_FakeItem()]

    reg.bind_model(_FakeModel())
    # Tests run without a model-loaded lifespan, so force ready=True for the
    # registration endpoint's 503 gate.
    original_ready = server._model_ready
    server._model_ready = True
    # The registration handler acquires the inference semaphore; make sure
    # it exists (normally created in lifespan).
    if not hasattr(server, "_INFER_SEM") or not isinstance(getattr(server, "_INFER_SEM", None), asyncio.Semaphore):
        server._INFER_SEM = asyncio.Semaphore(1)

    try:
        client = TestClient(server.app)
        # 1) Register
        audio_bytes = b"RIFF0000WAVEfmt "  # fake bytes — our fake model ignores them
        r = client.post(
            "/v1/voices",
            files={"audio": ("sample.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data={"name": "Alice", "language": "English"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        voice_id = body["id"]
        assert voice_id.startswith("vc_")
        assert body["kind"] == "custom"
        assert body["name"] == "Alice"

        # Sidecar files on disk
        assert (tmp_path / f"{voice_id}.safetensors").is_file()
        sidecar = json.loads((tmp_path / f"{voice_id}.json").read_text())
        assert sidecar["name"] == "Alice"
        assert sidecar["id"] == voice_id

        # 2) List includes it
        r = client.get("/v1/voices")
        assert r.status_code == 200
        ids = {v["id"] for v in r.json()["data"]}
        assert voice_id in ids

        # 3) Registered voice is resolvable and usable as a voice parameter
        assert server._resolve_voice(voice_id) == voice_id

        # 4) Delete
        r = client.delete(f"/v1/voices/{voice_id}")
        assert r.status_code == 200
        assert r.json() == {"id": voice_id, "deleted": True}
        assert not (tmp_path / f"{voice_id}.safetensors").exists()
        assert not (tmp_path / f"{voice_id}.json").exists()

        # 5) Gone from listing
        r = client.get("/v1/voices")
        ids = {v["id"] for v in r.json()["data"]}
        assert voice_id not in ids
    finally:
        reg.custom_dir = original_dir
        server._API_KEY = original_api_key
        server._model_ready = original_ready
        reg.bind_model(None)


def test_register_voice_auto_transcribes_when_asr_configured(tmp_path):
    """When ref_text is omitted and ASR_URL is set, we forward audio to ASR.

    Mocks httpx.AsyncClient to verify the ASR endpoint is called and the
    returned text flows into the sidecar metadata.
    """
    from fastapi.testclient import TestClient
    import torch

    server = _import_server()
    reg = server._voice_registry

    # Redirect registry to a temp dir.
    original_dir = reg.custom_dir
    reg.custom_dir = tmp_path
    original_asr_url = server.ASR_URL
    original_api_key = server._API_KEY
    original_ready = server._model_ready
    server.ASR_URL = "http://fake-asr:8000"
    server._API_KEY = ""
    server._model_ready = True
    if not isinstance(getattr(server, "_INFER_SEM", None), asyncio.Semaphore):
        server._INFER_SEM = asyncio.Semaphore(1)

    captured: dict = {}

    class _FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return {"text": "Hello world transcribed by ASR."}

    class _FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def post(self, url, data=None, files=None, headers=None):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            captured["file_keys"] = list((files or {}).keys())
            return _FakeResponse()

    class _FakeHttpxModule:
        AsyncClient = _FakeAsyncClient

    # Stub qwen_tts's create_voice_clone_prompt on the registry's model.
    class _FakeItem:
        ref_spk_embedding = torch.zeros(2048, dtype=torch.float32)
        ref_code = None
        x_vector_only_mode = False
        icl_mode = True
        ref_text = "set-by-fake"

    captured_prompt: dict = {}

    class _FakeModel:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            captured_prompt["ref_text"] = ref_text
            captured_prompt["x_vector_only_mode"] = x_vector_only_mode
            return [_FakeItem()]

    reg.bind_model(_FakeModel())

    try:
        with patch.dict(sys.modules, {"httpx": _FakeHttpxModule}):
            client = TestClient(server.app)
            r = client.post(
                "/v1/voices",
                files={"audio": ("bob.wav", io.BytesIO(b"RIFFwav"), "audio/wav")},
                data={"name": "Bob", "language": "English"},
            )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["auto_transcribed"] is True

        # ASR was called with the right URL + multipart file.
        assert captured["url"] == "http://fake-asr:8000/v1/audio/transcriptions"
        assert captured["file_keys"] == ["file"]
        assert captured["data"]["language"] == "English"

        # The transcribed text flowed into create_voice_clone_prompt as
        # ref_text, and we took the ICL path (x_vector_only_mode=False).
        assert captured_prompt["ref_text"] == "Hello world transcribed by ASR."
        assert captured_prompt["x_vector_only_mode"] is False
    finally:
        reg.custom_dir = original_dir
        server.ASR_URL = original_asr_url
        server._API_KEY = original_api_key
        server._model_ready = original_ready
        reg.bind_model(None)


def test_register_voice_falls_back_when_asr_fails(tmp_path):
    """When the ASR call raises, we silently use x_vector_only_mode."""
    from fastapi.testclient import TestClient
    import torch

    server = _import_server()
    reg = server._voice_registry

    original_dir = reg.custom_dir
    reg.custom_dir = tmp_path
    original_asr_url = server.ASR_URL
    original_api_key = server._API_KEY
    original_ready = server._model_ready
    server.ASR_URL = "http://fake-asr:8000"
    server._API_KEY = ""
    server._model_ready = True
    if not isinstance(getattr(server, "_INFER_SEM", None), asyncio.Semaphore):
        server._INFER_SEM = asyncio.Semaphore(1)

    class _ExplodingClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def post(self, *a, **kw):
            raise RuntimeError("connection refused (test)")

    class _FakeHttpxModule:
        AsyncClient = _ExplodingClient

    class _FakeItem:
        ref_spk_embedding = torch.zeros(2048, dtype=torch.float32)
        ref_code = None
        x_vector_only_mode = True
        icl_mode = False
        ref_text = None

    captured_prompt: dict = {}

    class _FakeModel:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            captured_prompt["x_vector_only_mode"] = x_vector_only_mode
            return [_FakeItem()]

    reg.bind_model(_FakeModel())

    try:
        with patch.dict(sys.modules, {"httpx": _FakeHttpxModule}):
            client = TestClient(server.app)
            r = client.post(
                "/v1/voices",
                files={"audio": ("x.wav", io.BytesIO(b"wav"), "audio/wav")},
                data={"name": "NoTranscript"},
            )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["auto_transcribed"] is False
        # Fell back to x_vector_only_mode because ASR errored.
        assert captured_prompt["x_vector_only_mode"] is True
    finally:
        reg.custom_dir = original_dir
        server.ASR_URL = original_asr_url
        server._API_KEY = original_api_key
        server._model_ready = original_ready
        reg.bind_model(None)


def test_register_voice_skips_asr_when_ref_text_supplied(tmp_path):
    """If the client provides ref_text we must NOT call ASR even when configured."""
    from fastapi.testclient import TestClient
    import torch

    server = _import_server()
    reg = server._voice_registry

    original_dir = reg.custom_dir
    reg.custom_dir = tmp_path
    original_asr_url = server.ASR_URL
    original_api_key = server._API_KEY
    original_ready = server._model_ready
    server.ASR_URL = "http://fake-asr:8000"
    server._API_KEY = ""
    server._model_ready = True
    if not isinstance(getattr(server, "_INFER_SEM", None), asyncio.Semaphore):
        server._INFER_SEM = asyncio.Semaphore(1)

    calls = {"count": 0}

    class _CountingClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def post(self, *a, **kw):
            calls["count"] += 1
            raise AssertionError("ASR must not be called when ref_text is supplied")

    class _FakeHttpxModule:
        AsyncClient = _CountingClient

    class _FakeItem:
        ref_spk_embedding = torch.zeros(2048, dtype=torch.float32)
        ref_code = None
        x_vector_only_mode = False
        icl_mode = True
        ref_text = "provided"

    class _FakeModel:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            return [_FakeItem()]

    reg.bind_model(_FakeModel())

    try:
        with patch.dict(sys.modules, {"httpx": _FakeHttpxModule}):
            client = TestClient(server.app)
            r = client.post(
                "/v1/voices",
                files={"audio": ("x.wav", io.BytesIO(b"wav"), "audio/wav")},
                data={"name": "Supplied", "ref_text": "The rain in Spain."},
            )
        assert r.status_code == 201, r.text
        assert calls["count"] == 0
        assert r.json()["auto_transcribed"] is False
    finally:
        reg.custom_dir = original_dir
        server.ASR_URL = original_asr_url
        server._API_KEY = original_api_key
        server._model_ready = original_ready
        reg.bind_model(None)


def test_register_voice_auto_transcribe_false_skips_asr(tmp_path):
    """Passing auto_transcribe=false forces x-vector-only even when ASR is set."""
    from fastapi.testclient import TestClient
    import torch

    server = _import_server()
    reg = server._voice_registry

    original_dir = reg.custom_dir
    reg.custom_dir = tmp_path
    original_asr_url = server.ASR_URL
    original_api_key = server._API_KEY
    original_ready = server._model_ready
    server.ASR_URL = "http://fake-asr:8000"
    server._API_KEY = ""
    server._model_ready = True
    if not isinstance(getattr(server, "_INFER_SEM", None), asyncio.Semaphore):
        server._INFER_SEM = asyncio.Semaphore(1)

    calls = {"count": 0}

    class _CountingClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def post(self, *a, **kw):
            calls["count"] += 1
            raise AssertionError("ASR must not be called when auto_transcribe=false")

    class _FakeHttpxModule:
        AsyncClient = _CountingClient

    class _FakeItem:
        ref_spk_embedding = torch.zeros(2048, dtype=torch.float32)
        ref_code = None
        x_vector_only_mode = True
        icl_mode = False
        ref_text = None

    captured_prompt: dict = {}

    class _FakeModel:
        def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
            captured_prompt["x_vector_only_mode"] = x_vector_only_mode
            return [_FakeItem()]

    reg.bind_model(_FakeModel())

    try:
        with patch.dict(sys.modules, {"httpx": _FakeHttpxModule}):
            client = TestClient(server.app)
            r = client.post(
                "/v1/voices",
                files={"audio": ("x.wav", io.BytesIO(b"wav"), "audio/wav")},
                data={"name": "OptOut", "auto_transcribe": "false"},
            )
        assert r.status_code == 201, r.text
        assert calls["count"] == 0
        assert r.json()["auto_transcribed"] is False
        assert captured_prompt["x_vector_only_mode"] is True
    finally:
        reg.custom_dir = original_dir
        server.ASR_URL = original_asr_url
        server._API_KEY = original_api_key
        server._model_ready = original_ready
        reg.bind_model(None)


def test_register_voice_requires_model_ready():
    """When _model_ready is False, POST /v1/voices must return 503."""
    from fastapi.testclient import TestClient
    server = _import_server()
    original_ready = server._model_ready
    original_api_key = server._API_KEY
    server._model_ready = False
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.post(
            "/v1/voices",
            files={"audio": ("s.wav", io.BytesIO(b"RIFF"), "audio/wav")},
            data={"name": "X"},
        )
        assert r.status_code == 503
    finally:
        server._model_ready = original_ready
        server._API_KEY = original_api_key


# -----------------------------------------------------------------------
# JSON body parsing
# -----------------------------------------------------------------------

class _FakeRequest:
    """Minimal Request stand-in for _parse_params testing."""

    def __init__(self, json_body: dict | None = None):
        self._json = json_body
        self.headers = {
            "content-type": "application/json" if json_body is not None else "multipart/form-data"
        }

    async def json(self):
        return self._json or {}


def test_parse_params_json_body_input_field():
    """OpenAI SDK sends 'input' in JSON body; should map to 'text'."""
    async def _run():
        server = _import_server()
        req = _FakeRequest({"input": "Hello world", "voice": "alloy"})
        p = await server._parse_params(req, text=None, voice="ryan", response_format="mp3",
                                        instruct=None, language="English", speed=1.0)
        assert p["text"] == "Hello world"
        assert p["voice"] == "alloy"
    asyncio.run(_run())


def test_parse_params_json_body_text_alias():
    """Our 'text' alias in JSON body should also work."""
    async def _run():
        server = _import_server()
        req = _FakeRequest({"text": "Bonjour", "language": "French"})
        p = await server._parse_params(req, text=None, voice="ryan", response_format="mp3",
                                        instruct=None, language="English", speed=1.0)
        assert p["text"] == "Bonjour"
        assert p["language"] == "French"
    asyncio.run(_run())


def test_parse_params_query_fallback():
    """Query params should be used when no JSON body is present."""
    async def _run():
        server = _import_server()
        req = _FakeRequest(None)  # no JSON body
        p = await server._parse_params(req, text="Query text", voice="eric",
                                        response_format="wav", instruct=None,
                                        language="English", speed=1.0)
        assert p["text"] == "Query text"
        assert p["voice"] == "eric"
        assert p["response_format"] == "wav"
    asyncio.run(_run())


# -----------------------------------------------------------------------
# ffmpeg pipe transcoder (mocked — no ffmpeg binary needed in CI)
# -----------------------------------------------------------------------

def test_transcode_wav_mp3_calls_ffmpeg():
    server = _import_server()
    fake_wav = b"RIFF....WAVEfmt "  # minimal fake WAV header
    fake_mp3 = b"\xff\xfb\x90\x00"  # fake MP3 header

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = fake_mp3
        mock_run.return_value.returncode = 0
        result = server._transcode_wav(fake_wav, "mp3")

    assert result == fake_mp3
    cmd = mock_run.call_args[0][0]
    assert "pipe:0" in cmd   # stdin input
    assert "pipe:1" in cmd   # stdout output
    assert "libmp3lame" in cmd


def test_transcode_wav_opus_calls_ffmpeg():
    server = _import_server()
    fake_wav = b"RIFF....WAVEfmt "
    fake_ogg = b"OggS"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = fake_ogg
        mock_run.return_value.returncode = 0
        result = server._transcode_wav(fake_wav, "opus")

    assert result == fake_ogg
    cmd = mock_run.call_args[0][0]
    assert "libopus" in cmd


# -----------------------------------------------------------------------
# Metrics endpoint and auth middleware (in-process via TestClient)
# -----------------------------------------------------------------------

def test_metrics_endpoint_exposed_without_auth():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        assert "qwen3_tts_requests_total" in body or "qwen3_tts_request_duration_seconds" in body
        assert "qwen3_tts_model_ready" in body
    finally:
        server._API_KEY = original


def test_auth_blocks_v1_when_key_set():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = "secret-key"
    try:
        client = TestClient(server.app)
        # /v1/models requires model_ready; here we only check auth — any
        # non-200 is fine as long as unauthenticated hits 401 first.
        r = client.get("/v1/models")
        assert r.status_code == 401
        r = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401
        # /v1/voices also requires auth
        r = client.get("/v1/voices")
        assert r.status_code == 401
        # Health and metrics must remain unauthenticated
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
    finally:
        server._API_KEY = original


def test_auth_disabled_when_no_key():
    from fastapi.testclient import TestClient
    server = _import_server()
    original = server._API_KEY
    server._API_KEY = ""
    try:
        client = TestClient(server.app)
        # With auth off, unauth requests hit the endpoint logic (which may
        # return 503 "model loading" — that's fine, we're only checking that
        # the middleware doesn't block with 401).
        r = client.get("/health")
        assert r.status_code == 200
    finally:
        server._API_KEY = original
