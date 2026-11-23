import json

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

import api

CHAT_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1680242783,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def stream_chunk(content, finish_reason=None):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1680242783,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
    }


class Upstream:
    """Fake OpenAI HTTP layer: records requests and replies via the configured handler."""

    def __init__(self):
        self.requests = []
        self.handler = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        request.read()
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
def upstream(monkeypatch):
    fake = Upstream()

    def make_client(api_key):
        return AsyncOpenAI(
            api_key=api_key,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(fake)),
        )

    monkeypatch.setattr(api, "make_client", make_client)
    return fake


@pytest.fixture
def client():
    return TestClient(api.app)


AUTH = {"Authorization": "Bearer sk-test"}


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_chat_non_streaming(client, upstream, path):
    upstream.handler = lambda req: httpx2.Response(200, json=CHAT_COMPLETION)
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    }

    resp = client.post(path, json=body, headers=AUTH)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hello!"
    sent = upstream.requests[0]
    assert sent.url.path == "/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer sk-test"
    sent_body = json.loads(sent.content)
    assert sent_body["temperature"] == 0.2
    assert sent_body["tools"] == body["tools"]
    assert sent_body["model"] == "gpt-4o-mini"


def test_chat_streaming(client, upstream):
    sse = "".join(
        f"data: {json.dumps(c)}\n\n" for c in [stream_chunk("Hel"), stream_chunk("lo", "stop")]
    ) + "data: [DONE]\n\n"
    upstream.handler = lambda req: httpx2.Response(
        200, content=sse.encode(), headers={"content-type": "text/event-stream"}
    )

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        headers=AUTH,
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [line[len("data: "):] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    content = "".join(json.loads(e)["choices"][0]["delta"]["content"] for e in events[:-1])
    assert content == "Hello"
    assert json.loads(upstream.requests[0].content)["stream"] is True


def test_per_request_api_key(client, upstream):
    upstream.handler = lambda req: httpx2.Response(200, json=CHAT_COMPLETION)
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}

    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-a"})
    client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer sk-b"})

    assert [r.headers["authorization"] for r in upstream.requests] == ["Bearer sk-a", "Bearer sk-b"]


@pytest.mark.parametrize("model", [None, "gpt-4o-transcribe"])
def test_transcription(client, upstream, model):
    upstream.handler = lambda req: httpx2.Response(200, json={"text": "hello world"})
    data = {"model": model} if model else {}

    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.mp3", b"fake-audio-bytes", "audio/mpeg")},
        data=data,
        headers=AUTH,
    )

    assert resp.status_code == 200
    assert resp.json() == {"text": "hello world"}
    sent = upstream.requests[0]
    assert sent.url.path == "/v1/audio/transcriptions"
    assert sent.headers["authorization"] == "Bearer sk-test"
    assert b'filename="speech.mp3"' in sent.content
    assert b"fake-audio-bytes" in sent.content
    assert (model or "whisper-1").encode() in sent.content


@pytest.mark.parametrize(
    "method_kwargs",
    [
        {"url": "/v1/chat/completions", "json": {"model": "m", "messages": []}},
        {"url": "/v1/completions", "json": {"model": "m", "messages": []}},
        {"url": "/v1/audio/transcriptions", "files": {"file": ("a.mp3", b"x", "audio/mpeg")}},
    ],
)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer "}])
def test_missing_auth_returns_401(client, upstream, method_kwargs, headers):
    resp = client.post(headers=headers, **method_kwargs)

    assert resp.status_code == 401
    assert upstream.requests == []


@pytest.mark.parametrize(
    "method_kwargs",
    [
        {"url": "/v1/chat/completions", "json": {"model": "m", "messages": []}},
        {"url": "/v1/chat/completions", "json": {"model": "m", "messages": [], "stream": True}},
        {"url": "/v1/audio/transcriptions", "files": {"file": ("a.mp3", b"x", "audio/mpeg")}},
    ],
)
@pytest.mark.parametrize("status", [401, 429])
def test_upstream_error_passthrough(client, upstream, method_kwargs, status):
    error = {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}}
    upstream.handler = lambda req: httpx2.Response(status, json=error)

    resp = client.post(headers=AUTH, **method_kwargs)

    assert resp.status_code == status
    assert resp.json() == error
