import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional

import openai
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from openai import AsyncOpenAI

from utils import resp_200

app = FastAPI()
logger = logging.getLogger(__name__)


def make_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


def get_api_key(authorization: Optional[str]) -> str:
    if authorization:
        scheme, _, token = authorization.strip().partition(" ")
        api_key = token.strip() if scheme.lower() == "bearer" else authorization.strip()
        if api_key:
            return api_key
    raise HTTPException(status_code=401, detail="OPENAI_API_KEY is required.")


def upstream_error_response(e: openai.OpenAIError) -> JSONResponse:
    if isinstance(e, openai.APIStatusError):
        try:
            content = e.response.json()
        except ValueError:
            content = {"error": {"message": e.message}}
        return JSONResponse(status_code=e.status_code, content=content)
    if isinstance(e, openai.APITimeoutError):
        return JSONResponse(status_code=504, content={"error": {"message": str(e)}})
    if isinstance(e, openai.APIConnectionError):
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})
    return JSONResponse(status_code=500, content={"error": {"message": str(e)}})


@app.get("/")
def do_echo():
    return {"message": "This is a greet from FastAPI.", "timestamp": datetime.now()}


async def sse_events(client: AsyncOpenAI, stream: Any) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            yield f"data: {chunk.to_json(indent=None)}\n\n"
        yield "data: [DONE]\n\n"
    except openai.OpenAIError as e:
        logger.error(e)
        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
    finally:
        await client.close()


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def do_proxy_chat(request: dict, authorization: Optional[str] = Header(None)):
    api_key = get_api_key(authorization)
    params = dict(request)
    model = params.pop("model", None)
    messages = params.pop("messages", None)
    stream = bool(params.pop("stream", False))
    if not model or messages is None:
        raise HTTPException(status_code=400, detail="'model' and 'messages' are required.")

    client = make_client(api_key)
    try:
        completion = await client.chat.completions.create(
            model=model, messages=messages, stream=stream, extra_body=params
        )
    except openai.OpenAIError as e:
        logger.error(e)
        await client.close()
        return upstream_error_response(e)

    if stream:
        return StreamingResponse(
            sse_events(client, completion),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    await client.close()
    return resp_200(data=completion.to_dict())


@app.post("/v1/audio/transcriptions")
async def do_proxy_whisper(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    authorization: Optional[str] = Header(None),
):
    api_key = get_api_key(authorization)
    contents = await file.read()
    optional = {
        "language": language,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature,
    }
    kwargs = {k: v for k, v in optional.items() if v is not None}

    async with make_client(api_key) as client:
        try:
            transcript = await client.audio.transcriptions.create(
                file=(file.filename or "audio", contents, file.content_type or "application/octet-stream"),
                model=model,
                **kwargs,
            )
        except openai.OpenAIError as e:
            logger.error(e)
            return upstream_error_response(e)

    if isinstance(transcript, str):
        return PlainTextResponse(transcript)
    return resp_200(data=transcript.to_dict())
