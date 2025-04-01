from functools import partial
from operator import call

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from httpx import Response as HttpxResponse

from chat.client import openai

from .vllm_schema import ChatCompletionRequest

router = APIRouter()


def to_fastapi_response(httpx_response: HttpxResponse):
    assert httpx_response.is_closed
    return Response(content=httpx_response.content, status_code=httpx_response.status_code, media_type=httpx_response.headers.get("Content-Type"))


@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest):
    if not body.stream:
        res = await openai.with_raw_response.chat.completions.create(**body.as_kwargs)
        return to_fastapi_response(res.http_response)

    res = await openai.chat.completions.create(stream=True, **body.as_kwargs)

    @partial(StreamingResponse, media_type=res.response.headers.get("Content-Type"), status_code=res.response.status_code)
    @call
    async def response():
        async for chunk in res.response.aiter_text():
            yield chunk

    return response
