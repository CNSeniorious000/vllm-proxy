from contextlib import suppress
from functools import partial
from json import JSONDecodeError, loads
from operator import call

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from httpx import Response as HttpxResponse
from langfuse.decorators import langfuse_context, observe

from chat.client import openai

from .vllm_schema import ChatCompletionRequest

router = APIRouter(tags=["Chat Completion"])


def to_fastapi_response(httpx_response: HttpxResponse):
    assert httpx_response.is_closed
    return Response(content=httpx_response.content, status_code=httpx_response.status_code, media_type=httpx_response.headers.get("Content-Type"))


@router.post("/v1/chat/completions", description="Alias for /chat/completions", deprecated=True)
@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    """See [vLLM's docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-api) for more details"""

    def start():
        print(langfuse_context.get_current_trace_url())
        langfuse_context.update_current_observation(input=body.messages, model=body.model, model_parameters=body.extra_body)
        langfuse_context.update_current_trace(input=body.model_dump(exclude_unset=True), metadata={**request.headers})
        langfuse_context.flush()

    def end(text: str):
        langfuse_context.update_current_observation(output=text)
        langfuse_context.update_current_trace(output=text)
        langfuse_context.flush()

    if not body.stream:

        @observe(capture_output=False, capture_input=False, as_type="generation", name="non-streaming response")
        async def completion():
            start()
            res = await openai.with_raw_response.chat.completions.create(stream=False, **body.as_kwargs)
            parsed = res.parse()
            text = parsed.choices[0].message.content
            assert text is not None, parsed
            end(text)
            return to_fastapi_response(res.http_response)

        return await completion()

    res = await openai.chat.completions.create(stream=True, **body.as_kwargs)

    @partial(StreamingResponse, media_type=res.response.headers.get("Content-Type"), status_code=res.response.status_code)
    @call
    @observe(capture_output=False, capture_input=False, as_type="generation", name="streaming response")
    async def response():
        start()
        text = ""
        async for message in res.response.aiter_lines():
            if message:
                yield message + "\n\n"
                with suppress(IndexError, KeyError, JSONDecodeError):
                    text += loads(message.removeprefix("data: "))["choices"][0]["delta"]["content"]
        end(text)

    return response
