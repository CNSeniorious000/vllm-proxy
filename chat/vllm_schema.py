from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from promplate import Message
from pydantic import BaseModel, Field

from env import default_model


class BaseModel(BaseModel):
    model_config = {"extra": "allow"}


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: str | None = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class ResponseFormat(BaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = True
    continuous_usage_stats: bool | None = False


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ChatCompletionToolsParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ChatCompletionRequest(BaseModel):
    @property
    def extra_body(self):
        return self.model_dump(
            exclude={"messages", "model", "stream"},
            exclude_defaults=True,
        )

    @property
    def as_kwargs(self):
        return {"messages": self.messages, "model": self.model, "extra_body": self.extra_body}

    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[Message]
    model: str | None = Field(default_model)
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
    )
    max_completion_tokens: int | None = None
    n: int = 1
    presence_penalty: float | None = 0.0
    response_format: ResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = Field(default_factory=list)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: Literal["none"] | Literal["auto"] | ChatCompletionNamedToolChoiceParam | None = None

    # NOTE this will be ignored by vLLM -- the model determines the behavior
    parallel_tool_calls: bool | None = False
    user: str | None = None

    # doc: begin-chat-completion-sampling-params
    best_of: int | None = None
    use_beam_search: bool = False
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    length_penalty: float = 1.0
    stop_token_ids: list[int] | None = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: int | None = Field(default=None, ge=1)
    prompt_logprobs: int | None = None
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=("If true, the new message will be prepended with the last message if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=("If true, the generation prompt will be added to the chat template. This is a parameter used by chat template in tokenizer config of the model."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            'This allows you to "prefill" part of the model\'s response for it. '
            "Cannot be used at the same time as `add_generation_prompt`."
        ),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=("If true, special tokens (e.g. BOS) will be added to the prompt on top of what is added by the chat template. For most models, the chat template takes care of adding the special tokens so this should be set to false (as is the default)."),
    )
    documents: list[dict[str, str]] | None = Field(
        default=None,
        description=('A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation). If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing "title" and "text" keys.'),
    )
    chat_template: str | None = Field(
        default=None,
        description=("A Jinja template to use for this conversion. As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the tokenizer does not define one."),
    )
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. Will be accessible by the chat template."),
    )
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    guided_json: str | dict | BaseModel | None = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: str | None = Field(
        default=None,
        description=("If specified, the output will follow the regex pattern."),
    )
    guided_choice: list[str] | None = Field(
        default=None,
        description=("If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: str | None = Field(
        default=None,
        description=("If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: str | None = Field(
        default=None,
        description=("If specified, will override the default guided decoding backend of the server for this specific request. If set, must be either 'outlines' / 'lm-format-enforcer'"),
    )
    guided_whitespace_pattern: str | None = Field(
        default=None,
        description=("If specified, will override the default whitespace pattern for guided json decoding."),
    )
    priority: int = Field(
        default=0,
        description=("The priority of the request (lower means earlier handling; default: 0). Any priority other than 0 will raise an error if the served model does not use priority scheduling."),
    )
    request_id: str = Field(
        default_factory=lambda: f"{uuid4().hex}",
        description=("The request_id related to this request. If the caller does not set it, a random uuid will be generated. This id is used through out the inference process and return in response."),
    )
    return_tokens_as_token_ids: bool | None = Field(
        default=None,
        description=("If specified with 'logprobs', tokens are represented  as strings of the form 'token_id:{token_id}' so that tokens that are not JSON-encodable can be identified."),
    )

    # doc: end-chat-completion-extra-params

    # Default sampling parameters for chat completion requests
    _DEFAULT_SAMPLING_PARAMS: dict = {
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
    }
