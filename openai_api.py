import asyncio
import logging
import queue
import time
from typing import List, Literal, Optional, Union

import chatglm_cpp
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread, Event
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse,ServerSentEvent
import uvicorn
from queue import Queue

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")
message_queue = Queue()
end_flag_received = False

def handle_message(message, token_cache):
    global end_flag_received
    #print("python: message", message, "token_ids", token_cache)
    message_queue.put(DeltaMessage(content=message))
    if  token_cache[0] == 2:
         end_flag_received = True
    # return ChatCompletionResponse(
    #     object="chat.completion.chunk",
    #     choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(content=message))],
    # )
    #generator = chunk.model_dump_json(exclude_unset=True)
    #EventSourceResponse(generator)
    #EventSourceResponse(generator, media_type="text/event-stream")
    # delta = DeltaMessage(
    #     content=message,
    #     role="assistant",
    #     function_call= None,
    # )
    #
    # choice_data = ChatCompletionResponseStreamChoice(
    #     index=0,
    #     delta=delta,
    #     finish_reason=None,
    # )
    # chunk = ChatCompletionResponse(choices=[choice_data], object="chat.completion.chunk")
    # generator = "{}".format(chunk.model_dump_json(exclude_unset=True))
    # event = ServerSentEvent(generator)
    # response.body.write(event.encode())
    # response.body.flush()
    #Response(content=generator, media_type="text/event-stream")


class Settings(BaseSettings):
    model: str = "C:\\Xiaowei\Xiaowei AI 2023\\demo\\openvino.genai\\model\\GPTQ_INT4-FP16\\"
    on_device: str = "GPU"
    num_threads: int = 0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)

    model_config = {
        "json_schema_extra": {"examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]}
    }


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl"
    model: str = "default-model"
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice], List[ChatCompletionResponseStreamChoice]]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl",
                    "model": "default-model",
                    "object": "chat.completion",
                    "created": 1691166146,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 29, "total_tokens": 46},
                }
            ]
        }
    }


settings = Settings()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
pipeline = chatglm_cpp.Pipeline(settings.model, settings.on_device)
lock = asyncio.Lock()


def stream_chat(messages, body):
    global end_flag_received
    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(role="assistant"))],
    )

    #for chunk in pipeline.chat(
    t = Thread(target=pipeline.chat,
               kwargs=dict(
                   messages=messages,
                   max_length=body.max_tokens,
                   do_sample=body.temperature > 0,
                   top_p=body.top_p,
                   temperature=body.temperature,
                   num_threads=settings.num_threads,
                   stream=True,
                   callback=handle_message
               )
               )

    t.daemon = True
    t.start()

    while True:
        try:
            # 从队列中获取消息，设置合适的超时时间
            chunk = message_queue.get(timeout=1)
            # 处理消息，例如发送 SSE 数据给客户端
            yield ChatCompletionResponse(
                object="chat.completion.chunk",
                choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(content=chunk.content))],
            )
        except queue.Empty:
            # 处理队列超时的情况
            pass
        if end_flag_received and message_queue.empty():
            break

    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    end_flag_received = False


async def predict(messages, body):

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        num_threads=settings.num_threads,
        stream=True,
        callback=handle_message
    )

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


async def stream_chat_event_publisher(history, body):
    output = ""
    try:
        async with lock:
             for chunk in stream_chat(history, body):
                await asyncio.sleep(0)  # yield control back to event loop for cancellation check
                output += chunk.choices[0].delta.content or ""
                yield chunk.model_dump_json(exclude_unset=True)
        logging.info(f'prompt: "{history[-1]}", stream response: "{output}"')
    except asyncio.CancelledError as e:
        logging.info(f'prompt: "{history[-1]}", stream response (partial): "{output}"')
        raise e

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages = [chatglm_cpp.ChatMessage(role=msg.role, content=msg.content) for msg in body.messages]

    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator,  media_type="text/event-stream")

    max_context_length = 512
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        callback=handle_message
    )
    logging.info(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
    # prompt_tokens = len(pipeline.tokenizer.encode_messages(messages, max_context_length))
    # completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))
    prompt_tokens =  pipeline.prompt_tokens
    completion_tokens = pipeline.completion_tokens

    return ChatCompletionResponse(
        object="chat.completion",
        choices=[ChatCompletionResponseChoice(message=ChatMessage(role="assistant", content=output.content))],
        usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "owner"
    permission: List = []


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "owner", "permission": []}],
                }
            ]
        }
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id="gpt-3.5-turbo")])

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)