from app.multimodal.vision_adapters import (
    OpenAICompatibleVisionAdapter,
    QwenVisionAdapter,
    build_vision_response_adapter,
)


class _Message:
    def __init__(self, content, reasoning=None) -> None:
        self.content = content
        self.reasoning = reasoning


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message


class _Response:
    def __init__(self, message: _Message) -> None:
        self.choices = [_Choice(message)]


def test_build_vision_response_adapter_selects_qwen_adapter() -> None:
    adapter = build_vision_response_adapter("qwen3-vl:2b")
    assert isinstance(adapter, QwenVisionAdapter)


def test_openai_compatible_adapter_uses_content_only() -> None:
    adapter = OpenAICompatibleVisionAdapter()
    response = _Response(_Message(content="", reasoning="fallback reasoning"))
    assert adapter.extract_text(response) == ""


def test_qwen_adapter_falls_back_to_reasoning_when_content_is_empty() -> None:
    adapter = QwenVisionAdapter()
    response = _Response(_Message(content="", reasoning="detected a vehicle crossing"))
    assert "vehicle crossing" in adapter.extract_text(response)

