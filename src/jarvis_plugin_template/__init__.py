import abc
from typing import Any, Dict, List, Optional, Tuple, TypeVar, TypedDict

from abstract_singleton import AbstractSingleton, Singleton

PromptGenerator = TypeVar("PromptGenerator")


class Message(TypedDict):
    role: str
    content: str


class JarvisPluginTemplate(AbstractSingleton, metaclass=Singleton):
    def __init__(self):
        super().__init__()

        self._name = "Jarvis Plugin Template"
        self._version = "0.1.0"
        self._description = "This is a template for Jarvis plugins."

    @abc.abstractmethod
    def can_handle_on_response(self) -> bool:
        return False

    @abc.abstractmethod
    def on_response(self, response: str, *args, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def can_handle_post_prompt(self) -> bool:
        return False

    @abc.abstractmethod
    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        pass

    @abc.abstractmethod
    def can_handle_on_planning(self) -> bool:
        return False

    @abc.abstractmethod
    def on_planning(self, prompt: PromptGenerator, message: List[Message]) -> Optional[str]:
        pass

    @abc.abstractmethod
    def can_handle_post_planning(self) -> bool:
        return False

    @abc.abstractmethod
    def post_planning(self, response: str) -> str:
        pass

    @abc.abstractmethod
    def can_handle_pre_instruction(self) -> bool:
        return False

    @abc.abstractmethod
    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        pass

    @abc.abstractmethod
    def can_handle_on_instruction(self) -> bool:
        return False

    @abc.abstractmethod
    def on_instruction(self, message: List[Message]) -> Optional[str]:
        pass

    @abc.abstractmethod
    def can_handle_post_instruction(self) -> bool:
        return False

    @abc.abstractmethod
    def post_instruction(self, response: str) -> str:
        pass

    @abc.abstractmethod
    def can_handle_pre_command(self) -> bool:
        return False

    @abc.abstractmethod
    def pre_command(
            self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def can_handle_post_command(self) -> bool:
        return False

    @abc.abstractmethod
    def post_command(self, command_name: str, response: str) -> str:
        pass

    @abc.abstractmethod
    def can_handle_chat_completion(
            self, message: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        return False

    @abc.abstractmethod
    def handle_chat_completion(
            self, messages: List[Message], model: str, temperature: float, max_tokens: int
    ) -> str:
        pass

    @abc.abstractmethod
    def can_handle_text_embedding(self, text: str) -> bool:
        return False

    @abc.abstractmethod
    def handle_text_embedding(self, text: str) -> list:
        pass

    @abc.abstractmethod
    def can_handle_user_input(self, user_input: str) -> bool:
        return False

    @abc.abstractmethod
    def user_input(self, user_input: str) -> str:
        pass

    @abc.abstractmethod
    def can_handle_report(self) -> bool:
        return False

    @abc.abstractmethod
    def report(self, message: str) -> None:
        pass
