from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
from typing import Dict, Optional, Union, List, Tuple
from ... import OpenAIWrapper
from openai.types.completion import Completion 
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage # type: ignore [attr-defined]
from ..._pydantic import model_dump

class AzureCognitiveSearchAgent(ConversableAgent):
    """An agent that uses Azure Cognitive Search to retrieve responses."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_reply([Agent, None], AzureCognitiveSearchAgent.generate_oai_reply)

    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
   
        # unroll tool_responses
        all_messages = []
        for message in messages:
            tool_responses = message.get("tool_responses", [])
            if tool_responses:
                all_messages += tool_responses
                # tool role on the parent message means the content is just concatenation of all of the tool_responses
                if message.get("role") != "tool":
                    all_messages.append({key: message[key] for key in message if key != "tool_responses"})
            else:
                all_messages.append(message)

        # TODO: #1143 handle token limit exceeded error
        response = client.create(
            context=messages[-1].pop("context", None),
            messages=self._oai_system_message + all_messages,
            cache=self.client_cache,
        )

        extracted_response = self.extract_text_or_completion_object(response)[0]

        # ensure function and tool calls will be accepted when sent back to the LLM
        if not isinstance(extracted_response, str):
            extracted_response = model_dump(extracted_response)
        if isinstance(extracted_response, dict):
            if extracted_response.get("function_call"):
                extracted_response["function_call"]["name"] = self._normalize_name(
                    extracted_response["function_call"]["name"]
                )
            for tool_call in extracted_response.get("tool_calls") or []:
                tool_call["function"]["name"] = self._normalize_name(tool_call["function"]["name"])
        return True, extracted_response
     
    def extract_text_or_completion_object(
        cls, response: Union[ChatCompletion, Completion]
    ) -> Union[List[str], List[ChatCompletionMessage]]:
        """Extract the text or ChatCompletion objects from a completion or chat response.

        Args:
            response (ChatCompletion | Completion): The response from openai.

        Returns:
            A list of text, or a list of ChatCompletion objects if function_call/tool_calls are present.
        """
        choices = response.choices
        if isinstance(response, Completion):
            return [choice.text for choice in choices]  # type: ignore [union-attr]

        return [  # type: ignore [return-value]
            choice.message
            if choice.message is not None and (choice.message.function_call is not None or choice.message.tool_calls is not None)
            else choice.message.content
            if choice.message is not None
            else choice.messages[-1]["content"]
            if choice.messages is not None and len(choice.messages) > 0 and "content" in choice.messages[-1]
            else None
            for choice in choices
        ]