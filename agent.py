import inspect
import json
import os

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from tools import TOOL_REGISTRY
from prompts import COORDINATOR_PROMPT, WORKER_PROMPT

load_dotenv()




class AgentInterface(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...
    
    @abstractmethod
    def next_assistant_message(self) -> Dict[str, Any]: ...




##################
# CoordinatorAgent
##################
class CoordinatorAgent(AgentInterface):
    def __init__(
        self,
        id: str = "Coordinator agent",
        model: str = "gpt-5",
        api_key: str | None = None,
        system_prompt: str = COORDINATOR_PROMPT,
    ):
        self._id = id
        if api_key is None: api_key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.openai_client = OpenAI(api_key=api_key)
        self.tools = self._build_tools()
        self.SYSTEM_PROMPT = system_prompt
        self.prompt = self._reset_prompt()

    def _reset_prompt(self) -> List[Dict[str, str]]:
        return [{
            "role": "system",
            "content": self.SYSTEM_PROMPT
        }]

    def _build_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for tool_name, tool in TOOL_REGISTRY.items():
            signature = inspect.signature(tool)
            properties = {name: {"type": "string"} for name in signature.parameters}
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (tool.__doc__ or "").strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(signature.parameters.keys()),
                        "additionalProperties": False
                    }
                }
            })
        return tools

    def _execute_llm_call(self, prompt: List[Dict[str, str]], tool_choice: str = "auto"):
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": prompt, # type: ignore
            "max_completion_tokens": 10000,
        }
        if self.tools is not None:
            params["tools"] = self.tools
            params["tool_choice"] = tool_choice
            params["parallel_tool_calls"] = False
        response = self.openai_client.chat.completions.create(**params)
        return response.choices[0].message

    def id(self) -> str: return self._id

    def next_assistant_message(self, tool_choice: str = "auto"):
        assistant_message = self._execute_llm_call(self.prompt, tool_choice = tool_choice)
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            self._format_prompt("assistant", assistant_message.content)
            return {"type": "final", "content": assistant_message.content}
        self._format_prompt("assistant", assistant_message.content, tool_calls)
        return {"type": "tools", "tool_calls": tool_calls}
    
    def _format_prompt(self, role, content, tool_call = None, tool_call_id = None):
        message = {
            "role": role,
            "content": content
        }
        if tool_call is not None: message["tool_calls"] = tool_call
        if tool_call_id is not None: message["tool_call_id"] = tool_call_id
        self.prompt.append(message)








##################
# WorkerAgent
##################
class ActionEvent(BaseModel):
    tool_name: str
    motivation: str


class WorkerAgent(AgentInterface):
    def __init__(
        self,
        id: str = "Worker agent",
        model: str = "gpt-5",
        api_key: str | None = None,
        system_prompt: str = WORKER_PROMPT,
    ):
        self._id = id
        if api_key is None: api_key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.openai_client = OpenAI(api_key=api_key)
        self.SYSTEM_PROMPT = system_prompt
        self.prompt = self._reset_prompt()
        self.tools = self._build_tools()

    def _reset_prompt(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def _execute_llm_call(self, prompt: List[Dict[str, str]]) -> ActionEvent:
        prompt = prompt + [{"role": "system", "content": self.tools}]
        response = self.openai_client.responses.parse(
            model=self.model,
            input=prompt, # type: ignore
            text_format=ActionEvent,
        )
        return response.output_parsed # type: ignore
    
    def _build_tools(self) -> str:
        tools = []
        for tool_name, tool in TOOL_REGISTRY.items():
            tool_description = (tool.__doc__ or "").strip()
            tools.append(f"- {tool_name}: {tool_description}" if tool_description else f"- {tool_name}")
        return "Available tools:\n" + "\n".join(tools)

    def _format_prompt(self, role, input):
        self.prompt.append({
            "role": role,
            "content": "" if input is None else input.strip()
        })

    def id(self) -> str: return self._id

    def next_assistant_message(self) -> Dict[str, Any]:
        action = self._execute_llm_call(self.prompt)
        self._format_prompt("assistant", json.dumps(action.model_dump()))
        return {"type": "action", **action.model_dump()}
