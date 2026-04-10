import inspect
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from formallymad.tools import TOOL_REGISTRY

load_dotenv()


class Recommendation(BaseModel):
    recommendation: str
    motivation: str


class Agent:
    def __init__(self,
                 id: str,
                 system_prompt: str,
                 extra_prompt: str | None = None,
                 model: str = "gpt-5",
                 api_key: str | None = None,
                 strength: float = 0.5) -> None:
        self._id = id
        self.strength = strength
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"], max_retries=10)
        self.system_prompt = system_prompt + (f"\n\nADDITIONAL INSTRUCTIONS:\n{extra_prompt}" if extra_prompt else "")
        self._tools = self._build_tools()


    def _build_tools(self) -> list[dict]:
        """
        Build OpenAI-format tool definitions from TOOL_REGISTRY using inspect for parameter types.
        """
        _PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}
        tools = []
        for tool_name, tool in TOOL_REGISTRY.items():
            sig = inspect.signature(tool)
            properties = {name: {"type": _PY_TO_JSON.get(param.annotation, "string")} for name, param in sig.parameters.items()}
            tools.append({"type": "function",
                          "name": tool_name,
                          "description": (tool.__doc__ or "").strip(),
                          "parameters": {"type": "object",
                                         "properties": properties,
                                         "required": list(sig.parameters.keys()),
                                         "additionalProperties": False}})
        return tools


    @property
    def id(self) -> str: return self._id


    def recommend(self, query: str) -> Recommendation:
        """
        Query the model, execute any requested tool calls, and return a final structured recommendation.
        Loops until the model stops calling tools.

        :param query: The user query or task to recommend an action for.
        """
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": query}]
        while True:
            response = self.client.responses.parse(model=self.model, input=messages, tools=self._tools, text_format=Recommendation) # type: ignore
            if response.output_parsed is not None:
                return response.output_parsed
            tool_calls = [item for item in response.output if item.type == "function_call"]
            messages += [{"type": "function_call", "call_id": tool_call.call_id, "name": tool_call.name, "arguments": tool_call.arguments} for tool_call in tool_calls]
            for call in tool_calls:
                result = TOOL_REGISTRY[call.name](**json.loads(call.arguments))
                messages.append({"type": "function_call_output",
                                 "call_id": call.call_id,
                                 "output": json.dumps(result)})


    def synthesize(self, query: str, recommendations: list[tuple["Agent", Recommendation]]) -> str:
        """
        Synthesize all worker recommendations into a single final answer (oracle role).

        :param query: The original user query.
        :param recommendations: List of (agent, recommendation) pairs from all workers.
        """
        formatted_recommendations = "\n\n".join(f"[AGENT: {agent.id}] RECOMENDATION: {rec.recommendation}\nMOTIVATION: {rec.motivation}" for agent, rec in recommendations)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": f"Query: {query}\n\nRecommendations:\n{formatted_recommendations}"},],
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content or ""
