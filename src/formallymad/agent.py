import inspect
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from formallymad.prompts import RECOMMENDER_PROMPT
from formallymad.tools import TOOL_REGISTRY

load_dotenv()


class Recommendation(BaseModel):
    recommendation: str
    motivation: str


class Agent:
    def __init__(self,
                 id: str,
                 system_prompt: str = RECOMMENDER_PROMPT,
                 role: str | None = None,
                 model: str = "gpt-5",
                 api_key: str | None = None,
                 strength: float = 0.5) -> None:
        self._id = id
        self.strength = strength
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"], max_retries=10)
        self.system_prompt = (f"YOUR ROLE: {role}\n\n" if role else "") + system_prompt
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

    def update_strength(self, value: float) -> None: self.strength = value


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


    def synthesize(self, question: str, recommendations: list[tuple["Agent", Recommendation]]) -> Recommendation:
        """
        Synthesize all worker recommendations into a structured final Recommendation (oracle role).

        :param question: The original question posed to the worker agents.
        :param recommendations: List of (agent, recommendation) pairs from all workers.
        """
        aliases = {agent.id: f"Agent{i}" for i, (agent, _) in enumerate(recommendations, 1)}
        formatted_recommendations = "\n\n".join(f"[AGENT: {aliases[agent.id]}] RECOMMENDATION: {rec.recommendation}\nMOTIVATION: {rec.motivation}" for agent, rec in recommendations)
        messages = [{"role": "system", "content": self.system_prompt.format(question=question)},
                    {"role": "user", "content": formatted_recommendations}]
        response = self.client.responses.parse(model=self.model, input=messages, text_format=Recommendation) # type: ignore
        return response.output_parsed # type: ignore


    def synthesize_with_attribution(self, question: str, recommendations: list[tuple["Agent", Recommendation]], options: list[str] | None = None) -> tuple[Recommendation, dict[str, float]]:
        """
        Run llmSHAP attribution then produce a structured final Recommendation (oracle role).
        The question is embedded in the oracle system prompt so agent blocks are the sole attribution units.
        The structured final answer is obtained via a separate synthesize() call.
        Uses self.model for all LLM calls.

        :param question: The original question posed to the worker agents.
        :param recommendations: List of (agent, recommendation) pairs from all workers.
        :param options: Valid recommendation option strings for label extraction in the value function.
                        When provided, uses LabelWeightedSimilarity; otherwise falls back to TF-IDF.
        """
        from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
        from llmSHAP.llm import OpenAIInterface
        from formallymad.value_function import LabelWeightedSimilarity

        aliases = {agent.id: f"Agent{i}" for i, (agent, _) in enumerate(recommendations, 1)}
        agent_ids = [agent.id for agent, _ in recommendations]
        handler = DataHandler({agent.id: f"AGENT: {aliases[agent.id]}\nRECOMENDATION: {rec.recommendation}.\nMOTIVATION: {rec.motivation}" for agent, rec in recommendations})
        codec = BasicPromptCodec(system=self.system_prompt.format(question=question))
        llm_interface = OpenAIInterface(model_name=self.model)
        # value_function = LabelWeightedSimilarity(options, label_weight=0.8) if options is not None else None
        # result = ShapleyAttribution(model=llm_interface, data_handler=handler, prompt_codec=codec, value_function=value_function, use_cache=True, num_threads=len(recommendations)*5, verbose=False, logging=True).attribution()
        result = ShapleyAttribution(model=llm_interface, data_handler=handler, prompt_codec=codec, use_cache=True, num_threads=len(recommendations)*5, verbose=False).attribution()
        attribution_scores = {agent_id: result.attribution[agent_id]["score"] for agent_id in agent_ids}
        return self.synthesize(question, recommendations), attribution_scores
