import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pyfiglet import Figlet

from agent import CoordinatorAgent, WorkerAgent
from tools import TOOL_REGISTRY




USER_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"
GRAY = "\u001b[38;5;245m"

def main() -> None:
    print(Figlet(font="big").renderText("Formally MAD"))

    workers = [WorkerAgent(id="Agent 1"), WorkerAgent(id="Agent 2"), WorkerAgent(id="Agent 3")]
    coordinator = CoordinatorAgent()

    while True:
        try:
            user_input = input(f"\n{USER_COLOR}>{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            break

        for agent in workers: agent._format_prompt("user", user_input)

        tool_proposals = []
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [(agent, pool.submit(agent.next_assistant_message)) for agent in workers]
            for agent, future in futures:
                step = future.result()
                tool_name = step["tool_name"]
                motivation = step["motivation"]
                tool_proposals.append((agent, tool_name, motivation))
                print(f"{GRAY}Agent {agent.id} proposed tool: {tool_name}{RESET_COLOR}")
        
        # TODO: This should be done using a QBAF
        tool_name = _majority_vote(tool_proposals)
        if tool_name != "skip":    
            coordinator._format_prompt("user", f"User input: {user_input}.\n The worker agents recommend tool: {tool_name}.")
            coordinator._format_prompt("user", f"Call tool {tool_name} with parameters of your choice.")
        else:
            coordinator._format_prompt("user", f"User input: {user_input}.\n The worker agents recommend to not call a tool.")
        coordinator_step = coordinator.next_assistant_message()
        while coordinator_step["type"] == "tools":
            coordinator_step = _handle_tool_call(coordinator, coordinator_step)
        assistant_text = coordinator_step["content"] or ""
        print(f"\n{ASSISTANT_COLOR}{assistant_text}{RESET_COLOR}")


def _handle_tool_call(agent, agent_tool_call):
    tool_call = agent_tool_call["tool_calls"][0]
    tool_name = tool_call.function.name # type: ignore
    tool_call_args = json.loads(tool_call.function.arguments or "{}") # type: ignore
    tool = TOOL_REGISTRY[tool_name]
    tool_call_result = tool(**tool_call_args)
    agent.prompt.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_call_result)
    })
    agent_result = agent.next_assistant_message()
    return agent_result


def _majority_vote(tool_proposals):
    counts = {}
    ordered = []
    for agent, tool_name, motivation in tool_proposals:
        counts[tool_name] = counts.get(tool_name, 0) + 1
        if tool_name not in ordered: ordered.append(tool_name)
    winner_key = max(ordered, key=lambda name: counts[name])
    winner_name = winner_key
    print(f"{GRAY}Majority vote winner: {winner_name}{RESET_COLOR}")
    return winner_name


if __name__ == "__main__":
    main()
