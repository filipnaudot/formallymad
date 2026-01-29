import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pyfiglet import Figlet

from agent import Agent
from tools import TOOL_REGISTRY

USER_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"
GRAY = "\u001b[38;5;245m"

def main() -> None:
    print(Figlet(font="big").renderText("Formally MAD"))

    agents = [Agent(), Agent(), Agent()]

    while True:
        try:
            user_input = input(f"\n{USER_COLOR}>{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            break

        for agent in agents:
            agent._format_prompt("user", user_input)

        assistant_text = ""
        while True:
            proposals = []
            extra_calls = []
            finals = 0
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                futures = [(agent, executor.submit(agent.next_assistant_message)) for agent in agents]
                for agent, future in futures:
                    step = future.result()
                    if step["type"] == "final":
                        finals += 1
                        assistant_text = step["content"] or ""
                        continue
                    tool_calls = step["tool_calls"]
                    if not tool_calls:
                        continue
                    call = tool_calls[0]
                    name = call.function.name # type: ignore
                    args = json.loads(call.function.arguments or "{}") # type: ignore
                    proposals.append((agent, call, name, args))
                    print(f"{GRAY}Agent proposed tool: {name} args={args}{RESET_COLOR}")
                    if len(tool_calls) > 1:
                        extra_calls.append((agent, tool_calls[1:]))

            if finals == len(agents): break
            if not proposals: continue


            ### THIS IS A TRMPORARY MAJORITY VOTE ###
            # TODO: This should be done using a QBAF
            tool_call_key, tool_name, tool_call_args = _majority_vote(proposals)
            ### END OF MAJORITY VOTE ###
            tool_call_result = _tool_call(tool_call_args, tool_name)
            print(f"{GRAY}Executed winner tool: {tool_name}{RESET_COLOR}")

            for agent, call, name, args in proposals:
                if (name, json.dumps(args, sort_keys=True)) == tool_call_key:
                    content = json.dumps(tool_call_result)
                else:
                    content = json.dumps({
                        "skipped": True,
                        "reason": "majority_vote",
                        "winner": {"name": tool_name, "args": tool_call_args},
                    })
                agent.prompt.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content
                })
            for agent, calls in extra_calls:
                for extra_call in calls:
                    agent.prompt.append({
                        "role": "tool",
                        "tool_call_id": extra_call.id,
                        "content": json.dumps({
                            "skipped": True,
                            "reason": "ignored_non_primary_tool_call",
                            "winner": {"name": tool_name, "args": tool_call_args},
                        })
                    })

        print(f"\n{ASSISTANT_COLOR}{assistant_text}{RESET_COLOR}")


def _tool_call(winner_args, tool_name):
    tool = TOOL_REGISTRY[tool_name]
    signature = inspect.signature(tool)
    kwargs = {
        param: winner_args.get(param)
        for param in signature.parameters
        if param in winner_args
    }
    winner_result = tool(**kwargs)
    return winner_result


def _majority_vote(proposals):
    counts = {}
    ordered = []
    for _, _, name, args in proposals:
        key = (name, json.dumps(args, sort_keys=True))
        counts[key] = counts.get(key, 0) + 1
        if key not in ordered:
            ordered.append(key)
    winner_key = max(ordered, key=lambda k: counts[k])
    winner_name, winner_args_json = winner_key
    winner_args = json.loads(winner_args_json)
    print(f"{GRAY}Majority vote winner: {winner_name} args={winner_args}{RESET_COLOR}")
    return winner_key,winner_name,winner_args


if __name__ == "__main__":
    main()
