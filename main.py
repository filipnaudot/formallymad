import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pyfiglet import Figlet

from agent import Agent
from tools import TOOL_REGISTRY
from prompts import WORKER_PROMPT, COORDINATOR_PROMPT, COORDINATOR_TIE_PROMPT, COORDINATOR_TIE_ERROR_PROMPT

USER_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"
GRAY = "\u001b[38;5;245m"

def main() -> None:
    print(Figlet(font="big").renderText("Formally MAD"))

    workers = [Agent(system_prompt=WORKER_PROMPT),
               Agent(system_prompt=WORKER_PROMPT),
               Agent(system_prompt=WORKER_PROMPT)]
    coordinator = Agent(system_prompt=COORDINATOR_PROMPT, coordinator=True)

    while True:
        try:
            user_input = input(f"\n{USER_COLOR}>{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            break

        for agent in workers: agent._format_prompt("user", user_input)
        coordinator._format_prompt("user", f"The user asked: {user_input}.\n This will now be passed to the workers.")

        assistant_text = ""
        proposals = []
        extra_calls = []
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = [(agent, pool.submit(agent.next_assistant_message)) for agent in workers]
            for agent, future in futures:
                step = future.result()
                if step["type"] == "final": continue
                
                tool_calls = step["tool_calls"]
                if not tool_calls: continue
                
                call = tool_calls[0]
                name = call.function.name # type: ignore
                args = json.loads(call.function.arguments or "{}") # type: ignore
                proposals.append((agent, call, name, args))
                print(f"{GRAY}Agent proposed tool: {name} args={args}{RESET_COLOR}")
                if len(tool_calls) > 1:
                    extra_calls.append((agent, tool_calls[1:]))
        

        ### THIS IS A TRMPORARY MAJORITY VOTE ###
        # TODO: This should be done using a QBAF
        tool_call_key, tool_name, tool_call_args = _majority_vote(proposals, coordinator)
        
        tool_call_result = _tool_call(tool_call_args, tool_name)
        print(f"{GRAY}Executed winner tool: {tool_name}{RESET_COLOR}")

        for agent, call, name, args in proposals:
            if name == tool_call_key:
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
        _append_coordinator_tool_result(coordinator, tool_name, tool_call_args, tool_call_result)
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

        coordinator_step = coordinator.next_assistant_message()
        assistant_text = coordinator_step["content"] or ""
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


def _majority_vote(proposals, coordinator_agent):
    counts = {}
    ordered = []
    for _, _, name, args in proposals:
        key = name
        counts[key] = counts.get(key, 0) + 1
        if key not in ordered: ordered.append(key)
    winner_key = max(ordered, key=lambda k: counts[k])
    winner_name = winner_key
    winner_args = _majority_vote_params(proposals, winner_name, coordinator_agent)
    print(f"{GRAY}Majority vote winner: {winner_name} args={winner_args}{RESET_COLOR}")
    return winner_key, winner_name, winner_args


def _majority_vote_params(proposals, winner_name, coordinator_agent):
    candidates = [args for _, _, name, args in proposals if name == winner_name]
    if not candidates: return {}
    
    all_params = set()
    for args in candidates: all_params.update(args.keys())
    
    merged = {}
    for param in sorted(all_params):
        values = []
        for args in candidates:
            if param in args:
                values.append(args[param])
        if not values: continue
        
        counts = {}
        ordered = []
        for val in values:
            key = json.dumps(val, sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
            if key not in ordered:
                ordered.append(key)
        max_count = max(counts.values())
        winners = [k for k in ordered if counts[k] == max_count]
        if len(winners) > 1:
            chosen = _ask_coordinator_param_tie(coordinator_agent, param, [json.loads(k) for k in winners])
        else:
            chosen = json.loads(winners[0])
        merged[param] = chosen
    return merged


def _ask_coordinator_param_tie(coordinator_agent, param, options):
    prompt = (
        f"{COORDINATOR_TIE_PROMPT.strip()}\n"
        f"Parameter: {param}\n"
        f"Options (0-indexed): {json.dumps(options)}"
    )
    for _ in range(3):
        coordinator_agent._format_prompt("user", prompt)
        step = coordinator_agent.next_assistant_message()
        if step["type"] == "final":
            try:
                data = json.loads(step["content"] or "{}")
                idx = int(data.get("choice"))
                if 0 <= idx < len(options):
                    return options[idx]
            except (ValueError, json.JSONDecodeError, TypeError):
                pass
        prompt = (
            f"{COORDINATOR_TIE_ERROR_PROMPT.strip()}\n"
            f"Parameter: {param}\n"
            f"Options (0-indexed): {json.dumps(options)}"
        )
    return options[0]


def _append_coordinator_tool_result(coordinator_agent, tool_name, tool_args, tool_result):
    message = (
        "Tool executed by workers in response to the users query.\n"
        f"Tool: {tool_name}\n"
        f"Args: {json.dumps(tool_args, sort_keys=True)}\n"
        f"Result: {json.dumps(tool_result)}"
    )
    coordinator_agent._format_prompt("user", message)


if __name__ == "__main__":
    main()
