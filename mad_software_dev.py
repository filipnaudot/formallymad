import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pyfiglet import Figlet

from qbaf import QBAFramework

from formallymad.agent import CoordinatorAgent, WorkerAgent
from formallymad.tools import TOOL_REGISTRY, SKIP_TOOL_NAME




USER_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"
GRAY = "\u001b[38;5;245m"

def main() -> None:
    print(Figlet(font="big").renderText("Formally MAD"))

    workers = [WorkerAgent(id="A1", strength=0.1, extra_prompt="Please do not use any tool to list files."),
               WorkerAgent(id="A2", strength=0.9, extra_prompt="Read as few file as possible, this is expensive. At most one ot two."), 
               WorkerAgent(id="A3", strength=0.2)]
    coordinator = CoordinatorAgent()

    while True:
        try:
            user_input = input(f"\n{USER_COLOR}>{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            break

        for agent in workers: agent._format_prompt("user", user_input)

        while True:
            tool_proposals = []
            with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                futures = [(agent, pool.submit(agent.next_assistant_message)) for agent in workers]
                for agent, future in futures:
                    step = future.result()
                    tool_name = step["tool_name"]
                    motivation = step["motivation"]
                    tool_proposals.append((agent, tool_name, motivation))
                    print(f"{GRAY}{agent.id()} proposed tool: {tool_name}{RESET_COLOR}")
                    print(f"{GRAY}{agent.id()} Motivation: {motivation}{RESET_COLOR}\n\n")

            tool_name =_QBAF(workers, tool_proposals, VISUALIZE=True)
            # tool_name = _majority_vote(tool_proposals)
            if tool_name == SKIP_TOOL_NAME:
                coordinator._format_prompt("user", f"[START USER INPUT] User input: {user_input} [END USER INPUT]\n [START INFORMATION] The worker agents recommend to not call a tool.[END INFORMATION]")
                assistant_text = coordinator.next_assistant_message(tool_choice="none")["content"] or ""
                print(f"\n{ASSISTANT_COLOR}{assistant_text}{RESET_COLOR}")
                break
            
            coordinator._format_prompt("user", f"[START USER INPUT] {user_input} [END USER INPUT]\n [START INFORMATION] The worker agents have chosen tool: {tool_name} . Call tool {tool_name} with parameters of your choice.[END INFORMATION]")
            coordinator_step = coordinator.next_assistant_message(tool_choice="required")
            if coordinator_step["type"] == "tools":
                _handle_tool_call(coordinator, workers, coordinator_step["tool_calls"])


def _handle_tool_call(coordinator_agent: CoordinatorAgent, workers: list[WorkerAgent], tool_calls):
    for tool_call in tool_calls:
        tool_name = tool_call.function.name # type: ignore
        tool_call_args = json.loads(tool_call.function.arguments or "{}") # type: ignore
        tool = TOOL_REGISTRY[tool_name]
        tool_call_result = tool(**tool_call_args)
        for agent in workers:
            agent._format_prompt("user", f"[START TOOL RESULT] tool={tool_name} args={json.dumps(tool_call_args)} result={json.dumps(tool_call_result)} [END TOOL RESULT]",)
        coordinator_agent._format_prompt("tool", json.dumps(tool_call_result), tool_call_id=tool_call.id)


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


def _QBAF(agents: list[WorkerAgent], tool_proposals: list[tuple[WorkerAgent, str, str]], *, VISUALIZE = False):
    print(f"{GRAY}Building QBAF{RESET_COLOR}")
    agent_args = [agent.id() for agent in agents]
    tool_args = [tool_name for _, tool_name, _ in tool_proposals]
    args = list(dict.fromkeys(agent_args + tool_args))
    print(f"ARGS: {args}")
    strengths_by_arg = {agent.id(): agent.strength() for agent in agents}
    strengths_by_arg.update({tool_name: 0.5 for _, tool_name, _ in tool_proposals})
    initial_strengths = [strengths_by_arg[arg] for arg in args]
    tool_by_agent_id = {agent.id(): tool_name for agent, tool_name, _ in tool_proposals}
    atts = []; supps = []
    first_agent_for_tool: set[str] = set()
    for index, agent in enumerate(agents):
        agent_tool = tool_by_agent_id.get(agent.id())
        if agent_tool is None: continue
        if agent_tool not in first_agent_for_tool:
            supps.append((agent.id(), agent_tool))
            first_agent_for_tool.add(agent_tool)        
        found_attack = found_support = False
        for prior in reversed(agents[:index]):
            prior_tool = tool_by_agent_id.get(prior.id())
            if prior_tool is None: continue
            if prior_tool == agent_tool and not found_support:
                supps.append((agent.id(), prior.id()))
                found_support = True
            elif prior_tool != agent_tool and not found_attack:
                atts.append((agent.id(), prior.id()))
                found_attack = True
    qbaf = QBAFramework(args, initial_strengths, atts, supps, semantics="QuadraticEnergy_model")
    if VISUALIZE:
        # Optional visualization deps live in the `visualize`` extra.
        from qbaf_visualizer.Visualizer import visualize
        import matplotlib.pyplot as plt
        visualize(qbaf, with_fs=True, round_to=3)
        plt.savefig("qbaf.png", dpi=300, bbox_inches="tight")
        plt.close()
    tool_final_strengths = {tool: round(qbaf.final_strengths[tool], 2) for tool in tool_args if tool in qbaf.final_strengths}
    max_tool, max_strength = max(tool_final_strengths.items(), key=lambda tool_and_strength: tool_and_strength[1])
    return max_tool












if __name__ == "__main__":
    main()
