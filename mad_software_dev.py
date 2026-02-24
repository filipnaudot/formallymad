import json
from concurrent.futures import ThreadPoolExecutor

from formallymad.agent import CoordinatorAgent, WorkerAgent
from formallymad.qbaf import QBAFResolver
from formallymad.tools import TOOL_REGISTRY, SKIP_TOOL_NAME
from formallymad.ui import FormallyMADUI


def main() -> None:
    ui = FormallyMADUI()
    ui.banner()

    workers = [WorkerAgent(id="A1", strength=0.1, extra_prompt="Please do not use any tool to list files."),
               WorkerAgent(id="A2", strength=0.5, extra_prompt="Read as few file as possible, this is expensive. At most one ot two."),
               WorkerAgent(id="A3", strength=0.2),
               WorkerAgent(id="A4", strength=0.2),
               WorkerAgent(id="A5", strength=0.2),
               WorkerAgent(id="A6", strength=0.2),
            ]
    coordinator = CoordinatorAgent()
    qbaf = QBAFResolver(workers)

    while True:
        try:
            user_input = ui.ask()
        except (KeyboardInterrupt, EOFError):
            break

        for agent in workers: agent._format_prompt("user", user_input)
        while True:
            with ui.loading("Collecting worker proposals..."):
                tool_proposals = []
                with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                    futures = [(agent, pool.submit(agent.next_assistant_message)) for agent in workers]
                    for agent, future in futures:
                        step = future.result()
                        tool_proposals.append((agent, step["tool_name"], step["motivation"]))
                tool_name, _ = qbaf.resolve(tool_proposals)
                

            ui.show_proposals((agent.id(), tool, motivation) for agent, tool, motivation in tool_proposals)
            ui.show_agent_metrics(qbaf.last_agent_stats)
            # tool_name = _majority_vote(tool_proposals)
            if tool_name == SKIP_TOOL_NAME:
                coordinator._format_prompt("user", f"[START USER INPUT] User input: {user_input} [END USER INPUT]\n [START INFORMATION] The worker agents recommend to not call a tool.[END INFORMATION]")
                with ui.loading("Generating assistant response..."):
                    assistant_text = coordinator.next_assistant_message(tool_choice="none")["content"] or ""
                ui.show_assistant(assistant_text)
                break

            coordinator._format_prompt("user", f"[START USER INPUT] {user_input} [END USER INPUT]\n [START INFORMATION] The worker agents have chosen tool: {tool_name} . Call tool {tool_name} with parameters of your choice.[END INFORMATION]")
            ui.show_selected_tool(tool_name)
            with ui.loading("Coordinator is preparing tool call..."):
                coordinator_step = coordinator.next_assistant_message(tool_choice="required")
            if coordinator_step["type"] == "tools":
                _handle_tool_call(coordinator, workers, coordinator_step["tool_calls"], ui)


def _handle_tool_call(coordinator_agent: CoordinatorAgent, workers: list[WorkerAgent], tool_calls: list, ui: FormallyMADUI) -> None:
    """Execute tool calls and feed results back to all agents."""
    for tool_call in tool_calls:
        tool_name = tool_call.function.name # type: ignore
        tool_call_args = json.loads(tool_call.function.arguments or "{}") # type: ignore
        tool = TOOL_REGISTRY[tool_name]
        with ui.loading(f"Executing tool: {tool_name}"):
            tool_call_result = tool(**tool_call_args)
        for agent in workers:
            agent._format_prompt("user", f"[START TOOL RESULT] tool={tool_name} args={json.dumps(tool_call_args)} result={json.dumps(tool_call_result)} [END TOOL RESULT]",)
        coordinator_agent._format_prompt("tool", json.dumps(tool_call_result), tool_call_id=tool_call.id)
        ui.show_tool_result(tool_name, tool_call_args, tool_call_result)


def _majority_vote_resolver(tool_proposals: list[tuple[WorkerAgent, str, str]]) -> str:
    """Pick the tool name that got the most votes."""
    counts = {}
    ordered = []
    for agent, tool_name, motivation in tool_proposals:
        counts[tool_name] = counts.get(tool_name, 0) + 1
        if tool_name not in ordered: ordered.append(tool_name)
    winner_key = max(ordered, key=lambda name: counts[name])
    winner_name = winner_key
    return winner_name


if __name__ == "__main__":
    main()
