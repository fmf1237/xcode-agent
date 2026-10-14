"""Interactive CLI: python main.py"""
from xcode_agent import Agent, AgentLLMError, ContextManager, LLMClient, MaxRoundsExceeded, ToolRegistry
from xcode_agent.config import Settings
from xcode_agent.file_tools import register_file_tools


def main() -> None:
    settings = Settings.from_env()
    llm = LLMClient(settings.api_key, settings.base_url, settings.model, settings.timeout, settings.retries)
    registry = ToolRegistry()
    register_file_tools(registry, settings.workspace)
    context = ContextManager(settings.history_limit)
    agent = Agent(llm, registry, context, settings.max_rounds, print if settings.verbose else None)
    print("xcode-agent ready. Type exit to quit or reset to clear the session.")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() == "exit":
            break
        if question.lower() == "reset":
            context.reset()
            print("Session reset.")
            continue
        if not question:
            continue
        try:
            print(f"Agent> {agent.run(question)}")
        except (AgentLLMError, MaxRoundsExceeded) as exc:
            print(f"Agent error: {exc}")


if __name__ == "__main__":
    main()
