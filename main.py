from pathlib import Path

from agent import Agent
from config import MEMORY_FILE
from ingest import ingest_document, resolve_input_path


def run() -> int:
    print("Agentic RAG Assistant")
    print("Type 'exit' to quit | Type 'ingest <filepath>' to add documents")
    print("Type 'clear memory' to reset saved chat history")
    print("----------------------------------------")

    agent = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye.")
                return 0

            if user_input.lower() == "clear memory":
                if agent is None:
                    memory_file = Path(MEMORY_FILE)
                    if memory_file.exists():
                        memory_file.unlink()
                else:
                    agent.clear_memory()
                print("Saved chat memory cleared.")
                continue

            if user_input.startswith("ingest "):
                file_path = user_input.removeprefix("ingest ").strip()
                try:
                    count = ingest_document(file_path)
                    resolved_name = resolve_input_path(file_path).name
                    print(f"Created and stored {count} chunks from {resolved_name}.")
                except Exception as exc:
                    print(f"Ingestion failed: {exc}")
                continue

            print("[thinking...]")
            if agent is None:
                agent = Agent()
            print(f"Assistant: {agent.chat(user_input)}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            return 0
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(run())
