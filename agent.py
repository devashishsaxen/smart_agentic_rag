import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx

from config import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    MAX_HISTORY_MESSAGES,
    MAX_TOKENS,
    MEMORY_FILE,
)
from retriever import NO_CONTEXT_RESPONSE, RETRIEVER_TOOL, get_known_source_terms, retrieve


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a knowledge base. Decide whether "
    "the user's question needs document retrieval. If the question is about "
    "ingested documents or may depend on them, use the retrieve_context tool "
    "first. If the question is general knowledge, small talk, or simple "
    "reasoning, answer directly. If retrieval is used and the retrieved context "
    "does not answer the question, say that clearly instead of pretending to try again. "
    "When you answer from retrieved context, cite the supporting source labels exactly "
    "as they appear, including file and page references."
)

DOCUMENT_HINTS = {
    "document",
    "pdf",
    "policy",
    "act",
    "section",
    "clause",
    "complaint",
    "complaints",
    "committee",
    "report",
    "employer",
    "district officer",
    "internal committee",
    "posh",
    "novatech",
    "flowai",
}

GENERAL_KNOWLEDGE_HINTS = {
    "capital",
    "state",
    "ut",
    "union territory",
    "math",
    "weather",
    "population",
    "president",
    "prime minister",
}


class Agent:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is missing. Add it to a .env file or your environment."
            )

        self.api_key = api_key
        self.base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")
        self.model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
        self.memory_file = Path(MEMORY_FILE)
        self.history = self._load_memory()

    @staticmethod
    def _tool_definition():
        return {
            "type": "function",
            "function": RETRIEVER_TOOL,
        }

    def _messages(self, history=None):
        active_history = self.history if history is None else history
        return [{"role": "system", "content": SYSTEM_PROMPT}, *active_history]

    def _load_memory(self):
        if not self.memory_file.exists():
            return []

        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            return []

        history = data.get("history", [])
        if not isinstance(history, list):
            return []
        return history[-MAX_HISTORY_MESSAGES:]

    def _save_memory(self):
        payload = {"history": self.history[-MAX_HISTORY_MESSAGES:]}
        self.memory_file.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _trim_history(self):
        self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def clear_memory(self):
        self.history = []
        if self.memory_file.exists():
            self.memory_file.unlink()

    def _create_message(
        self,
        history=None,
        use_tools: bool = True,
        force_retrieval: bool = False,
    ):
        payload = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": self._messages(history),
            "temperature": 0,
        }
        if use_tools:
            payload["tools"] = [self._tool_definition()]
            if force_retrieval:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "retrieve_context"},
                }
            else:
                payload["tool_choice"] = "auto"

        last_error = None
        for attempt in range(3):
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()

            last_error = response
            wait_seconds = attempt + 1
            print(f"[agent] NVIDIA rate limit hit. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

        last_error.raise_for_status()

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _contains_hint(normalized_text: str, hints: set[str]) -> bool:
        token_set = set(normalized_text.split())
        for hint in hints:
            if " " in hint:
                if hint in normalized_text:
                    return True
            elif hint in token_set:
                return True
        return False

    def _looks_document_related(self, user_message: str) -> bool:
        normalized = self._normalize(user_message)
        if self._contains_hint(normalized, DOCUMENT_HINTS):
            return True

        source_terms = get_known_source_terms()
        return self._contains_hint(normalized, source_terms)

    def _looks_general_knowledge(self, user_message: str) -> bool:
        normalized = self._normalize(user_message)
        if self._looks_document_related(normalized):
            return False
        return self._contains_hint(normalized, GENERAL_KNOWLEDGE_HINTS)

    @staticmethod
    def _no_context_answer() -> str:
        return (
            "I could not find enough support for that answer in the ingested documents. "
            "Try asking with the document name, section, topic, or a more specific phrase."
        )

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        force_retrieval = self._looks_document_related(user_message)
        use_tools = not self._looks_general_knowledge(user_message)

        if force_retrieval:
            print("[agent] Routing decision: document-style question, forcing retrieval.")
        elif not use_tools:
            print("[agent] Routing decision: general question, answering directly.")

        response = self._create_message(
            history=self.history,
            use_tools=use_tools,
            force_retrieval=force_retrieval,
        )
        message = response["choices"][0]["message"]

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            follow_up_history = self.history + [
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": tool_calls,
                }
            ]
            saw_context = False

            for tool_call in tool_calls:
                function_data = tool_call.get("function", {})
                tool_name = function_data.get("name", "")
                if tool_name != "retrieve_context":
                    print(f"[agent] Model requested unsupported tool: {tool_name}")
                    result = f"Unknown tool requested: {tool_name}"
                else:
                    try:
                        arguments = json.loads(function_data.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    query = arguments.get("query", user_message)
                    print(f"[agent] Tool call triggered: {tool_name}")
                    result = retrieve(query, verbose=True)
                    if result != NO_CONTEXT_RESPONSE:
                        saw_context = True

                follow_up_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": result,
                    }
                )

            if not saw_context:
                final_text = self._no_context_answer()
                self.history.append({"role": "assistant", "content": final_text})
                self._trim_history()
                self._save_memory()
                return final_text

            response = self._create_message(history=follow_up_history, use_tools=False)
            message = response["choices"][0]["message"]

        final_text = (message.get("content") or "").strip()
        self.history.append({"role": "assistant", "content": final_text})
        self._trim_history()
        self._save_memory()
        return final_text or "I could not generate a response."
