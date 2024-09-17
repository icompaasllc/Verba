import os
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from goldenverba.components.types import InputConfig
from goldenverba.components.util import get_environment
import httpx
import json

load_dotenv()


class GroqGenerator(Generator):
    """
    Groq Generator.
    """

    def __init__(self):
        super().__init__()
        self.name = "Groq"
        self.description = "Using Groq LLM models to generate answers to queries"
        self.context_window = 10000

        models = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

        self.config["Model"] = InputConfig(
            type="dropdown",
            value=models[0],
            description="Select an Groq Embedding Model",
            values=models,
        )

        if os.getenv("GROQ_API_KEY") is None:
            self.config["API Key"] = InputConfig(
                type="password",
                value="",
                description="You can set your Groq API Key here or set it as environment variable `GROQ_API_KEY`",
                values=[],
            )
        if os.getenv("GROQ_BASE_URL") is None:
            self.config["URL"] = InputConfig(
                type="text",
                value="https://api.groq.com/openai/v1",
                description="You can change the Base URL here if needed",
                values=[],
            )

    async def generate_stream(
        self,
        config: dict,
        query: str,
        context: str,
        conversation: list[dict] = [],
    ):
        system_message = config.get("System Message").value
        model = config.get("Model", {"value": "llama-3.1-70b-versatile"}).value
        openai_key = get_environment(
            config, "API Key", "GROQ_API_KEY", "No Groq API Key found"
        )
        openai_url = get_environment(
            config, "URL", "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )

        messages = self.prepare_messages(query, context, conversation, system_message)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }
        data = {
            "messages": messages,
            "model": model,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{openai_url}/chat/completions",
                json=data,
                headers=headers,
                timeout=None,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        json_line = json.loads(line[6:])
                        choice = json_line["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            yield {
                                "message": choice["delta"]["content"],
                                "finish_reason": choice.get("finish_reason"),
                            }
                        elif "finish_reason" in choice:
                            yield {
                                "message": "",
                                "finish_reason": choice["finish_reason"],
                            }

    def prepare_messages(
        self, query: str, context: str, conversation: list[dict], system_message: str
    ) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]

        for message in conversation:
            messages.append({"role": message.type, "content": message.content})

        messages.append(
            {
                "role": "user",
                "content": f"Answer this query: '{query}' with this provided context: {context}",
            }
        )

        return messages
