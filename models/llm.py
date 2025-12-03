import os
import time
import json
from openai import OpenAI

TOOL_NAME = "t2sql_tool"
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Process a natural language request and return an appropriate response in one of three ways:\n\n"
            "- If the request is clear and answerable, return a SQL SELECT query, a short user-facing message, and a concise title.\n"
            "- If the request is ambiguous (i.e., missing context or has multiple interpretations), ask a clarifying follow-up question, but do not guess.\n"
            "- If the request is an improper phrase (i.e., thank you), answer properly in a short way.\n"
            "- If the request cannot be answered even with clarification (e.g., no possible mapping to available data), explain why.\n\n"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["sql", "ambiguous", "unanswerable", "improper"],
                    "nullable": False,
                    "description": "Classify the request outcome."
                },
                "sql": {
                    "type": "string",
                    "nullable": True,
                    "description": "SQL SELECT statement matching the user's intent. Required if type is 'sql'."
                },
                "message": {
                    "type": "string",
                    "nullable": True,
                    "description": (
                        "User-facing message.\n"
                        "- For 'sql': Leave message empty.\n"
                        "- For 'ambiguous': Ask a clarifying question.\n"
                        "- For 'unanswerable': Explain why the request cannot be answered.\n"
                        "- For 'improper': Just give a short fitting answer.\n"
                    )
                }
            }
        }
    }
}

class LLM:

    def __init__(self, provider:str = "openai", model:str = "gpt-5"):
        self.provider = provider
        self.model = model

        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                organization=os.getenv('OPENAI_API_ORGANIZATION'),
                project=os.getenv('OPENAI_API_PROJECT'),
            )
        elif self.provider == "google":
            self.client = OpenAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        elif self.provider == "together":
            self.client = OpenAI(
                api_key=os.getenv("TOGETHERAI_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )
    
    # sending request to llm and receiving response
    def ask(self, messages):

        tool_to_use = TOOL

        start_time = time.perf_counter() # start timer

        chat_kwargs = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "tools": [tool_to_use],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}}
        }

        # only OpenAI supports reasoning_effort
        if self.provider == "openai":
            chat_kwargs["reasoning_effort"] = "minimal"

        response = self.client.chat.completions.create(**chat_kwargs)

        end_time = time.perf_counter()  # end timer
        duration_seconds = end_time - start_time

        message = response.choices[0].message
        tool_call = message.tool_calls[0] if message.tool_calls else None
        
        tool_output = {}
        try:
            arguments = json.loads(tool_call.function.arguments)
            tool_output = {
                "type": arguments.get("type"),
                "sql": arguments.get("sql"),
                "message": arguments.get("message")
            }
        except Exception as e:
            print("Exception when deconstructing response")
            print(str(tool_call))
            tool_output = {
                "type": None,
                "sql": None,
                "message": None,
            }      
        
        return {
            "response": tool_output,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "duration_seconds": duration_seconds,
        }
