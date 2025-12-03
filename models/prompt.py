import os

from models.llm import LLM


INIT_INSTRUCTION = (
    "You are a helpful assistant that processes natural language requests by returning SQL queries, "
    "asking clarifying questions, or explaining why a question can't be answered. Always use the provided "
    "function tool to respond. Do not reply directly. Your response must include:\n"
    "- `type`: One of 'sql', 'improper', 'ambiguous', or 'unanswerable'\n"
    "- `sql`: Required only when type is 'sql'\n"
    "- `message`: A friendly explanation of the results to the question or follow-up question\n"
    "Only refer to tables and fields defined in the schema. Do not guess. "
    "If the question is ambiguous, ask a specific, guiding follow-up."
    "Always generate a message. Only generate SQL when you're confident about the user's intent. "
    "Do not return any answer unless using the function tool.\n"
)

class Prompter:

    def __init__(self, provider:str = "openai", model:str = "gpt-5", schema_string:str = None):
        self.provider = provider
        self.model = model
        self.llm = LLM(provider=self.provider, model=self.model)

        if schema_string:
            self.schema_string = schema_string
        else:
            raise ValueError("Schema string must not be empty!")
        
    def ask_question(self, question):

        messages = self._build_messages(question)
        response = self.llm.ask(messages=messages)

        return response


    def _build_messages(self, question):
        
        messages = [
            { "role": "system", "content": INIT_INSTRUCTION },
            { "role": "system", "content": self.schema_string },
            { "role": "user", "content": question }
        ]

        return messages


