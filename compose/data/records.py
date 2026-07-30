"""Accessors shared by flat evaluation records and LLaVA conversation records."""

from typing import Dict

from llava.constants import DEFAULT_IMAGE_TOKEN


def question_text(record: Dict[str, object]) -> str:
    if "text" in record:
        return str(record["text"])
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if message.get("from") == "human":
                value = str(message["value"])
                return value.replace(DEFAULT_IMAGE_TOKEN, "", 1).lstrip("\n")
    raise ValueError("record has no human question")


def answer_text(record: Dict[str, object]) -> str:
    if "answer" in record:
        return str(record["answer"])
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for message in reversed(conversations):
            if message.get("from") == "gpt":
                return str(message["value"])
    raise ValueError("record has no teacher answer")
