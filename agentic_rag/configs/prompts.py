"""Prompts used for LLMs."""

## Ignore line too long
# ruff: noqa: E501

LLM_SYSTEM_PROMPT = "You are an AI assistant that specializes in providing factually accurate and contextually relevant responses by using both your trained knowledge and retrieved information. Always prioritize the retrieved information when answering user queries. If the retrieved data doesn't fully answer the question, use your general knowledge to provide additional context. Make sure your responses are clear, concise, and helpful."

RAG_PROMPT = """
Context information is below.
---------------------
$context
---------------------
Given the context information and not prior knowledge, answer the query.
Query: $query
Answer:
"""


HYDE_PROMPT = """Given a question, generate a paragraph of text that answers the question.
Question: $query
Paragraph:
"""
