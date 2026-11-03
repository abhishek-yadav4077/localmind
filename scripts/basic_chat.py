"""Assignment 24 — Task 1: Ollama setup & basic chat, traced in LangSmith.

Prereqs:
    ollama serve                 # start the local Ollama server
    ollama pull llama3.2         # pull a model
    cp .env.example .env         # set LANGSMITH_TRACING=true and LANGSMITH_API_KEY

Run:
    python scripts/basic_chat.py
    python scripts/basic_chat.py --model mistral --prompt "Explain RAG in 2 sentences."

LangSmith tracing needs no code: LangChain reads LANGSMITH_TRACING /
LANGSMITH_API_KEY / LANGSMITH_PROJECT from the environment and sends a trace
for every run. We only add a run name, tags and metadata so it's easy to find.
"""

import argparse

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaLLM

load_dotenv()  # must run before the first LangChain call so tracing env vars are seen

parser = argparse.ArgumentParser(description="Send one prompt to a local Ollama model.")
parser.add_argument("--model", default="llama3.2")
parser.add_argument("--prompt", default="What is Ollama? Answer in two sentences.")
args = parser.parse_args()

trace_config = {
    "run_name": "basic_chat",
    "tags": ["assignment-24", "basic-chat", args.model],
    "metadata": {"model": args.model, "script": "basic_chat.py"},
}

# 1) LLM wrapper: plain text in, plain text out (completion-style).
llm = OllamaLLM(model=args.model, temperature=0.2)
print("--- OllamaLLM (text completion) ---")
print(llm.invoke(args.prompt, config=trace_config))

# 2) Chat model wrapper + LCEL chain: messages in, AIMessage out.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, concise assistant."),
        ("human", "{question}"),
    ]
)
chain = prompt | ChatOllama(model=args.model, temperature=0.2) | StrOutputParser()
print("\n--- ChatOllama (chat model, LCEL chain) ---")
print(chain.invoke({"question": args.prompt}, config=trace_config))
