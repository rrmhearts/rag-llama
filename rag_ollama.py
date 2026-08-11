import sys
# Updated import to use the new 'ddgs' package name
from ddgs import DDGS
import ollama


def search_web(query, max_results=4):
  """Performs a free DuckDuckGo search using the ddgs library."""
  print(f"\n🌐 Searching the web for: '{query}'...")
  try:
    results = DDGS().text(query, max_results=max_results)
    snippets = []
    for i, r in enumerate(results, 1):
      # Fallbacks added for URL and Body keys across different library versions
      title = r.get("title", "No Title")
      url = r.get("href") or r.get("url", "")
      body = r.get("body") or r.get("snippet", "")
      snippets.append(
          f"[{i}] Title: {title}\nURL: {url}\nSnippet: {body}\n---"
      )
    return "\n".join(snippets)
  except Exception as e:
    return f"Search failed: {e}"


def build_rag_prompt(user_query, context):
  """Injects zero-indexed web context above the user's question."""
  return f"""You are a helpful research assistant. Use ONLY the real-time web search context provided below to answer the question accurately and concisely. If the context doesn't contain the answer, state what you know or what is missing.

### Web Search Context:
{context}

### User Question:
{user_query}
"""


def main():
  model_name = "gemma3:1b"
  print(
      f"=== Zero-Indexed Web RAG Chat ({model_name}) ===\nType 'exit' or"
      " 'quit' to close.\n"
  )

  while True:
    try:
      user_input = input("\n>>> You: ").strip()
      if not user_input:
        continue
      if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

      # 1. Zero-indexed web retrieval
      context = search_web(user_input)

      # 2. Build augmented prompt
      augmented_prompt = build_rag_prompt(user_input, context)

      # 3. Stream response from local Ollama
      print(f"\n🧠 {model_name}: ", end="", flush=True)

      print("PROMPT:", augmented_prompt)

      stream = ollama.chat(
          model=model_name,
          messages=[{"role": "user", "content": augmented_prompt}],
          stream=True,
      )

      for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
      print()  # Newline after stream finishes

    except KeyboardInterrupt:
      print("\nExiting...")
      break
    except Exception as e:
      print(f"\n[Error]: {e}")


if __name__ == "__main__":
  main()