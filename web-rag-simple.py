import sys
from bs4 import BeautifulSoup
from ddgs import DDGS
import ollama
import requests


def fetch_page_text(url, max_chars=3000):
  """Scrapes and cleans the body text of a webpage, truncating to fit context."""
  print(f"📄 Scraping top result: {url} ...")
  try:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, navigation, and footer tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
      tag.decompose()

    # Extract clean text and collapse extra whitespace
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")
  except Exception as e:
    return f"[Could not scrape page text: {e}]"


def search_web_with_first_page(query, max_results=4):
  """Searches DuckDuckGo and scrapes the full text of the first result."""
  print(f"\n🌐 Searching the web for: '{query}'...")
  try:
    results = list(DDGS().text(query, max_results=max_results))
    if not results:
      return "No results found."

    snippets = []

    for i, r in enumerate(results, 1):
      title = r.get("title", "No Title")
      url = r.get("href") or r.get("url", "")
      body = r.get("body") or r.get("snippet", "")

      # For the FIRST result, scrape the page text
      if i == 1 and url:
        page_text = fetch_page_text(url)
        snippets.append(
            f"[{i}] Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {body}\n"
            f"--- SCRAPED PAGE CONTENT (TOP RESULT) ---\n"
            f"{page_text}\n"
            f"-----------------------------------------"
        )
      else:
        snippets.append(
            f"[{i}] Title: {title}\nURL: {url}\nSnippet: {body}\n---"
        )

    return "\n".join(snippets)
  except Exception as e:
    return f"Search failed: {e}"


def build_rag_prompt(user_query, context):
  """Injects web snippets and the top-result page content above the user's question."""
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

      # 1. Zero-indexed web retrieval + scrape top hit
      context = search_web_with_first_page(user_input)

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