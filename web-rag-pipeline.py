import re
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import ollama
from ddgs import DDGS
import os

# =====================================================================
# 1. HTML Tagging Finite State Machine (FSM)
# =====================================================================

def segment_and_tag_text(raw_text):
    """
    Applies a Finite State Machine (FSM) heuristic to segment the raw text 
    sentence-by-sentence, ignoring decimal points (e.g., 3.14) and 
    common abbreviations, and encloses each segment in a sequential tag.
    
    Returns:
        tagged_text (str): The text with <TAG-i>...</TAG-i> wrappers.
        tag_map (dict): A dictionary mapping "TAG-i" to the raw sentence content.
    """
    segments = []
    current_segment = []
    i = 0
    n = len(raw_text)

    while i < n:
        char = raw_text[i]
        current_segment.append(char)
        
        # Check for sentence-ending delimiters
        if char in ['!', '?']:
            segments.append("".join(current_segment).strip())
            current_segment = []
        elif char == '.':
            # FSM Check: Distinguish decimal points from periods
            is_decimal_or_abbr = False
            # Decimal check: digit before and digit after
            if i > 0 and raw_text[i-1].isdigit() and i + 1 < n and raw_text[i+1].isdigit():
                is_decimal_or_abbr = True
            
            # Common abbreviation checks (e.g., 'vs.', 'e.g.', 'i.e.', 'Mr.', 'Dr.')
            # Lookback check for common abbreviations
            lookback_text = "".join(current_segment[-4:]).lower()
            if any(abbr in lookback_text for abbr in ["vs.", "e.g.", "i.e.", "mr.", "dr.", "inc.", "co."]):
                is_decimal_or_abbr = True
                
            if not is_decimal_or_abbr:
                segments.append("".join(current_segment).strip())
                current_segment = []
        i += 1
        
    if current_segment:
        remaining = "".join(current_segment).strip()
        if remaining:
            segments.append(remaining)
            
    # Wrap sentences in content tags
    tagged_segments = []
    tag_map = {}
    tag_counter = 1
    
    for seg in segments:
        # Filter out noisy whitespace or empty lines
        clean_seg = re.sub(r'\s+', ' ', seg).strip()
        if clean_seg and len(clean_seg) > 5:  # Skip trivial segments
            tag_name = f"TAG-{tag_counter}"
            tagged_segments.append(f"<{tag_name}>{clean_seg}</{tag_name}>")
            tag_map[tag_name] = clean_seg
            tag_counter += 1
            
    return "\n".join(tagged_segments), tag_map


# =====================================================================
# 2. Web Ingestion & Cleanup (BeautifulSoup)
# =====================================================================

def fetch_and_clean_html(url, timeout=10):
    """
    Scrapes a web page, parses HTML using BeautifulSoup, and extracts 
    clean body text while discarding headers, footers, scripts, and nav bars.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts, styles, navigation, headers, and footers
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
                
            # Extract plain text from body
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ')
            else:
                text = soup.get_text(separator=' ')
                
            # Clean up excessive whitespace
            clean_lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    clean_lines.append(line)
            return " ".join(clean_lines)
            
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""


# =====================================================================
# 3. Collaborative Pipeline Components (Ollama & DDGS)
# =====================================================================

def parser_llm(user_query, model_name="qwen2.5:1.5b"):
    """
    Step 1 of the Paradigm: Multi-functional Parser-LLM.
    Determines if search is needed and extracts clean keywords in a single pass.
    """
    system_prompt = (
        "You are an AI assistant that determines whether a user's question requires real-time web search "
        "and extracts search keywords. Analyze the query carefully.\n"
        "Respond strictly in the following format:\n"
        "SEARCH_NEEDED: [YES/NO]\n"
        "KEYWORDS: [search keywords or None]"
    )
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {user_query}"}
        ],
        options={"temperature": 0.0}
    )
    
    content = response['message']['content'].strip()
    print("\n--- [Parser-LLM Output] ---")
    print(content)
    print("---------------------------\n")
    
    search_needed = False
    keywords = ""
    
    # Parse output lines
    for line in content.splitlines():
        if "SEARCH_NEEDED:" in line:
            search_needed = "YES" in line.upper()
        elif "KEYWORDS:" in line:
            keywords = line.replace("KEYWORDS:", "").strip()
            
    return search_needed, keywords


def web_search(keywords, max_results=3):
    """
    Queries DuckDuckGo search and extracts a list of relevant links.
    """
    urls = []
    try:
        print(f"Executing web search for: '{keywords}'...")
        with DDGS() as ddgs:
            results = ddgs.text(keywords, max_results=max_results)
            for r in results:
                print(f"Found: {r['title']} -> {r['href']}")
                urls.append(r['href'])
    except Exception as e:
        print(f"Web search error: {e}")
    return urls


def extractor_llm(request, tagged_content, model_name="qwen2.5:1.5b"):
    """
    Step 5 of the Paradigm: Extractor-LLM.
    Uses a highly constrained prompt to analyze tagged content and return 
    ONLY the relevant tag identifiers (or None), optimizing token consumption.
    """
    prompt_template = f"""
Analyze the following user question (Request) and the tagged text segment (Tagged Content). 
Your task is to identify which sequential tag blocks (e.g., <TAG-i>) contain facts that are 
directly relevant to answering the Request.

CRITICAL INSTRUCTIONS:
- You must output ONLY the tag names themselves as a comma-separated list (e.g., TAG-1, TAG-3, TAG-4).
- If no tags contain relevant information, output exactly: None.
- Do not output any explanation, markdown, summaries, introduction, or thoughts. 
- Do not output any prose. Your output must strictly be either the tag names or 'None'.

Request: {request}

Tagged Content:
\"\"\"
{tagged_content}
\"\"\"

Output tags or 'None':"""

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt_template}
        ],
        options={"temperature": 0.0}
    )
    
    return response['message']['content'].strip()


def generator_llm(user_query, retrieved_facts, model_name="qwen2.5:1.5b"):
    """
    Step 7 of the Paradigm: Backbone Generative-LLM.
    Generates the final response grounded completely in the clean extracted context.
    """
    system_prompt = (
        "You are a factual assistant. Answer the user's question using ONLY the provided facts. "
        "Every claim you make must be grounded strictly in the context. If the context is empty or "
        "does not contain the answer, reply that you cannot find the answer in the provided search results."
    )
    
    user_prompt = f"""
Facts:
\"\"\"
{retrieved_facts}
\"\"\"

Question: {user_query}
"""
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.3}
    )
    return response['message']['content'].strip()


# =====================================================================
# 4. End-to-End Pipeline Execution
# =====================================================================

def run_web_rag_pipeline(question, slm_model="qwen2.5:1.5b"):
    print(f"Initializing Web RAG Pipeline for question: '{question}'...")
    
    # 1. Intent check & Keyword Extraction (Parser-LLM)
    search_needed, keywords = parser_llm(question, model_name=slm_model)
    
    extracted_context_parts = []
    
    if search_needed and keywords and keywords.lower() != "none":
        # 2. Web search
        search_urls = web_search(keywords, max_results=3)
        
        # 3. Fetch, clean, tag, and extract facts
        for idx, url in enumerate(search_urls, 1):
            print(f"\nProcessing Source #{idx}: {url}")
            raw_body = fetch_and_clean_html(url)
            
            if not raw_body:
                continue
                
            # Apply tagging FSM
            tagged_content, tag_map = segment_and_tag_text(raw_body)
            print("TAGGED_CONTENT:", tagged_content)
            # Extract relevant tags (Extractor-LLM)
            extracted_tags_str = extractor_llm(question, tagged_content, model_name=slm_model)
            print(f"  Extractor returned tags: '{extracted_tags_str}'")
            
            # Retrieve segment content mapped to extracted tags
            if extracted_tags_str and extracted_tags_str.lower() != "none":
                # Find all occurrences of TAG-i patterns
                tags_found = re.findall(r'TAG-\d+', extracted_tags_str)
                valid_segments = []
                for t in tags_found:
                    if t in tag_map:
                        valid_segments.append(tag_map[t])
                
                if valid_segments:
                    source_context = " ".join(valid_segments)
                    extracted_context_parts.append(f"[Source: {url}]: {source_context}")
                    print(f"  ✓ Successfully extracted {len(valid_segments)} relevant facts.")
            else:
                print("  ✗ No relevant facts found in this source.")
                
    else:
        print("Parser-LLM determined that no web search is needed or query is general knowledge.")

    # Merge facts
    final_context = "\n\n".join(extracted_context_parts) if extracted_context_parts else ""
    
    # 4. Final Answer Generation (Generative-LLM)
    print("\nGenerating final answer...")
    final_answer = generator_llm(question, final_context, model_name=slm_model)
    
    print("\n=== FINAL ANSWER ===")
    print(final_answer)
    print("====================\n")
    return final_answer


if __name__ == "__main__":
    # Test Question (Requires Real-time freshness)
    question = "Who is the richest man in the world?"
    run_web_rag_pipeline(question)
