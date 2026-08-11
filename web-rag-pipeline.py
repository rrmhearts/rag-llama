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

def segment_and_tag_text(raw_text: str):
    """
    Strips bracketed citations (e.g., [7], [1, 2], [3-5]), segments raw text 
    sentence-by-sentence while ignoring decimals and common abbreviations, 
    and encloses each clean segment in a sequential XML-style tag.
    
    Returns:
        tagged_text (str): The text with <TAG-i>...</TAG-i> wrappers.
        tag_map (dict): A dictionary mapping "TAG-i" to the raw sentence content.
    """
    if not raw_text:
        return "", {}

    # 1. Strip bracketed numeric citations: [7], [12, 14], [3–5], etc.
    text = re.sub(r'\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]', '', raw_text)
    
    # Clean up any leftover space before punctuation caused by removing citations
    # e.g., "efficiency [7]." -> "efficiency ." -> "efficiency."
    text = re.sub(r'\s+([.!?])', r'\1', text)

    # 2. MASK DECIMALS: Replace period between digits with a temporary placeholder
    # e.g., "3.14" -> "3<DOT>14"
    text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)

    # 3. MASK ABBREVIATIONS: Replace periods in common abbreviations with <DOT>
    # Using a callback function guarantees clean case-insensitive matching
    abbrev_pattern = re.compile(
        r'\b(?:Mr|Mrs|Ms|Dr|Prof|Gen|Rep|Sen|St|Jr|Sr|vs|etc|i\.e\.|e\.g\.|'
        r'Fig|Eq|No|Vol|Inc|Co|Corp|Ltd|U\.S\.|U\.K\.)',
        re.IGNORECASE
    )
    text = abbrev_pattern.sub(lambda m: m.group(0).replace('.', '<DOT>'), text)

    # 4. SPLIT SENTENCES: Now safe to split after any '.', '!', or '?' followed by whitespace
    # Using a 1-character fixed-width lookbehind (?<=[.!?]) works natively in Python re
    raw_segments = re.split(r'(?<=[.!?])\s+', text.strip())

    # 5. UNMASK & TAG: Restore periods and wrap clean segments
    tagged_segments = []
    tag_map = {}
    tag_counter = 1
    
    for seg in raw_segments:
        # Restore masked periods
        unmasked_seg = seg.replace('<DOT>', '.')
        
        # Normalize internal whitespace
        clean_seg = re.sub(r'\s+', ' ', unmasked_seg).strip()
        
        # Skip empty or trivial segments (e.g., stray punctuation)
        if len(clean_seg) > 5:
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
    clean body text while discarding headers, footers, scripts, nav bars,
    tables, and obvious references/sources sections.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            
            # 1. Remove standard boilerplate + TABLES
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "table"]):
                element.decompose()
                
            # 2. Remove common inline citation tags/classes (e.g., Wikipedia superscripts, footnote links)
            for ref_elem in soup.find_all(
                lambda tag: tag.name in ["sup", "ol", "ul", "div", "section"] and 
                any(
                    cls in " ".join(tag.get("class", [])).lower() or 
                    cls in (tag.get("id") or "").lower()
                    for cls in ["reference", "references", "citation", "citations", "cite", "biblio", "footnote"]
                )
            ):
                ref_elem.decompose()

            # 3. Strip trailing "References", "Sources Cited", or "Bibliography" sections by heading
            ref_heading_pattern = re.compile(
                r'^(references|sources(\s+cited)?|bibliography|works\s+cited|citations|footnotes)\s*$', 
                re.IGNORECASE
            )
            for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                if ref_heading_pattern.match(heading.get_text(strip=True)):
                    # Remove all subsequent siblings (the actual reference list below the heading)
                    for sibling in list(heading.find_next_siblings()):
                        sibling.decompose()
                    heading.decompose()
                
            # 4. Extract plain text from body
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ')
            else:
                text = soup.get_text(separator=' ')
                
            # 5. Clean up excessive whitespace
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


# Minimal stopword set to prevent trivial matches on query phrasing
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "from",
    "by", "with", "of", "for", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "what", "who", "where", "when", "why", "how",
    "which", "that", "this", "these", "those", "it", "its", "can", "could",
    "should", "would", "about", "did", "have", "has", "had"
}

def extractor_llm(request, tagged_content, model_name="qwen2.5:1.5b", window=1):
    """
    Step 5 of the Paradigm: Extractor-LLM.
    Pre-filters tagged_content using keyword matching + a nearby-tag window,
    then uses a highly constrained prompt to return ONLY relevant tag identifiers.
    """
    # 1. Extract query keywords (lowercase, alphanumeric alphanumeric stems, ignoring stopwords)
    raw_words = re.findall(r'\b[a-z0-9]+\b', request.lower())
    keywords = {word for word in raw_words if word not in STOPWORDS and len(word) > 1}

    # 2. Parse all <TAG-i> blocks from tagged_content
    # Returns a list of tuples: [('TAG-1', 'Sentence one...'), ('TAG-2', 'Sentence two...')]
    parsed_tags = re.findall(r'<(TAG-\d+)>(.*?)</\1>', tagged_content, re.DOTALL)

    if not parsed_tags:
        return "None"

    # 3. Identify matching tag indices and expand to nearby tags (±window)
    matching_indices = set()
    for i, (_, seg_text) in enumerate(parsed_tags):
        seg_lower = seg_text.lower()
        if any(kw in seg_lower for kw in keywords):
            matching_indices.add(i)

    # If keywords exist but nothing matched at all, short-circuit immediately
    if keywords and not matching_indices:
        return "None"

    # Expand matches to include nearby neighbor tags
    kept_indices = set()
    for idx in matching_indices:
        for offset in range(-window, window + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(parsed_tags):
                kept_indices.add(neighbor_idx)

    # If query was entirely stopwords (keywords empty), fallback to keeping all tags
    if not keywords:
        kept_indices = set(range(len(parsed_tags)))

    # 4. Reconstruct the pre-filtered tagged content string
    sorted_indices = sorted(kept_indices)
    filtered_tags = [
        f"<{parsed_tags[i][0]}>{parsed_tags[i][1]}</{parsed_tags[i][0]}>"
        for i in sorted_indices
    ]
    filtered_content = "\n".join(filtered_tags)

    # 5. Build prompt with pre-filtered content
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
{filtered_content}
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
