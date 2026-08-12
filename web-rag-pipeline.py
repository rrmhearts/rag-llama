import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Set, Tuple
from bs4 import BeautifulSoup
from ddgs import DDGS
import ollama

# =====================================================================
# 0. Configuration & Global Constants
# =====================================================================

# Minimal stopword set to prevent trivial matches on query phrasing
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "from",
    "by", "with", "of", "for", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "what", "who", "where", "when", "why", "how",
    "which", "that", "this", "these", "those", "it", "its", "can", "could",
    "should", "would", "about", "did", "have", "has", "had"
}

# Add this near your STOPWORDS and DEFAULT_HEADERS
MERGE_PRONOUNS = {
    "He", "She", "It", "They", "This", "That", "These", "Those", 
    "His", "Her", "Its", "Their"
}

DEFAULT_HEADERS: Dict[str, str] = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Pre-compiled regex patterns for performance
ABBREV_PATTERN = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Gen|Rep|Sen|St|Jr|Sr|vs|etc|i\.e\.|e\.g\.|'
    r'Fig|Eq|No|Vol|Inc|Co|Corp|Ltd|U\.S\.|U\.K\.)',
    re.IGNORECASE
)
REF_HEADING_PATTERN = re.compile(
    r'^(references|sources(\s+cited)?|bibliography|works\s+cited|citations|footnotes)\s*$',
    re.IGNORECASE
)


# =====================================================================
# 1. HTML Tagging Finite State Machine (FSM)
# =====================================================================

def segment_and_tag_text(raw_text: str) -> Tuple[str, Dict[str, str]]:
    """
    Processes text block-by-block, ensuring sentences don't bleed across 
    HTML boundaries. Enforces strict capitalization, punctuation, and word counts.
    """
    if not raw_text:
        return "", {}

    # Strip bracketed numeric citations
    text = re.sub(r'\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]', '', raw_text)
    
    # Process block by block (separated by our injected \n\n)
    html_blocks = text.split('\n\n')
    processed_segments = []
    
    for block in html_blocks:
        block = block.strip()
        if not block:
            continue
            
        # MASK DECIMALS & ABBREVIATIONS within the isolated block
        block = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', block)
        block = ABBREV_PATTERN.sub(lambda m: m.group(0).replace('.', '<DOT>'), block)
        
        # SPLIT SENTENCES inside this safe HTML block
        sentences = re.split(r'(?<=[.!?])\s+', block)
        
        for seg in sentences:
            unmasked_seg = seg.replace('<DOT>', '.').strip()
            if not unmasked_seg:
                continue
                
            # STRICT HEURISTICS
            # Must start with uppercase
            if not re.match(r'^[A-Z]', unmasked_seg): 
                continue
            # Must end with valid punctuation
            if unmasked_seg[-1] not in '.!?': 
                continue
                
            # Must have at least 4 alphabetical words (filters out data rows/number lists)
            alpha_words = [w for w in unmasked_seg.split() if re.search(r'[a-zA-Z]', w)]
            if len(alpha_words) < 4:
                continue
                
            # Pronoun merging logic
            first_word = re.sub(r'[^a-zA-Z]', '', unmasked_seg.split()[0])
            if first_word in MERGE_PRONOUNS and processed_segments:
                processed_segments[-1] = processed_segments[-1] + " " + unmasked_seg
            else:
                processed_segments.append(unmasked_seg)

    # UNMASK & TAG
    tagged_segments = []
    tag_map = {}
    tag_counter = 1
    
    for clean_seg in processed_segments:
        tag_name = f"TAG-{tag_counter}"
        tagged_segments.append(f"<{tag_name}>{clean_seg}</{tag_name}>")
        tag_map[tag_name] = clean_seg
        tag_counter += 1
            
    return "\n".join(tagged_segments), tag_map


# =====================================================================
# 2. Web Ingestion & Cleanup (BeautifulSoup)
# =====================================================================

def _clean_html_soup(soup: BeautifulSoup) -> None:
    """Helper function to decompose boilerplate, tables, reference sections, ads, and high-density link blocks."""
    # 1. Expand standard boilerplate removal
    for element in soup(["script", "style", "nav", "header", "footer", "aside", "table", "form", "iframe", "noscript"]):
        element.decompose()

    # 2. AD & PROMO FILTER: Remove elements by common ad, sidebar, and social media signatures
    ad_signatures = re.compile(r'ad|advert|banner|sponsor|promo|sidebar|social|share|widget|popup', re.IGNORECASE)
    for element in soup.find_all(['div', 'ul', 'section', 'span']):
        # CRITICAL FIX: Skip elements that were already destroyed when a parent was decomposed
        if element.attrs is None or not element.parent:
            continue
            
        # Safely extract class and id, handling edge cases where class might be a string
        classes = element.get('class', [])
        class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
        id_str = element.get('id') or ""
        
        if ad_signatures.search(class_str + " " + id_str):
            element.decompose()
            
    # 3. LINK DENSITY FILTER: Mathematically identify link farms and hidden menus
    for block in soup.find_all(['div', 'ul', 'ol', 'section', 'li']):
        # CRITICAL FIX: Skip destroyed elements here as well
        if block.attrs is None or not block.parent:
            continue
            
        text_length = len(block.get_text(strip=True))
        if text_length == 0:
            block.decompose()
            continue
            
        # Calculate how much of the text inside this block is wrapped in an anchor <a> tag
        link_text_length = sum(len(a.get_text(strip=True)) for a in block.find_all('a'))
        link_density = link_text_length / text_length
        
        if link_density >= 0.50:
            block.decompose()

    # 4. Remove common inline citation tags/classes
    for ref_elem in soup.find_all(
        lambda tag: tag.name in ["sup", "ol", "ul", "div", "section"] and 
        any(
            cls in " ".join(tag.get("class", [])).lower() or 
            cls in (tag.get("id") or "").lower()
            for cls in ["reference", "references", "citation", "citations", "cite", "biblio", "footnote"]
        )
    ):
        if ref_elem.parent:
            ref_elem.decompose()

    # 5. Strip trailing "References", "Sources Cited", or "Bibliography" sections by heading
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if not heading.parent:
            continue
            
        if REF_HEADING_PATTERN.match(heading.get_text(strip=True)):
            for sibling in list(heading.find_next_siblings()):
                sibling.decompose()
            heading.decompose()


def fetch_and_clean_html(url: str, timeout: int = 10) -> str:
    """
    Scrapes a web page, parses HTML using BeautifulSoup, and extracts 
    clean text while preserving block-level boundaries.
    """
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')
            
            _clean_html_soup(soup)
            
            # NEW: Inject hard boundaries around block-level tags to prevent text mashing
            block_tags = ['p', 'div', 'article', 'section', 'blockquote', 'li', 'h1', 'h2', 'h3', 'br', 'hr']
            for tag in soup.find_all(block_tags):
                tag.insert_before('\n\n')
                tag.insert_after('\n\n')
                
            body = soup.find('body')
            text = body.get_text(separator=' ') if body else soup.get_text(separator=' ')
            
            # Clean up excessive whitespace but preserve double newlines as block boundaries
            clean_text = re.sub(r'[ \t]+', ' ', text)
            clean_text = re.sub(r'\n\s*\n+', '\n\n', clean_text)
            
            return clean_text.strip()
            
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return ""

# =====================================================================
# 3. Collaborative Pipeline Components (Ollama & DDGS)
# =====================================================================

def parser_llm(user_query: str, model_name: str = "qwen2.5:1.5b") -> Tuple[bool, str]:
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


def web_search(keywords: str, max_results: int = 3) -> List[str]:
    """
    Queries DuckDuckGo search and extracts a list of relevant links.
    """
    urls = []
    try:
        print(f"Executing web search for: '{keywords}'...")
        with DDGS() as ddgs:
            results = ddgs.text(keywords, max_results=max_results)
            for r in results:
                # ddgs.text() returns 'href' while ddgs.news() returns 'url'
                url = r.get('href') or r.get('url')
                if url:
                    print(f"Found Web: {r.get('title', 'No Title')} -> {url}")
                    urls.append(url)
    except Exception as e:
        print(f"Web search error: {e}")
    return urls


def news_search(keywords: str, max_results: int = 3, timelimit: str = "w") -> List[str]:
    """
    Queries DuckDuckGo news search for recent articles and extracts relevant links.
    Supports timelimit parameters such as 'd' (day), 'w' (week), or 'm' (month).
    """
    urls = []
    try:
        print(f"Executing recent news search for: '{keywords}' (timelimit='{timelimit}')...")
        with DDGS() as ddgs:
            results = ddgs.news(keywords, timelimit=timelimit, max_results=max_results)
            for r in results:
                url = r.get('url') or r.get('href')
                if url:
                    source = r.get('source', 'Unknown Source')
                    date = r.get('date', 'Unknown Date')
                    print(f"Found News [{source} | {date}]: {r.get('title', 'No Title')} -> {url}")
                    urls.append(url)
    except Exception as e:
        print(f"News search error: {e}")
    return urls


def _prefilter_tagged_content(request: str, tagged_content: str, window: int) -> str:
    """Helper function to filter tagged content using keyword matching + window expansion."""
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
    return "\n".join(filtered_tags)


def extractor_llm(request: str, tagged_content: str, model_name: str = "qwen2.5:1.5b", window: int = 1) -> str:
    """
    Step 5 of the Paradigm: Extractor-LLM.
    Pre-filters tagged_content using keyword matching + a nearby-tag window,
    then uses a highly constrained prompt to return ONLY relevant tag identifiers.
    """
    filtered_content = _prefilter_tagged_content(request, tagged_content, window)
    if filtered_content == "None":
        return "None"

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


def generator_llm(user_query: str, retrieved_facts: str, model_name: str = "qwen2.5:1.5b") -> str:
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

def _extract_matched_segments(extracted_tags_str: str, tag_map: Dict[str, str]) -> List[str]:
    """Helper to map extracted TAG-i names back to their original segment sentences."""
    if not extracted_tags_str or extracted_tags_str.lower() == "none":
        return []
    
    # Find all occurrences of TAG-i patterns
    tags_found = re.findall(r'TAG-\d+', extracted_tags_str)
    valid_segments = []
    for t in tags_found:
        if t in tag_map:
            valid_segments.append(tag_map[t])
    return valid_segments


def run_web_rag_pipeline(question: str, slm_model: str = "qwen2.5:1.5b") -> str:
    print(f"Initializing Web RAG Pipeline for question: '{question}'...")
    
    # 1. Intent check & Keyword Extraction (Parser-LLM)
    search_needed, keywords = parser_llm(question, model_name=slm_model)
    
    extracted_context_parts = []
    
    if search_needed and keywords and keywords.lower() != "none":
        # 2. Web & News search (fetch general pages + recent news articles)
        question_urls = web_search(question, max_results=3)
        question_news = news_search(question, max_results=3)
        web_urls = web_search(keywords, max_results=3)
        news_urls = news_search(keywords, max_results=3, timelimit="w")
        
        # Merge and deduplicate URLs while preserving order
        seen_urls = set()
        search_urls = []
        for u in question_news + news_urls + question_urls + web_urls:
            if u not in seen_urls:
                seen_urls.add(u)
                search_urls.append(u)
        
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
            valid_segments = _extract_matched_segments(extracted_tags_str, tag_map)
            if valid_segments:
                source_context = " ".join(valid_segments)
                extracted_context_parts.append(f"{source_context}")
                print(f"  ✓ Successfully extracted {len(valid_segments)} relevant facts.")
            else:
                print("  ✗ No relevant facts found in this source.")
                
    else:
        print("Parser-LLM determined that no web search is needed or query is general knowledge.")

    # Merge facts
    final_context = "\n\n".join(extracted_context_parts) if extracted_context_parts else ""
    
    print("Final context: ", final_context)
    # 4. Final Answer Generation (Generative-LLM)
    print("\nGenerating final answer...")
    final_answer = generator_llm(question, final_context, model_name=slm_model)
    
    print("\n=== FINAL ANSWER ===")
    print(final_answer)
    print("====================\n")
    return final_answer


if __name__ == "__main__":
    # Test Question (Requires Real-time freshness)
    # question = "What is the capital of Wisconsin? "
    # question = "In the news, what country did Donald Trump recently leave?"
    # question = "Who is the richest man in the world?"
    question = "Who won the latest Formula 1 race and what team do they drive for?"
    # question = "Is an orca a type of dolphin?"
    run_web_rag_pipeline(question)


    # Error fetching URL  'NoneType' object has no attribute 'get'