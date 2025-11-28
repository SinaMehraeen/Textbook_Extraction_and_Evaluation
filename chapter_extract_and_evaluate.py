"""Combined chapter extraction + evaluation using OpenRouter (Gemini 2.5 Flash Lite).

- Reads textbooks as Markdown from --textbook-dir
- Loads evaluation rubric from --criteria-csv (Index, Category, Item)
- For each textbook, sends ONE request to OpenRouter:
    - Extracts cleaned chapters
    - Evaluates each chapter against ALL rubric criteria
- Writes, per book:
    - <book_id>_extracted.md : chapters + evaluation tables in Markdown
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


LOGGER = logging.getLogger("chapter_extract_and_evaluate")

# ---------- Constants ----------

DEFAULT_TIMEOUT = 900  # 15 minutes for large books

# ---------- Data models ----------

@dataclass(frozen=True)
class Criterion:
    index: int
    category: str
    item: str

@dataclass
class Chapter:
    book_id: str
    chapter: str
    title: str
    content: str
    word_count: int
    source_path: str

# ---------- HTTP Session ----------

class TimeoutHTTPAdapter(HTTPAdapter):
    """HTTP adapter with default timeout."""
    def __init__(self, timeout=None, *args, **kwargs):
        self.timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        return super().send(request, **kwargs)

def create_http_session(timeout=DEFAULT_TIMEOUT, retries=3):
    """Create a requests Session with retry and timeout configuration."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    
    adapter = TimeoutHTTPAdapter(timeout=timeout, max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# ---------- Rubric loading ----------

def load_criteria(csv_path: Path) -> List[Criterion]:
    """Load evaluation criteria from CSV with columns: Index, Category, Item."""
    criteria = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if idx := row.get("Index"):
                criteria.append(
                    Criterion(
                        index=int(idx),
                        category=(row.get("Category") or "").strip(),
                        item=(row.get("Item") or "").strip(),
                    )
                )
    
    if not criteria:
        raise ValueError(f"No criteria loaded from {csv_path}")
    
    criteria.sort(key=lambda c: c.index)
    LOGGER.info("Loaded %d criteria from %s", len(criteria), csv_path)
    return criteria

# ---------- Textbook discovery & reading ----------

def discover_textbooks(textbook_dir: Path) -> List[Path]:
    """Find all markdown files in the textbook directory."""
    paths = sorted(textbook_dir.glob("*.md"))
    LOGGER.info("Found %d textbook(s) in %s", len(paths), textbook_dir)
    return paths

def truncate_to_chars(text: str, max_chars: int) -> str:
    """Truncate text to maximum character count."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    LOGGER.warning("Truncating book from %d to %d characters", len(text), max_chars)
    return text[:max_chars]

# ---------- Prompt building ----------

SYSTEM_INSTRUCTION = """You are an expert academic editor and educational evaluator tasked with two jobs:
1. Extract clean chapter text from textbooks
2. Evaluate each chapter against a quality rubric

Be precise, thorough, and follow the output format exactly."""

def build_combined_prompt(book_id: str, excerpt: str, criteria: Sequence[Criterion]) -> str:
    """Build prompt that asks the model to both extract chapters and evaluate them."""
    rubric_lines = [f"{c.index}. [{c.category}] {c.item}" for c in criteria]
    rubric_block = "\n".join(rubric_lines)
    min_idx = min(c.index for c in criteria)
    max_idx = max(c.index for c in criteria)

    return f"""# TASK: Extract and Evaluate Textbook Chapters

FILE ID: {book_id}

## STEP 1: CHAPTER EXTRACTION

Extract ONLY main chapter body text. Follow these rules strictly:

**What to INCLUDE:**
- Main narrative text from numbered chapters (e.g., "Chapter 1", "1. Introduction")
- Section headings within chapters
- Equations that are part of the narrative flow
- Essential inline formatting (bold, italics) for readability

**What to EXCLUDE:**
- Front matter: Preface, Acknowledgments, Table of Contents
- Back matter: Index, Bibliography, Glossary, Appendices
- Supplementary content: Sidebars, text boxes, activities, exercises, solutions
- Non-narrative elements: Tables, table captions, figure captions, figure titles
- Metadata: Page numbers, headers, footers, learning objectives lists

**Quality criteria:**
- Each chapter should contain continuous, readable prose
- Remove distracting elements while preserving the author's voice
- Maintain logical flow and context

---

## STEP 2: CHAPTER EVALUATION

Evaluate EACH extracted chapter against ALL {len(criteria)} rubric items below:

{rubric_block}

**Rating scale:**
- **1** = Clear positive evidence present (the chapter demonstrates this quality)
- **0** = Evidence absent, unclear, or contradictory

**For each rating, provide:**
- A specific rationale (max 30 words)
- Reference to chapter content, not general statements
- If not applicable, rate 0 and explain why

**Critical:** You must provide ratings for indices {min_idx} through {max_idx} for EVERY chapter.

---

## OUTPUT FORMAT

Use this exact structure with delimiters:

```
<BOOK_TITLE>
[Book title or FILE ID]
</BOOK_TITLE>

<CHAPTER>
number: [integer]
title: [chapter title]
word_count: [approximate word count of chapter text]
---
[Clean chapter text in Markdown]
---
RATINGS:
[index]: [0 or 1] | [rationale]
[index]: [0 or 1] | [rationale]
...
</CHAPTER>

<CHAPTER>
[next chapter...]
</CHAPTER>
```

**Validation checklist:**
- [ ] Used exact delimiter format (no variations)
- [ ] Each chapter includes word_count field
- [ ] Each chapter has ALL {len(criteria)} ratings
- [ ] Ratings are only 0 or 1 (no other values)
- [ ] Only extracted actual chapters from the text
- [ ] Rationales are specific and concise

---

## TEXTBOOK CONTENT

{excerpt}
"""

# ---------- Response parsing ----------

def parse_delimited_output(text: str, fallback_book_id: str) -> Tuple[str, List[Dict]]:
    """Parse the delimited format output from the model."""
    # Extract book title
    book_title = fallback_book_id
    if (title_start := text.find("<BOOK_TITLE>")) != -1:
        if (title_end := text.find("</BOOK_TITLE>")) != -1:
            book_title = text[title_start + 12:title_end].strip()

    # Extract chapters
    chapters = []
    for block in text.split("<CHAPTER>")[1:]:
        try:
            chapter_content = block.split("</CHAPTER>")[0].strip()
            if not chapter_content:
                continue

            # Parse chapter metadata
            parts = chapter_content.split("---")
            meta_text = parts[0].strip()
            
            chapter_data = {}
            for line in meta_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    chapter_data[key.strip()] = value.strip()

            # Extract main content
            main_content = parts[1].strip() if len(parts) >= 2 else ""

            # Parse ratings
            ratings = []
            if "RATINGS:" in chapter_content:
                ratings_text = chapter_content.split("RATINGS:")[1].strip()
                for line in ratings_text.split("\n"):
                    if "|" not in line:
                        continue
                    index_rating, rationale = line.split("|", 1)
                    if ":" not in index_rating:
                        continue
                    index, rating = index_rating.split(":", 1)
                    try:
                        ratings.append({
                            "index": int(index.strip()),
                            "rating": int(rating.strip()),
                            "rationale": rationale.strip()
                        })
                    except ValueError:
                        continue

            chapters.append({
                "number": int(chapter_data.get("number", 0)),
                "title": chapter_data.get("title", "Untitled Chapter"),
                "text": main_content,
                "ratings": ratings
            })
        except Exception as e:
            LOGGER.warning("Failed to parse chapter block: %s", e)
            continue

    return book_title, chapters

def format_payload_to_chapters_and_records(
    book_title: str,
    chapters_data: List[Dict],
    criteria: Sequence[Criterion],
    source_path: str,
) -> Tuple[List[Chapter], List[Dict]]:
    """Convert parsed data into Chapter objects and evaluation records."""
    chapters = []
    records = []
    criterion_map = {c.index: c for c in criteria}

    for idx, ch in enumerate(chapters_data, start=1):
        number = str(ch.get("number") or idx)
        title = (ch.get("title") or f"Chapter {number}").strip()
        text_content = (ch.get("text") or "").strip()

        if not text_content:
            LOGGER.warning("Skipping empty chapter %s - %s", number, title)
            continue

        chapter_obj = Chapter(
            book_id=book_title,
            chapter=number,
            title=title,
            content=text_content,
            word_count=len(text_content.split()),
            source_path=source_path,
        )
        chapters.append(chapter_obj)

        # Build rating lookup
        rating_by_index = {int(r["index"]): r for r in ch.get("ratings", []) if "index" in r}

        # Create one record per criterion
        for c_idx, criterion in sorted(criterion_map.items()):
            r_data = rating_by_index.get(c_idx, {})
            records.append({
                "book_id": book_title,
                "chapter_number": number,
                "chapter_title": title,
                "criterion_index": criterion.index,
                "criterion_category": criterion.category,
                "criterion_item": criterion.item,
                "rating": r_data.get("rating"),
                "rationale": r_data.get("rationale", ""),
                "source_path": source_path,
            })

    return chapters, records

# ---------- Output writers ----------

def write_markdown_for_book(
    chapters: List[Chapter],
    records: List[Dict],
    criteria: Sequence[Criterion],
    output_dir: Path,
    book_filename: str,
) -> None:
    """Write chapters and evaluation tables to Markdown file."""
    if not chapters:
        LOGGER.warning("No chapters to write for %s", book_filename)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{book_filename}_extracted.md"

    # Group evaluation records by chapter
    eval_by_chapter = {}
    for rec in records:
        ch_num = str(rec["chapter_number"])
        eval_by_chapter.setdefault(ch_num, []).append(rec)

    with output_file.open("w", encoding="utf-8") as f:
        # YAML frontmatter
        f.write(f"""---
book_id: {chapters[0].book_id}
source_file: {book_filename}
total_chapters: {len(chapters)}
---

# {chapters[0].book_id}

""")

        # Each chapter with evaluation table
        for chapter in chapters:
            f.write(f"""## Chapter {chapter.chapter}: {chapter.title}

*Word count: {chapter.word_count}*

{chapter.content}

### Evaluation
""")

            # Write evaluation records
            ch_recs = sorted(
                eval_by_chapter.get(chapter.chapter, []),
                key=lambda r: r["criterion_index"]
            )

            if not ch_recs:
                for c in criteria:
                    f.write(f"{c.index}, , \n")
            else:
                for rec in ch_recs:
                    idx = rec["criterion_index"]
                    rating = "" if rec["rating"] is None else str(rec["rating"])
                    rationale = (rec.get("rationale") or "").replace("\n", " ")
                    f.write(f"{idx}, {rating}, {rationale}\n")

            f.write("\n---\n\n")

    LOGGER.info("✓ Wrote %d chapters with evaluations to %s", len(chapters), output_file)

# ---------- Caching & Progress ----------

def get_cache_key(book_id: str, book_text: str, criteria: Sequence[Criterion], args: argparse.Namespace) -> str:
    """Generate cache key based on content and parameters."""
    components = [
        book_id,
        hashlib.md5(book_text.encode()).hexdigest(),
        hashlib.md5(''.join(f"{c.index}{c.category}{c.item}" for c in criteria).encode()).hexdigest(),
        hashlib.md5(f"{args.model}{args.temperature}{args.max_output_tokens}{args.max_book_chars}".encode()).hexdigest(),
    ]
    return hashlib.md5(''.join(components).encode()).hexdigest()

def load_cache(cache_key: str, cache_dir: Path) -> Dict | None:
    """Load data from cache."""
    cache_file = cache_dir / f"{cache_key}.pkl"
    if not cache_file.exists():
        return None
    
    try:
        with cache_file.open('rb') as f:
            return pickle.load(f)
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None

def save_cache(cache_key: str, data: Dict, cache_dir: Path) -> None:
    """Save data to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / f"{cache_key}.pkl").open('wb') as f:
        pickle.dump(data, f)

def load_progress(progress_file: Path) -> set:
    """Load completed books set."""
    if not progress_file.exists():
        return set()
    try:
        with progress_file.open('rb') as f:
            return pickle.load(f)
    except Exception:
        return set()

def save_progress(progress_file: Path, completed_books: set) -> None:
    """Save completed books set."""
    with progress_file.open('wb') as f:
        pickle.dump(completed_books, f)

# ---------- OpenRouter call ----------

def call_openrouter(
    api_key: str,
    model: str,
    system_instruction: str,
    user_content: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int = 900,
) -> str:
    """Call OpenRouter API and return response content."""
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://chapter-extraction-project",
                "X-Title": "chapter_extract_and_evaluate",
            },
            json=request_body,
            timeout=timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {data['error']}")
            
        return data["choices"][0]["message"]["content"]
        
    except requests.RequestException as e:
        LOGGER.error("Request failed: %s", e)
        raise
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        LOGGER.error("Invalid response format: %s", e)
        raise RuntimeError(f"Invalid OpenRouter response: {e}") from e

# ---------- Main processing ----------

def process_books(
    book_payloads: Sequence[Tuple[str, str, str]],
    criteria: Sequence[Criterion],
    args: argparse.Namespace,
) -> None:
    """Process all books with extraction and evaluation."""
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("API key required via --api-key or OPENROUTER_API_KEY env var")

    # Cache and resume are enabled by default
    use_cache = not args.no_cache
    use_resume = not args.no_resume
    
    cache_dir = args.output / "cache"
    progress_file = args.output / "progress.pkl"
    completed_books = load_progress(progress_file) if use_resume else set()
    
    # Filter completed books
    remaining = [(bid, txt, src) for bid, txt, src in book_payloads if bid not in completed_books]
    
    if use_resume and len(remaining) < len(book_payloads):
        LOGGER.info("Resuming: %d of %d books remaining", len(remaining), len(book_payloads))

    total_chapters = 0
    successful_books = 0

    with tqdm(remaining, desc="Processing books", unit="book") as pbar:
        for book_id, book_text, source_path in pbar:
            pbar.set_description(f"Processing: {book_id[:30]}")
            
            # Try cache first
            raw_text = None
            if use_cache:
                cache_key = get_cache_key(book_id, book_text, criteria, args)
                if cached := load_cache(cache_key, cache_dir):
                    raw_text = cached.get("raw_text")
                    pbar.set_postfix_str("💾 Cached")

            # Call API if needed
            if raw_text is None:
                excerpt = truncate_to_chars(book_text, args.max_book_chars)
                prompt = build_combined_prompt(book_id, excerpt, criteria)
                
                pbar.set_postfix_str("⏳ Calling API")
                
                try:
                    raw_text = call_openrouter(
                        api_key, args.model, SYSTEM_INSTRUCTION,
                        prompt, args.temperature, args.max_output_tokens,
                        timeout=args.timeout
                    )
                    
                    if use_cache:
                        save_cache(cache_key, {"raw_text": raw_text}, cache_dir)
                        
                except Exception as e:
                    LOGGER.error("API call failed for %s: %s", book_id, e)
                    pbar.set_postfix_str("❌ Failed")
                    continue

            # Parse and format
            try:
                book_title, chapters_data = parse_delimited_output(raw_text, book_id)
                
                if not chapters_data:
                    raise ValueError("No chapters found")
                    
                chapters, records = format_payload_to_chapters_and_records(
                    book_title, chapters_data, criteria, source_path
                )
                
            except Exception as e:
                LOGGER.error("Parse failed for %s: %s", book_id, e)
                if args.save_raw:
                    debug_dir = args.output / "raw_responses"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    (debug_dir / f"{book_id}_raw.txt").write_text(raw_text, encoding="utf-8")
                pbar.set_postfix_str("❌ Parse failed")
                continue

            if not chapters:
                LOGGER.warning("No valid chapters for %s", book_id)
                pbar.set_postfix_str("⚠️ No chapters")
                continue

            # Write output
            write_markdown_for_book(chapters, records, criteria, args.output, book_id)
            
            total_chapters += len(chapters)
            successful_books += 1
            
            if use_resume:
                completed_books.add(book_id)
                save_progress(progress_file, completed_books)

            pbar.set_postfix_str(f"✓ {len(chapters)} chapters")

    LOGGER.info("✓ Complete: %d chapters across %d books", total_chapters, successful_books)

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and evaluate textbook chapters using OpenRouter"
    )
    parser.add_argument("--textbook-dir", type=Path, default=Path("textbooks"))
    parser.add_argument("--criteria-csv", type=Path, default=Path("Evaluation_Criteria.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-book-chars", type=int, default=400000)
    parser.add_argument("--max-output-tokens", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900,
                       help="API request timeout in seconds (default: 900 = 15 minutes)")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching (caching enabled by default)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Disable resume (resume enabled by default)")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    criteria = load_criteria(args.criteria_csv)
    textbooks = discover_textbooks(args.textbook_dir)
    book_payloads = [
        (path.stem, path.read_text(encoding="utf-8"), str(path)) 
        for path in textbooks
    ]

    process_books(book_payloads, criteria, args)

if __name__ == "__main__":
    main()