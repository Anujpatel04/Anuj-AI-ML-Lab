import argparse
import json
import os
import sys
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import chromadb
import requests
import streamlit as st
import tiktoken
from openai import OpenAI


FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "index.json")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


SYSTEM_PROMPT = """You are a domain expert assistant.
Answer questions using only the provided context.
If the answer is not in the context, say: "I don't know based on my knowledge base."
Be accurate, concise, and professional.
Include source URLs when possible.
"""


def ensure_env() -> None:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if not FIRECRAWL_API_KEY:
        raise EnvironmentError("FIRECRAWL_API_KEY is not set.")


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def ensure_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(CACHE_INDEX_PATH):
        with open(CACHE_INDEX_PATH, "w", encoding="utf-8") as file:
            json.dump({}, file)


def load_cache_index() -> Dict[str, Dict]:
    ensure_cache()
    with open(CACHE_INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_cache_index(index: Dict[str, Dict]) -> None:
    with open(CACHE_INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(index, file, indent=2)


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def generate_queries(domain: str) -> List[str]:
    return [
        f"{domain} basics",
        f"{domain} fundamentals",
        f"{domain} best practices",
        f"{domain} key concepts",
        f"{domain} introduction",
    ]


def firecrawl_search(query: str, limit: int = 5) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {"query": query, "limit": limit}
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    results = data.get("data", [])
    return [item.get("url") for item in results if item.get("url")]


def firecrawl_scrape(url: str) -> Tuple[str, str, str]:
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {"url": url, "formats": ["markdown"]}
    response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json().get("data", {})

    text = data.get("markdown") or data.get("content") or ""
    title = data.get("title") or ""
    final_url = data.get("url") or url
    return text.strip(), title.strip(), final_url.strip()


def tokenize_chunks(text: str) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [chunk for chunk in chunks if chunk]


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_openai_client()
    embeddings = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        response = client.embeddings.create(model=DEFAULT_EMBED_MODEL, input=batch)
        embeddings.extend([item.embedding for item in response.data])
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[embeddings] tokens={usage.total_tokens}")
    return embeddings


def build_collection(domain: str) -> str:
    return f"domain_{slugify(domain)}"


def load_or_create_collection(domain: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CACHE_DIR)
    name = build_collection(domain)
    return client.get_or_create_collection(name=name)


def cache_exists(domain: str) -> bool:
    index = load_cache_index()
    name = build_collection(domain)
    entry = index.get(name)
    if not entry:
        return False
    collection = load_or_create_collection(domain)
    return collection.count() > 0


def update_cache(domain: str, urls: List[str]) -> None:
    index = load_cache_index()
    name = build_collection(domain)
    index[name] = {"domain": domain, "urls": urls, "timestamp": int(time.time())}
    save_cache_index(index)


def _log(message: str, log_fn: Optional[Callable[[str], None]] = None) -> None:
    if log_fn:
        log_fn(message)
    else:
        print(message)


def scrape_domain(domain: str, log_fn: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    queries = generate_queries(domain)
    urls = []
    for query in queries:
        try:
            urls.extend(firecrawl_search(query, limit=3))
        except Exception as exc:
            _log(f"[search] failed for '{query}': {exc}", log_fn)

    urls = list(dict.fromkeys(urls))[:12]
    if not urls:
        raise RuntimeError("No URLs found for the domain.")

    _log(f"[scrape] found {len(urls)} URLs", log_fn)
    pages = []
    for url in urls:
        try:
            text, title, final_url = firecrawl_scrape(url)
            if text:
                pages.append({"text": text, "title": title, "url": final_url})
                _log(f"[scrape] ok: {final_url}", log_fn)
        except Exception as exc:
            _log(f"[scrape] failed: {url} ({exc})", log_fn)
    update_cache(domain, urls)
    return pages


def index_domain(domain: str, log_fn: Optional[Callable[[str], None]] = None) -> None:
    if cache_exists(domain):
        _log("[index] cache found, skipping scraping", log_fn)
        return

    pages = scrape_domain(domain, log_fn=log_fn)
    collection = load_or_create_collection(domain)

    documents = []
    metadatas = []
    ids = []

    for page in pages:
        chunks = tokenize_chunks(page["text"])
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"title": page["title"], "url": page["url"]})
            ids.append(str(uuid.uuid4()))

    if not documents:
        raise RuntimeError("No content collected to index.")

    embeddings = embed_texts(documents)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    _log(f"[index] stored {len(documents)} chunks", log_fn)


def retrieve_context(domain: str, question: str) -> List[Dict[str, str]]:
    collection = load_or_create_collection(domain)
    client = get_openai_client()
    embedding = client.embeddings.create(model=DEFAULT_EMBED_MODEL, input=[question]).data[0].embedding

    results = collection.query(query_embeddings=[embedding], n_results=TOP_K)
    contexts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append({"text": doc, "url": meta.get("url", "")})
    return contexts


def answer_question(domain: str, question: str) -> str:
    contexts = retrieve_context(domain, question)
    context_block = "\n\n".join(
        [f"Source: {item['url']}\n{item['text']}" for item in contexts]
    )

    prompt = (
        f"Domain: {domain}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context. If missing, say you don't know based on your knowledge base."
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    usage = getattr(response, "usage", None)
    if usage:
        print(f"[chat] tokens={usage.total_tokens}")

    return (response.choices[0].message.content or "").strip()


def chat_loop(domain: str) -> None:
    print("\nAsk questions about the domain. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        answer = answer_question(domain, question)
        print(f"\nAssistant: {answer}\n")


def run_streamlit() -> None:
    st.set_page_config(page_title="Domain Q&A Chatbot", layout="centered")
    st.title("Domain-Specific Q&A Chatbot")
    st.write("Select a domain, index sources, and ask questions from the scraped knowledge base.")

    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "indexed_domain" not in st.session_state:
        st.session_state.indexed_domain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not OPENAI_API_KEY or not FIRECRAWL_API_KEY:
        st.error("Missing required API keys in environment variables.")
        st.info("Required: OPENAI_API_KEY and FIRECRAWL_API_KEY")
        st.stop()

    domain = st.text_input("Domain", placeholder="e.g., Healthcare, Finance, AI, Law")
    index_col, status_col = st.columns([1, 2])

    log_box = st.empty()

    def log_fn(message: str) -> None:
        st.session_state.logs.append(message)
        log_box.code("\n".join(st.session_state.logs[-15:]))

    with index_col:
        index_clicked = st.button("Scrape & Index")

    with status_col:
        if domain:
            if cache_exists(domain):
                st.success("Cache found for this domain. Indexing will be skipped.")
            else:
                st.info("No cache found. Scraping will run.")

    if index_clicked:
        if not domain.strip():
            st.error("Please enter a domain before indexing.")
        else:
            st.session_state.logs = []
            with st.spinner("Indexing domain..."):
                try:
                    index_domain(domain.strip(), log_fn=log_fn)
                    st.session_state.indexed_domain = domain.strip()
                    st.success("Indexing completed.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    if st.session_state.indexed_domain:
        st.subheader(f"Ask questions about {st.session_state.indexed_domain}")
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)

        user_question = st.chat_input("Ask a question")
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(st.session_state.indexed_domain, user_question)
                    st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))


def run_cli() -> None:
    ensure_env()
    parser = argparse.ArgumentParser(description="Domain-Specific Q&A Chatbot")
    parser.add_argument("--domain", type=str, help="Domain name (e.g., Healthcare, AI)")
    args = parser.parse_args()

    domain = args.domain or input("Enter a domain: ").strip()
    if not domain:
        raise ValueError("Domain is required.")

    print(f"[domain] {domain}")
    print("[index] starting scrape and indexing...")
    index_domain(domain)
    chat_loop(domain)


if __name__ == "__main__":
    try:
        try:
            from streamlit.runtime import exists

            is_streamlit = exists()
        except Exception:
            is_streamlit = bool(getattr(st, "_is_running_with_streamlit", False))

        if is_streamlit:
            run_streamlit()
        else:
            run_cli()
    except KeyboardInterrupt:
        sys.exit(130)
