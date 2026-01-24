## Personal Context Memory Agent

Streamlit app that stores high-signal user context in a local JSON file
and injects relevant memory into responses.

### Features
- JSON memory store (`memory.json`) for preferences, writing style, decisions
- Relevance-based memory injection
- Manual chat flow with memory viewer and clear-memory confirmation

### Run
```
pip install -r requirements.txt
streamlit run app.py
```

### Notes
- Replace `call_llm()` in `app.py` with your preferred LLM provider.
- Memory is stored in `memory.json` in the same folder.
