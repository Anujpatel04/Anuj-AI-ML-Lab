# Prompt Optimization Agent

A CLI-based agent that takes a weak user prompt and returns a clear, structured, and LLM-optimized version using the OpenAI API. It also reports detected intent and issues in the original prompt.

## What it does

- Accepts a raw prompt string
- Infers intent, task type, and missing context
- Produces a higher-quality prompt with constraints and output format
- Returns results in a strict JSON schema

## How to run

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Optional: set a model (default is `gpt-4o-mini`):

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

Run the CLI:

```bash
python app.py --prompt "write a summary about AI"
```

Or pipe input:

```bash
echo "help me write a product description" | python app.py
```

Run the Streamlit UI:

```bash
streamlit run app.py
```

## Example input/output

Input:
```
make it better
```

Output:
```json
{
  "original_prompt": "make it better",
  "detected_intent": "Improve an existing text with unspecified requirements.",
  "issues_found": [
    "No context or source text provided",
    "No target audience or tone specified",
    "No output format defined"
  ],
  "optimized_prompt": "You are an editor. Improve the following text for clarity and professionalism. Preserve meaning and key details. Keep the length within 10% of the original. Output only the revised text.\n\nText:\n<PASTE TEXT HERE>",
  "optimization_notes": "Added missing context requirements, role, constraints, and output format."
}
```

## Design decisions

- **Single OpenAI call** for deterministic, low-latency optimization
- **Strict JSON validation** to guarantee schema compliance
- **Single file** for easy portability and setup
- **CLI + Streamlit** for terminal and UI usage
- **Environment-based config** for API key and model selection
