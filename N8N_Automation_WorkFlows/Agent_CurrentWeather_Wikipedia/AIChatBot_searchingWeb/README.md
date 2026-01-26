## Conversational AI Agent with Tool Use (n8n + LangChain)

This workflow implements a conversational agent in n8n using LangChain. It
maintains short-term memory, decides when to call external tools, and answers
queries with contextual, grounded responses. The design is modular and suitable
for experimentation and learning.

### Capabilities
- Multi-turn conversational agent with windowed memory (last 20 messages)
- Tool-augmented reasoning for factual and real-time queries
- Web search via SerpAPI and knowledge lookup via Wikipedia
- OpenAI Chat Model integration
- Fully visual n8n workflow with inline notes

### Architecture
User Message  
→ Manual Chat Trigger  
→ AI Agent (LangChain)  
   - OpenAI Chat Model  
   - Window Buffer Memory  
   - Wikipedia Tool  
   - SerpAPI Tool  
→ Final Response

### Workflow Components
1. Manual Chat Trigger  
   Entry point for user prompts and real-time input.

2. AI Agent (LangChain)  
   Core reasoning node that routes to tools or answers directly.

3. Chat OpenAI  
   Language model used for reasoning and response generation.

4. Window Buffer Memory  
   Stores the last 20 messages to keep context without long-term bloat.

5. Tools  
   - Wikipedia Tool for encyclopedic and background queries  
   - SerpAPI Tool for live web search and recent updates

### Configuration
Set credentials in n8n before running:
- OpenAI API key
- SerpAPI API key

Ensure credentials are linked to the appropriate nodes.

### Usage
1. Import the workflow JSON into n8n.
2. Configure OpenAI and SerpAPI credentials.
3. Activate the workflow.
4. Send a message via the Manual Chat Trigger.
5. Review the agent output and tool usage in the execution logs.
