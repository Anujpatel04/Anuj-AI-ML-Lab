from firecrawl import FirecrawlApp
import streamlit as st
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.openai import OpenAIChat

# Load environment variables from .env file in root directory
env_path = Path('/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env')
if not env_path.exists():
    root_dir = Path(__file__).parent.parent.parent
    env_path = root_dir / '.env'

if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

# Get DeepSeek API key from environment
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# Set Firecrawl API key (hardcoded)
FIRECRAWL_API_KEY = "fc-1f2b3a8e651549c0bfa02a1a53e2aad3"

# Configure DeepSeek API (OpenAI-compatible)
if DEEPSEEK_API_KEY:
    os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

# Set page configuration
st.set_page_config(
    page_title="AI Startup Insight Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Startup Insight Agent</h1>
    <p>Powered by Firecrawl FIRE-1 & DeepSeek AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.markdown("""
    This professional tool extracts comprehensive company information from websites 
    using Firecrawl's FIRE-1 agent and provides AI-powered business analysis.
    """)
    
    st.markdown("---")
    st.markdown("### 🔄 How It Works")
    st.markdown("""
    1. **🔍 FIRE-1 Agent** - Extracts structured data from websites
    2. **🧠 DeepSeek AI** - Analyzes data for business insights
    3. **📊 Results** - Professional presentation of findings
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    if DEEPSEEK_API_KEY:
        st.success("✅ DeepSeek API: Configured")
    else:
        st.error("❌ DeepSeek API: Not found in .env")
    
    st.success("✅ Firecrawl API: Configured")
    
    st.markdown("---")
    st.markdown("### 📚 Features")
    st.markdown("""
    - Advanced web extraction
    - AI-powered analysis
    - Multi-URL processing
    - Structured data output
    """)
    

# Show warning if DeepSeek API key is not found
if not DEEPSEEK_API_KEY:
    st.error("⚠️ **DEEPSEEK_API_KEY not found in .env file.** Please add it to your .env file in the root directory.")

# Main content
st.markdown("## 🔥 Firecrawl FIRE-1 Agent Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.info("**🌐 Advanced Web Extraction**\n\nFirecrawl's FIRE-1 agent intelligently navigates websites to extract structured data from complex layouts and dynamic content.")
    
    st.success("**🖱️ Interactive Navigation**\n\nThe agent interacts with buttons, links, input fields, and dynamic elements to access comprehensive information.")

with col2:
    st.warning("**📄 Multi-page Processing**\n\nFIRE handles pagination and multi-step processes, gathering data across entire websites.")
    
    st.info("**📊 Intelligent Data Structuring**\n\nThe agent automatically structures extracted information according to your specified schema for immediate use.")

st.markdown("---")

# Input section
st.markdown("## 🌐 Enter Website URLs")
st.markdown("Provide one or more company website URLs (one per line) to extract comprehensive information.")

website_urls = st.text_area(
    "Website URLs (one per line)", 
    placeholder="https://example.com\nhttps://another-company.com",
    height=120,
    help="Enter one or more website URLs, each on a new line"
)

# Define a JSON schema directly without Pydantic
extraction_schema = {
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "The official name of the company or startup"
        },
        "company_description": {
            "type": "string",
            "description": "A description of what the company does and its value proposition"
        },
        "company_mission": {
            "type": "string",
            "description": "The company's mission statement or purpose"
        },
        "product_features": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Key features or capabilities of the company's products/services"
        },
        "contact_phone": {
            "type": "string",
            "description": "Company's contact phone number if available"
        }
    },
    "required": ["company_name", "company_description", "product_features"]
}



# Start extraction when button is clicked
if st.button("🚀 Start Analysis", type="primary"):
    if not DEEPSEEK_API_KEY:
        st.error("⚠️ Please configure DEEPSEEK_API_KEY in your .env file before proceeding.")
    elif not website_urls.strip():
        st.error("❌ Please enter at least one website URL")
    else:
        try:
            with st.spinner("🔍 Initializing Firecrawl agent..."):
                # Initialize the FirecrawlApp with the API key
                app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
                
                # Parse the input URLs more robustly
                # Split by newline, strip whitespace from each line, and filter out empty lines
                urls = [url.strip() for url in website_urls.split('\n') if url.strip()]
                
                # Debug: Show the parsed URLs
                st.info(f"Attempting to process these URLs: {urls}")
                
                if not urls:
                    st.error("❌ No valid URLs found after parsing. Please check your input.")
                else:
                    # Create tabs for each URL
                    tabs = st.tabs([f"Website {i+1}: {url[:50]}..." if len(url) > 50 else f"Website {i+1}: {url}" for i, url in enumerate(urls)])
                    
                    # Initialize the Agno agent once (outside the loop)
                    # Patch to convert developer role to system role for DeepSeek compatibility
                    try:
                        from agno.models.openai.chat import OpenAIChat as OpenAIChatClass
                        original_format_message = OpenAIChatClass._format_message
                        
                        def patched_format_message(self, message):
                            """Patch to convert developer role to system role for DeepSeek compatibility"""
                            formatted = original_format_message(self, message)
                            if isinstance(formatted, dict):
                                if formatted.get('role') == 'developer':
                                    formatted['role'] = 'system'
                            elif hasattr(formatted, 'role') and formatted.role == 'developer':
                                formatted.role = 'system'
                            return formatted
                        
                        OpenAIChatClass._format_message = patched_format_message
                    except Exception:
                        pass
                    
                        agno_agent = Agent(
                        model=OpenAIChat(
                            id="deepseek-chat",
                            api_key=DEEPSEEK_API_KEY,
                            base_url="https://api.deepseek.com"
                        ),
                            instructions="""You are an expert business analyst who provides concise, insightful summaries of companies.
                            You will be given structured data about a company including its name, description, mission, and product features.
                            Your task is to analyze this information and provide a brief, compelling summary that highlights:
                            1. What makes this company unique or innovative
                            2. The core value proposition for customers
                            3. The potential market impact or growth opportunities
                            
                            Keep your response under 150 words, be specific, and focus on actionable insights.
                            """,
                            markdown=True
                        )
                    
                    # Process each URL one at a time
                    for i, (url, tab) in enumerate(zip(urls, tabs)):
                        with tab:
                            st.markdown(f"### 🔍 Analyzing: {url}")
                            st.markdown("---")
                            
                            with st.spinner(f"FIRE agent is extracting information from {url}..."):
                                try:
                                    # Extract data for this single URL
                                    # Firecrawl extract method takes parameters directly, not in a params dict
                                    extraction_prompt = '''
Analyze this company website thoroughly and extract comprehensive information.

1. Company Information:
   - Identify the official company name
     Explain: This is the legal name the company operates under.
   - Extract a detailed yet concise description of what the company does
   - Find the company's mission statement or purpose
     Explain: What problem is the company trying to solve? How do they aim to make a difference?

2. Product/Service Information:
   - Identify 3-5 specific product features or service offerings
     Explain: What are the key things their product or service can do? Describe as if explaining to a non-expert.
   - Focus on concrete capabilities rather than marketing claims
     Explain: What does the product actually do, in simple terms, rather than how it's advertised?
   - Be specific about what the product/service actually does
     Explain: Give examples of how a customer might use this product or service in their daily life.

3. Contact Information:
   - Find direct contact methods (phone numbers)
     Explain: How can a potential customer reach out to speak with someone at the company?
   - Only extract contact information that is explicitly provided
     Explain: We're looking for official contact details, not inferring or guessing.

Important guidelines:
- Be thorough but concise in your descriptions
- Extract factual information, not marketing language
- If information is not available, do not make assumptions
- For each piece of information, provide a brief, simple explanation of what it means and why it's important
- Include a layman's explanation of what the company does, as if explaining to someone with no prior knowledge of the industry or technology involved
'''
                                    
                                    # Call extract with parameters as keyword arguments
                                    # The extract method signature: extract(urls, prompt, schema, agent, ...)
                                    try:
                                        from firecrawl.v2.types import AgentOptions
                                        agent_options = AgentOptions(model="FIRE-1")
                                    except (ImportError, TypeError):
                                        # Fallback: use dict if AgentOptions not available or doesn't work
                                        agent_options = {"model": "FIRE-1"}
                                    
                                    data = app.extract(
                                        urls=[url],
                                        prompt=extraction_prompt,
                                        schema=extraction_schema,
                                        agent=agent_options
                                    )
                                    
                                    # Check if extraction was successful
                                    # ExtractResponse is a Pydantic model, access attributes directly
                                    if data and data.data:
                                        # Display extracted data
                                        st.subheader("📊 Extracted Information")
                                        company_data = data.data
                                        
                                        # Display company name prominently
                                        if 'company_name' in company_data and company_data['company_name']:
                                            st.markdown(f"### 🏢 {company_data['company_name']}")
                                            st.markdown("---")
                                        
                                        # Display other extracted fields in a professional format
                                        for key, value in company_data.items():
                                            if key == 'company_name':
                                                continue  # Already displayed above
                                                
                                            display_key = key.replace('_', ' ').title()
                                            
                                            if value:  # Only display if there's a value
                                                with st.container():
                                                if isinstance(value, list):
                                                        st.markdown(f"#### {display_key}")
                                                    for item in value:
                                                            st.markdown(f"• {item}")
                                                elif isinstance(value, str):
                                                        st.markdown(f"#### {display_key}")
                                                        st.markdown(f"{value}")
                                                elif isinstance(value, bool):
                                                        st.markdown(f"**{display_key}:** {'Yes' if value else 'No'}")
                                                else:
                                                        st.markdown(f"**{display_key}:** {value}")
                                                    st.markdown("---")
                                        
                                        # Process with Agno agent
                                        with st.spinner("🧠 Generating AI-powered business analysis..."):
                                            try:
                                                # Run the agent with the extracted data
                                                agent_response: RunOutput = agno_agent.run(f"Analyze this company data and provide insights: {json.dumps(company_data)}")
                                                
                                                # Display the agent's analysis in a highlighted box
                                                st.subheader("🧠 AI Business Analysis")
                                                st.markdown("---")
                                                st.markdown(agent_response.content)
                                            except Exception as e:
                                                st.warning(f"⚠️ AI analysis could not be generated: {str(e)}")
                                        
                                        # Show raw data in expander
                                        with st.expander("🔍 View Raw API Response"):
                                            # Convert Pydantic model to dict for JSON display
                                            st.json(data.model_dump() if hasattr(data, 'model_dump') else data.dict() if hasattr(data, 'dict') else str(data))
                                            
                                        # Add processing details
                                        with st.expander("ℹ️ Processing Details"):
                                            st.markdown("**FIRE Agent Actions:**")
                                            st.markdown("- 🔍 Scanned website content and structure")
                                            st.markdown("- 🖱️ Interacted with necessary page elements")
                                            st.markdown("- 📊 Extracted and structured data according to schema")
                                            st.markdown("- 🧠 Applied AI reasoning to identify relevant information")
                                            
                                            if data.status:
                                                st.markdown(f"**Status:** {data.status}")
                                            if data.expires_at:
                                                st.markdown(f"**Data Expires:** {data.expires_at}")
                                            if data.id:
                                                st.markdown(f"**Extraction ID:** {data.id}")
                                    else:
                                        st.error(f"❌ No data was extracted from {url}. The website might be inaccessible, or the content structure may not match the expected format.")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error processing {url}: {str(e)}")
                                    st.exception(e)
        except Exception as e:
            st.error(f"❌ Error during extraction: {str(e)}")
            st.exception(e)

