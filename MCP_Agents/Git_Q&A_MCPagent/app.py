"""
Streamlit Frontend for Codebase Q&A MCP Agent
"""

import streamlit as st
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from config.settings import Settings

# Import agent class from main
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import the agent class - we'll define it inline to avoid circular imports
from ingestion.repo_scanner import RepoScanner
from parsing.ast_parser import ASTParser
from embeddings.chunker import CodeChunker
from embeddings.embedder import CodeEmbedder
from retrieval.vector_store import VectorStore
from mcp_context.context_manager import MCPContextManager
from qa.qa_engine import QAEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Codebase Q&A Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-item {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'repo_path' not in st.session_state:
    st.session_state.repo_path = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None


def clone_github_repo(repo_url: str) -> Optional[str]:
    """
    Clone a GitHub repository to a temporary directory.
    
    Args:
        repo_url: GitHub repository URL
    
    Returns:
        Path to cloned repository or None if failed
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="codebase_qa_")
        st.session_state.temp_dir = temp_dir
        
        # Extract repo name from URL
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        clone_path = os.path.join(temp_dir, repo_name)
        
        # Clone repository
        with st.spinner(f"Cloning repository from {repo_url}..."):
            result = subprocess.run(
                ['git', 'clone', repo_url, clone_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                st.error(f"Failed to clone repository: {result.stderr}")
                return None
        
        st.success(f"Repository cloned successfully!")
        return clone_path
    
    except subprocess.TimeoutExpired:
        st.error("Repository cloning timed out. The repository might be too large.")
        return None
    except Exception as e:
        st.error(f"Error cloning repository: {str(e)}")
        return None


def initialize_agent(repo_path: str) -> bool:
    """Initialize the Codebase Q&A Agent."""
    try:
        if not Settings.get_api_key():
            st.error("API key not found. Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env file.")
            return False
        
        st.session_state.agent = CodebaseQAAgent(repo_path)
        st.session_state.repo_path = repo_path
        return True
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return False


def index_repository(force: bool = False):
    """Index the repository."""
    if not st.session_state.agent:
        st.error("Agent not initialized. Please connect to a repository first.")
        return
    
    try:
        with st.spinner("Indexing repository... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # This is a simplified progress - in production, you'd want to hook into actual progress
            status_text.text("Scanning repository...")
            progress_bar.progress(10)
            
            st.session_state.agent.index_repository(force_reindex=force)
            
            progress_bar.progress(100)
            status_text.text("Indexing complete!")
            
            st.session_state.indexed = True
            st.success("Repository indexed successfully! You can now ask questions.")
            st.rerun()  # Refresh to update UI
            
    except Exception as e:
        st.error(f"Error indexing repository: {str(e)}")


def answer_question(query: str) -> Dict[str, Any]:
    """Answer a question about the codebase."""
    if not st.session_state.agent:
        return {'answer': 'Agent not initialized.', 'sources': []}
    
    # Check if index exists
    index_path = Path(Settings.INDEX_PATH)
    if not index_path.exists() and not st.session_state.indexed:
        return {'answer': 'Repository not indexed. Please index first.', 'sources': []}
    
    # Set indexed flag if index exists
    if index_path.exists() and not st.session_state.indexed:
        st.session_state.indexed = True
    
    try:
        answer_data = st.session_state.agent.answer_question(query)
        return answer_data
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error answering question: {error_details}")
        return {'answer': f'Error: {str(e)}\n\nDetails: {error_details}', 'sources': []}


# Import the agent class
class CodebaseQAAgent:
    """Main Codebase Q&A Agent class."""
    
    def __init__(self, repo_path: str, index_path: Optional[str] = None):
        """Initialize the agent."""
        self.repo_path = Path(repo_path).resolve()
        self.index_path = index_path or Settings.INDEX_PATH
        
        # Initialize components
        self.scanner = RepoScanner(str(self.repo_path))
        self.parser = ASTParser()
        self.chunker = CodeChunker(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP
        )
        self.embedder = CodeEmbedder(
            model_name=Settings.EMBEDDING_MODEL,
            use_openai=Settings.USE_OPENAI_EMBEDDINGS,
            api_key=Settings.get_api_key() if Settings.USE_OPENAI_EMBEDDINGS else None
        )
        embedding_dim = self.embedder.get_dimension()
        self.vector_store = VectorStore(dimension=embedding_dim, index_path=self.index_path)
        self.context_manager = MCPContextManager(context_path=Settings.CONTEXT_PATH)
        self.qa_engine = QAEngine(
            api_key=Settings.get_api_key(),
            base_url=Settings.get_base_url(),
            model=Settings.QA_MODEL
        )
    
    def index_repository(self, force_reindex: bool = False) -> None:
        """Index the repository."""
        logger.info("Starting repository indexing...")
        
        if not force_reindex and Path(self.index_path).exists():
            logger.info(f"Index already exists at {self.index_path}. Use --force to reindex.")
            return
        
        # Scan repository
        files = self.scanner.scan()
        repo_metadata = self.scanner.get_repo_metadata()
        repo_metadata['total_files'] = len(files)
        
        languages = {}
        for file in files:
            lang = file.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        repo_metadata['languages'] = languages
        
        self.context_manager.set_repo_metadata(repo_metadata)
        
        # Parse files
        all_chunks = []
        ast_summaries = {}
        
        for i, file_info in enumerate(files, 1):
            file_path = file_info['full_path']
            language = file_info.get('language')
            
            if not language:
                continue
            
            try:
                ast_data = self.parser.parse_file(file_path, language=language)
                ast_summaries[file_info['path']] = {
                    'functions': len(ast_data.get('functions', [])),
                    'classes': len(ast_data.get('classes', [])),
                    'imports': len(ast_data.get('imports', []))
                }
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                chunks = self.chunker.chunk_code(content, file_info['path'], ast_data)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        for file_path, summary in ast_summaries.items():
            self.context_manager.add_ast_summary(file_path, summary)
        
        # Generate embeddings
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embedded_batch = self.embedder.embed_chunks(batch)
            self.vector_store.add_chunks(embedded_batch)
        
        # Save index
        self.vector_store.save()
        
        architecture = {
            'total_files': len(files),
            'languages': languages,
            'total_chunks': len(all_chunks),
            'index_size': self.vector_store.index.ntotal
        }
        self.context_manager.set_architecture_overview(architecture)
    
    def answer_question(self, query: str) -> dict:
        """Answer a question about the codebase."""
        if not Path(self.index_path).exists():
            return {
                'answer': "Repository not indexed. Please run indexing first.",
                'sources': [],
                'error': 'index_not_found'
            }
        
        if self.vector_store.index.ntotal == 0:
            try:
                self.vector_store.load(self.index_path)
            except Exception as e:
                return {
                    'answer': f"Error loading index: {e}",
                    'sources': [],
                    'error': 'index_load_error'
                }
        
        query_embedding = self.embedder.embed_query(query)
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=Settings.TOP_K)
        context_summary = self.context_manager.get_context_summary()
        
        answer_data = self.qa_engine.answer_question(query, retrieved_chunks, context_summary)
        
        context_used = [chunk.get('file_path', '') for chunk in retrieved_chunks]
        self.context_manager.add_conversation_turn(query, answer_data['answer'], context_used)
        
        return answer_data


# Main UI
st.markdown('<h1 class="main-header">💬 Codebase Q&A Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about any GitHub repository</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key check
    api_key = Settings.get_api_key()
    if api_key:
        st.success("✓ API Key configured")
    else:
        st.error("⚠️ API Key not found")
        st.info("Set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env file")
    
    st.divider()
    
    st.header("📊 Repository Status")
    if st.session_state.repo_path:
        st.info(f"**Connected:**\n{st.session_state.repo_path}")
    else:
        st.info("No repository connected")
    
    # Check if index actually exists
    index_path = Path(Settings.INDEX_PATH)
    index_exists = index_path.exists()
    
    if st.session_state.indexed or index_exists:
        if index_exists:
            st.success("✓ Repository indexed")
            # Update flag if index exists
            st.session_state.indexed = True
        else:
            st.warning("⚠️ Index flag set but file missing")
    else:
        st.warning("⚠️ Repository not indexed")
    
    st.divider()
    
    # Clear session
    if st.button("🔄 Clear Session", use_container_width=True):
        # Cleanup temp directory
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
        
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.chat_history = []
        st.rerun()

# Main content
tab1, tab2 = st.tabs(["🔗 Connect Repository", "💬 Ask Questions"])

with tab1:
    st.header("Connect to GitHub Repository")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repo",
            help="Enter the full GitHub repository URL"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        connect_button = st.button("🔗 Connect", use_container_width=True)
    
    # Local path option
    st.divider()
    st.subheader("Or use local repository")
    
    local_path = st.text_input(
        "Local Repository Path",
        placeholder="/path/to/local/repo",
        help="Enter the path to a local Git repository"
    )
    
    local_connect = st.button("📁 Connect Local", use_container_width=True)
    
    # Handle connections
    if connect_button and repo_url:
        if not repo_url.startswith(('http://', 'https://', 'git@')):
            st.error("Please enter a valid GitHub URL")
        else:
            # Clone repository
            cloned_path = clone_github_repo(repo_url)
            if cloned_path:
                if initialize_agent(cloned_path):
                    st.success("Repository connected! Go to 'Ask Questions' tab to index and ask questions.")
    
    if local_connect and local_path:
        if os.path.exists(local_path):
            if initialize_agent(local_path):
                st.success("Repository connected! Go to 'Ask Questions' tab to index and ask questions.")
        else:
            st.error("Path does not exist. Please check the path.")
    
    # Indexing section
    if st.session_state.agent:
        st.divider()
        st.subheader("Index Repository")
        st.info("Index the repository to enable question answering. This may take a few minutes.")
        
        col1, col2 = st.columns(2)
        with col1:
            index_button = st.button("📚 Index Repository", use_container_width=True, type="primary")
        with col2:
            force_index = st.button("🔄 Re-index", use_container_width=True)
        
        if index_button:
            index_repository(force=False)
        
        if force_index:
            index_repository(force=True)

with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.agent:
        st.warning("⚠️ Please connect to a repository first in the 'Connect Repository' tab.")
        st.stop()
    
    # Check if index exists
    index_exists = False
    if st.session_state.agent:
        index_path = Path(Settings.INDEX_PATH)
        index_exists = index_path.exists()
    
    if not st.session_state.indexed and not index_exists:
        st.warning("⚠️ Repository not indexed. Please index the repository first.")
        if st.button("📚 Index Now", use_container_width=True):
            index_repository(force=False)
        st.stop()
    elif index_exists and not st.session_state.indexed:
        # Index exists but flag not set - set it
        st.session_state.indexed = True
    
    # Chat interface
    st.subheader("💬 Chat with your codebase")
    
    # Display chat history
    for i, (role, message) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            if role == "assistant":
                # For assistant messages, check if it's a dict with answer and sources
                if isinstance(message, dict):
                    answer = message.get('answer', '')
                    sources = message.get('sources', [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Sources", expanded=False):
                            for source in sources:
                                file_path = source.get('file_path', 'unknown')
                                start_line = source.get('start_line', 0)
                                end_line = source.get('end_line', 0)
                                similarity = source.get('similarity', 0.0)
                                st.markdown(f"**{file_path}** (Lines {start_line}-{end_line}) - Similarity: {similarity:.2%}")
                else:
                    st.write(message)
            else:
                st.write(message)
    
    # Question input
    query = st.chat_input("Ask a question about the codebase...")
    
    # Handle example question buttons
    if 'example_question' in st.session_state:
        query = st.session_state.example_question
        del st.session_state.example_question
    
    if query:
        # Add user message to history
        st.session_state.chat_history.append(("user", query))
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    logger.info(f"Processing question: {query}")
                    answer_data = answer_question(query)
                    answer = answer_data.get('answer', 'No answer generated.')
                    sources = answer_data.get('sources', [])
                    
                    logger.info(f"Got answer, sources count: {len(sources)}")
                    
                    # Display answer
                    if answer:
                        st.markdown(answer)
                    else:
                        st.warning("No answer generated. Please try rephrasing your question.")
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
                            for source in sources:
                                file_path = source.get('file_path', 'unknown')
                                start_line = source.get('start_line', 0)
                                end_line = source.get('end_line', 0)
                                similarity = source.get('similarity', 0.0)
                                st.markdown(f"**{file_path}** (Lines {start_line}-{end_line}) - Similarity: {similarity:.2%}")
                    else:
                        st.info("No sources found for this question.")
                    
                    # Add assistant message to history with full data
                    st.session_state.chat_history.append(("assistant", answer_data))
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error in chat interface: {error_details}")
                    error_msg = f"Error answering question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", {'answer': error_msg, 'sources': []}))
        
        st.rerun()
    
    # Example questions
    st.divider()
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "Where is authentication implemented?",
        "How does the request flow work?",
        "Which files handle database connections?",
        "What is the main entry point?",
        "Where are API endpoints defined?"
    ]
    
    cols = st.columns(len(example_questions))
    for i, question in enumerate(example_questions):
        with cols[i]:
            if st.button(question, use_container_width=True, key=f"example_{i}"):
                st.session_state.example_question = question
                st.rerun()

# Cleanup on exit
if st.session_state.temp_dir and not st.session_state.repo_path:
    if os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass

