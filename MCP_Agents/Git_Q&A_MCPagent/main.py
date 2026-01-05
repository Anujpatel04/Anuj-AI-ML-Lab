#!/usr/bin/env python3
"""
Codebase Q&A MCP Agent - Main Entry Point

A production-grade Codebase Q&A Agent using Model Context Protocol (MCP)
that analyzes Git repositories and answers architecture and code-level questions.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ingestion.repo_scanner import RepoScanner
from parsing.ast_parser import ASTParser
from embeddings.chunker import CodeChunker
from embeddings.embedder import CodeEmbedder
from retrieval.vector_store import VectorStore
from mcp_context.context_manager import MCPContextManager
from qa.qa_engine import QAEngine
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, Settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodebaseQAAgent:
    """Main Codebase Q&A Agent class."""
    
    def __init__(self, repo_path: str, index_path: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            repo_path: Path to Git repository
            index_path: Path to save/load index (optional)
        """
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
        # Get embedding dimension from embedder
        embedding_dim = self.embedder.get_dimension()
        self.vector_store = VectorStore(dimension=embedding_dim, index_path=self.index_path)
        self.context_manager = MCPContextManager(context_path=Settings.CONTEXT_PATH)
        self.qa_engine = QAEngine(
            api_key=Settings.get_api_key(),
            base_url=Settings.get_base_url(),
            model=Settings.QA_MODEL
        )
    
    def index_repository(self, force_reindex: bool = False) -> None:
        """
        Index the repository: scan, parse, chunk, and embed.
        
        Args:
            force_reindex: Force reindexing even if index exists
        """
        logger.info("Starting repository indexing...")
        
        # Check if index exists
        if not force_reindex and Path(self.index_path).exists():
            logger.info(f"Index already exists at {self.index_path}. Use --force to reindex.")
            return
        
        # Step 1: Scan repository
        logger.info("Step 1: Scanning repository...")
        files = self.scanner.scan()
        repo_metadata = self.scanner.get_repo_metadata()
        repo_metadata['total_files'] = len(files)
        
        # Count languages
        languages = {}
        for file in files:
            lang = file.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        repo_metadata['languages'] = languages
        
        self.context_manager.set_repo_metadata(repo_metadata)
        logger.info(f"Found {len(files)} files")
        
        # Step 2: Parse files and extract AST
        logger.info("Step 2: Parsing files and extracting AST...")
        all_chunks = []
        ast_summaries = {}
        
        for i, file_info in enumerate(files, 1):
            if i % 10 == 0:
                logger.info(f"Parsed {i}/{len(files)} files...")
            
            file_path = file_info['full_path']
            language = file_info.get('language')
            
            if not language:
                continue
            
            try:
                # Parse AST
                ast_data = self.parser.parse_file(file_path, language=language)
                
                # Store AST summary
                ast_summaries[file_info['path']] = {
                    'functions': len(ast_data.get('functions', [])),
                    'classes': len(ast_data.get('classes', [])),
                    'imports': len(ast_data.get('imports', []))
                }
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Chunk code
                chunks = self.chunker.chunk_code(content, file_info['path'], ast_data)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        # Store AST summaries in context
        for file_path, summary in ast_summaries.items():
            self.context_manager.add_ast_summary(file_path, summary)
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(files)} files")
        
        # Step 3: Generate embeddings
        logger.info("Step 3: Generating embeddings...")
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embedded_batch = self.embedder.embed_chunks(batch)
            self.vector_store.add_chunks(embedded_batch)
            logger.info(f"Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")
        
        # Step 4: Save index
        logger.info("Step 4: Saving index...")
        self.vector_store.save()
        
        # Generate architecture overview
        architecture = {
            'total_files': len(files),
            'languages': languages,
            'total_chunks': len(all_chunks),
            'index_size': self.vector_store.index.ntotal
        }
        self.context_manager.set_architecture_overview(architecture)
        
        logger.info("Repository indexing complete!")
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(files)} files")
    
    def answer_question(self, query: str) -> dict:
        """
        Answer a question about the codebase.
        
        Args:
            query: User question
        
        Returns:
            Answer dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing question: {query}")
        
        # Check if index exists
        if not Path(self.index_path).exists():
            logger.error("Index not found. Please run indexing first.")
            return {
                'answer': "Repository not indexed. Please run indexing first.",
                'sources': [],
                'error': 'index_not_found'
            }
        
        # Load index if not already loaded
        if self.vector_store.index.ntotal == 0:
            try:
                self.vector_store.load(self.index_path)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                return {
                    'answer': f"Error loading index: {e}",
                    'sources': [],
                    'error': 'index_load_error'
                }
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=Settings.TOP_K)
        
        # Get context summary
        context_summary = self.context_manager.get_context_summary()
        
        # Generate answer
        answer_data = self.qa_engine.answer_question(query, retrieved_chunks, context_summary)
        
        # Store conversation turn
        context_used = [chunk.get('file_path', '') for chunk in retrieved_chunks]
        self.context_manager.add_conversation_turn(query, answer_data['answer'], context_used)
        
        return answer_data
    
    def interactive_mode(self) -> None:
        """Run in interactive Q&A mode."""
        print("\n" + "="*60)
        print("Codebase Q&A Agent - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to exit\n")
        
        while True:
            try:
                query = input("Question: ").strip()
                
                if query.lower() in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                answer_data = self.answer_question(query)
                
                print("\n" + "-"*60)
                print("Answer:")
                print(answer_data['answer'])
                print("\nSources:")
                for source in answer_data.get('sources', [])[:3]:
                    print(f"  - {source['file_path']}:{source['start_line']}-{source['end_line']} (similarity: {source['similarity']:.2f})")
                print("-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Codebase Q&A MCP Agent - Analyze Git repositories and answer questions"
    )
    parser.add_argument(
        'repo_path',
        type=str,
        help='Path to Git repository'
    )
    parser.add_argument(
        '--index',
        action='store_true',
        help='Index the repository'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reindexing'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Ask a single question (non-interactive)'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default=None,
        help='Path to save/load index'
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not Settings.get_api_key():
        logger.error("API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = CodebaseQAAgent(args.repo_path, index_path=args.index_path)
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Index if requested
    if args.index:
        try:
            agent.index_repository(force_reindex=args.force)
        except Exception as e:
            logger.error(f"Error indexing repository: {e}")
            sys.exit(1)
    
    # Answer question or start interactive mode
    if args.query:
        answer_data = agent.answer_question(args.query)
        print("\nAnswer:")
        print(answer_data['answer'])
        print("\nSources:")
        for source in answer_data.get('sources', []):
            print(f"  - {source['file_path']}:{source['start_line']}-{source['end_line']}")
    else:
        agent.interactive_mode()


if __name__ == "__main__":
    main()

