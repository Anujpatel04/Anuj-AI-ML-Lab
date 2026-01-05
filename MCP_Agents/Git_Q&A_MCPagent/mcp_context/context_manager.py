"""MCP Context Manager for storing and retrieving persistent context."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class MCPContextManager:
    """Manages persistent context using MCP (Model Context Protocol)."""
    
    def __init__(self, context_path: str = ".mcp_context.json"):
        """
        Initialize the MCP context manager.
        
        Args:
            context_path: Path to store context file
        """
        self.context_path = Path(context_path)
        self.context: Dict[str, Any] = {
            'repo_metadata': {},
            'architecture_overview': {},
            'ast_summaries': {},
            'module_relationships': {},
            'conversation_history': []
        }
        self._load_context()
    
    def _load_context(self) -> None:
        """Load context from disk if it exists."""
        if self.context_path.exists():
            try:
                with open(self.context_path, 'r', encoding='utf-8') as f:
                    self.context = json.load(f)
                logger.info(f"Loaded context from {self.context_path}")
            except Exception as e:
                logger.warning(f"Error loading context: {e}")
    
    def save_context(self) -> None:
        """Save context to disk."""
        try:
            self.context_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.context_path, 'w', encoding='utf-8') as f:
                json.dump(self.context, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved context to {self.context_path}")
        except Exception as e:
            logger.error(f"Error saving context: {e}")
    
    def set_repo_metadata(self, metadata: Dict[str, Any]) -> None:
        """Store repository metadata."""
        self.context['repo_metadata'] = metadata
        self.save_context()
    
    def get_repo_metadata(self) -> Dict[str, Any]:
        """Get repository metadata."""
        return self.context.get('repo_metadata', {})
    
    def add_ast_summary(self, file_path: str, summary: Dict[str, Any]) -> None:
        """Add AST summary for a file."""
        self.context['ast_summaries'][file_path] = summary
        self.save_context()
    
    def get_ast_summaries(self) -> Dict[str, Any]:
        """Get all AST summaries."""
        return self.context.get('ast_summaries', {})
    
    def set_architecture_overview(self, overview: Dict[str, Any]) -> None:
        """Set architecture overview."""
        self.context['architecture_overview'] = overview
        self.save_context()
    
    def get_architecture_overview(self) -> Dict[str, Any]:
        """Get architecture overview."""
        return self.context.get('architecture_overview', {})
    
    def add_module_relationship(self, source: str, target: str, relationship_type: str) -> None:
        """Add a module relationship."""
        if 'module_relationships' not in self.context:
            self.context['module_relationships'] = {}
        
        if source not in self.context['module_relationships']:
            self.context['module_relationships'][source] = []
        
        self.context['module_relationships'][source].append({
            'target': target,
            'type': relationship_type
        })
        self.save_context()
    
    def get_module_relationships(self) -> Dict[str, List[Dict[str, str]]]:
        """Get all module relationships."""
        return self.context.get('module_relationships', {})
    
    def add_conversation_turn(self, query: str, answer: str, context_used: List[str]) -> None:
        """Add a conversation turn to history."""
        self.context['conversation_history'].append({
            'query': query,
            'answer': answer,
            'context_used': context_used
        })
        # Keep only last 50 turns
        if len(self.context['conversation_history']) > 50:
            self.context['conversation_history'] = self.context['conversation_history'][-50:]
        self.save_context()
    
    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history."""
        history = self.context.get('conversation_history', [])
        if last_n:
            return history[-last_n:]
        return history
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of all stored context."""
        return {
            'repo_metadata': self.get_repo_metadata(),
            'architecture_overview': self.get_architecture_overview(),
            'ast_summaries_count': len(self.get_ast_summaries()),
            'module_relationships_count': sum(len(v) for v in self.get_module_relationships().values()),
            'conversation_turns': len(self.get_conversation_history())
        }

