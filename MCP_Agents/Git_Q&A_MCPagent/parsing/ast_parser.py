"""AST parser using Tree-sitter to extract code structures."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available. Install with: pip install tree-sitter")

logger = logging.getLogger(__name__)


class ASTParser:
    """Parses code files using Tree-sitter to extract AST structures."""
    
    SUPPORTED_LANGUAGES = {
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 
        'go', 'rust', 'ruby', 'php'
    }
    
    def __init__(self):
        """Initialize the AST parser."""
        self.parsers: Dict[str, Parser] = {}
        self._init_parsers()
    
    def _init_parsers(self) -> None:
        """Initialize Tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available. AST parsing will be limited.")
            return
        
        # For production, you would build language grammars here
        # This is a simplified version that works with basic parsing
        # In a full implementation, you'd need to compile language grammars
        logger.info("AST parser initialized (basic mode)")
    
    def parse_file(self, file_path: str, language: Optional[str] = None, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse a code file and extract AST structures.
        
        Args:
            file_path: Path to the file
            language: Programming language (auto-detected if None)
            content: File content (read from file if None)
        
        Returns:
            Dictionary containing:
            - functions: List of function definitions
            - classes: List of class definitions
            - imports: List of import statements
            - symbols: List of all symbols
        """
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return self._empty_result()
        
        if language is None:
            language = self._detect_language(file_path)
        
        if language not in self.SUPPORTED_LANGUAGES:
            # Fallback to simple text parsing
            return self._simple_parse(content, file_path, language)
        
        # Try Tree-sitter parsing if available
        if TREE_SITTER_AVAILABLE:
            try:
                return self._tree_sitter_parse(content, file_path, language)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")
                return self._simple_parse(content, file_path, language)
        
        return self._simple_parse(content, file_path, language)
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
        }
        return ext_to_lang.get(Path(file_path).suffix.lower())
    
    def _tree_sitter_parse(self, content: str, file_path: str, language: str) -> Dict[str, Any]:
        """Parse using Tree-sitter (placeholder - requires compiled grammars)."""
        # In production, this would use actual Tree-sitter parsing
        # For now, fall back to simple parsing
        return self._simple_parse(content, file_path, language)
    
    def _simple_parse(self, content: str, file_path: str, language: Optional[str]) -> Dict[str, Any]:
        """Simple regex-based parsing as fallback."""
        functions = []
        classes = []
        imports = []
        symbols = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Extract imports
            if language == 'python':
                if stripped.startswith('import ') or stripped.startswith('from '):
                    imports.append({
                        'statement': stripped,
                        'line': i,
                        'file': file_path
                    })
                
                # Extract class definitions
                if stripped.startswith('class '):
                    class_name = stripped.split('(')[0].replace('class ', '').strip()
                    classes.append({
                        'name': class_name,
                        'line': i,
                        'file': file_path,
                        'signature': stripped
                    })
                
                # Extract function definitions
                if stripped.startswith('def '):
                    func_name = stripped.split('(')[0].replace('def ', '').strip()
                    functions.append({
                        'name': func_name,
                        'line': i,
                        'file': file_path,
                        'signature': stripped
                    })
            
            elif language in ('javascript', 'typescript'):
                # Extract imports
                if stripped.startswith('import ') or stripped.startswith('export '):
                    imports.append({
                        'statement': stripped,
                        'line': i,
                        'file': file_path
                    })
                
                # Extract class definitions
                if 'class ' in stripped and stripped.split('class ')[0].strip() == '':
                    parts = stripped.split('class ')
                    if len(parts) > 1:
                        class_name = parts[1].split(' ')[0].split('{')[0].strip()
                        classes.append({
                            'name': class_name,
                            'line': i,
                            'file': file_path,
                            'signature': stripped
                        })
                
                # Extract function definitions
                if 'function ' in stripped or (stripped.startswith('const ') and '=' in stripped and '=>' in stripped):
                    if 'function ' in stripped:
                        func_name = stripped.split('function ')[1].split('(')[0].strip()
                    else:
                        func_name = stripped.split('=')[0].replace('const ', '').replace('let ', '').replace('var ', '').strip()
                    
                    functions.append({
                        'name': func_name,
                        'line': i,
                        'file': file_path,
                        'signature': stripped[:100]  # Limit signature length
                    })
        
        # Build symbols list
        for cls in classes:
            symbols.append({
                'name': cls['name'],
                'type': 'class',
                'line': cls['line'],
                'file': file_path
            })
        
        for func in functions:
            symbols.append({
                'name': func['name'],
                'type': 'function',
                'line': func['line'],
                'file': file_path
            })
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'symbols': symbols,
            'file_path': file_path,
            'language': language
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty parsing result."""
        return {
            'functions': [],
            'classes': [],
            'imports': [],
            'symbols': [],
            'file_path': '',
            'language': None
        }





