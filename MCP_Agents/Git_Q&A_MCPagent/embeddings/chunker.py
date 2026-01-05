"""Code chunking module for intelligently splitting code into chunks."""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CodeChunker:
    """Chunks code intelligently at function/class level when possible."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the code chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_code(self, content: str, file_path: str, ast_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk code intelligently based on AST structure.
        
        Args:
            content: Full file content
            file_path: Path to the file
            ast_data: Parsed AST data from ASTParser
        
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        # Strategy 1: Function-level chunking (preferred)
        functions = ast_data.get('functions', [])
        classes = ast_data.get('classes', [])
        
        if functions or classes:
            chunks.extend(self._chunk_by_structure(content, file_path, functions, classes))
        
        # Strategy 2: If no structure found or remaining content, use line-based chunking
        if not chunks or len(content) > sum(len(c.get('content', '')) for c in chunks):
            remaining_chunks = self._chunk_by_lines(content, file_path, ast_data)
            chunks.extend(remaining_chunks)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = f"{file_path}:{i}"
            chunk['file_path'] = file_path
            chunk['language'] = ast_data.get('language')
        
        return chunks
    
    def _chunk_by_structure(self, content: str, file_path: str, 
                           functions: List[Dict], classes: List[Dict]) -> List[Dict]:
        """Chunk code by function/class boundaries."""
        chunks = []
        lines = content.split('\n')
        
        # Create a map of line numbers to structures
        structure_map = {}
        for func in functions:
            line = func.get('line', 0)
            structure_map[line] = {'type': 'function', 'data': func}
        
        for cls in classes:
            line = cls.get('line', 0)
            structure_map[line] = {'type': 'class', 'data': cls}
        
        # Extract chunks for each structure
        sorted_lines = sorted(structure_map.keys())
        
        for i, start_line in enumerate(sorted_lines):
            end_line = sorted_lines[i + 1] - 1 if i + 1 < len(sorted_lines) else len(lines)
            
            structure = structure_map[start_line]
            struct_data = structure['data']
            
            # Extract the structure's content
            struct_lines = lines[start_line - 1:end_line]
            struct_content = '\n'.join(struct_lines)
            
            if len(struct_content) > self.chunk_size:
                # If structure is too large, split it further
                sub_chunks = self._split_large_content(struct_content, start_line)
                chunks.extend(sub_chunks)
            else:
                chunk = {
                    'content': struct_content,
                    'start_line': start_line,
                    'end_line': end_line,
                    'type': structure['type'],
                    'name': struct_data.get('name', 'unknown'),
                    'signature': struct_data.get('signature', '')
                }
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_lines(self, content: str, file_path: str, ast_data: Dict) -> List[Dict]:
        """Fallback: chunk by lines with overlap."""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'start_line': start_line,
                    'end_line': i - 1,
                    'type': 'text',
                    'name': None
                })
                
                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                start_line = i - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'start_line': start_line,
                'end_line': len(lines),
                'type': 'text',
                'name': None
            })
        
        return chunks
    
    def _split_large_content(self, content: str, start_line: int) -> List[Dict]:
        """Split content that's too large into smaller chunks."""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        chunk_start = start_line
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1
            
            if current_size + line_size > self.chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'start_line': chunk_start,
                    'end_line': chunk_start + len(current_chunk) - 1,
                    'type': 'text',
                    'name': None
                })
                
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                chunk_start = chunk_start + len(current_chunk) - len(overlap_lines) - 1
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'start_line': chunk_start,
                'end_line': chunk_start + len(current_chunk) - 1,
                'type': 'text',
                'name': None
            })
        
        return chunks
    
    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get overlap lines from the end of current chunk."""
        num_overlap_lines = max(1, self.chunk_overlap // 50)  # Approximate lines
        return lines[-num_overlap_lines:] if len(lines) > num_overlap_lines else lines

