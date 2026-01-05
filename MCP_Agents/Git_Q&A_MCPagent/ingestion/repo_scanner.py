"""Repository scanner that recursively scans Git repositories while respecting .gitignore."""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class RepoScanner:
    """Scans a Git repository and extracts file information."""
    
    def __init__(self, repo_path: str):
        """
        Initialize the repository scanner.
        
        Args:
            repo_path: Path to the Git repository root
        """
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        self.ignored_patterns: Set[str] = set()
        self._load_gitignore()
    
    def _load_gitignore(self) -> None:
        """Load .gitignore patterns from repository root."""
        gitignore_path = self.repo_path / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.ignored_patterns.add(line)
        
        # Always ignore common patterns
        common_ignores = {
            '.git', '.gitignore', '__pycache__', '*.pyc', '.pytest_cache',
            'node_modules', '.venv', 'venv', 'env', '.env', 'dist', 'build',
            '.mypy_cache', '.ruff_cache', '.DS_Store', '*.egg-info'
        }
        self.ignored_patterns.update(common_ignores)
    
    def _is_ignored(self, file_path: Path) -> bool:
        """Check if a file should be ignored based on .gitignore patterns."""
        rel_path = file_path.relative_to(self.repo_path)
        rel_str = str(rel_path)
        
        for pattern in self.ignored_patterns:
            if pattern in rel_str or rel_str.endswith(pattern.lstrip('/')):
                return True
            if pattern.startswith('*') and rel_str.endswith(pattern[1:]):
                return True
        
        return False
    
    def _get_file_language(self, file_path: Path) -> Optional[str]:
        """Detect programming language from file extension."""
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
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.md': 'markdown',
        }
        return ext_to_lang.get(file_path.suffix.lower())
    
    def scan(self) -> List[Dict]:
        """
        Scan the repository and return file metadata.
        
        Returns:
            List of dictionaries containing file information:
            - path: Relative path from repo root
            - full_path: Absolute path
            - language: Detected programming language
            - size: File size in bytes
            - is_binary: Whether file is binary
        """
        files = []
        
        logger.info(f"Scanning repository: {self.repo_path}")
        
        for root, dirs, filenames in os.walk(self.repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._is_ignored(Path(root) / d)]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                if self._is_ignored(file_path):
                    continue
                
                try:
                    rel_path = file_path.relative_to(self.repo_path)
                    language = self._get_file_language(file_path)
                    
                    # Skip binary files (except images which we might want to track)
                    if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico'}:
                        continue
                    
                    stat = file_path.stat()
                    size = stat.st_size
                    
                    # Check if file is binary (simple heuristic)
                    is_binary = False
                    if size > 0:
                        try:
                            with open(file_path, 'rb') as f:
                                chunk = f.read(1024)
                                if b'\x00' in chunk:
                                    is_binary = True
                        except Exception:
                            continue
                    
                    if is_binary and language is None:
                        continue
                    
                    files.append({
                        'path': str(rel_path),
                        'full_path': str(file_path),
                        'language': language,
                        'size': size,
                        'is_binary': is_binary
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Found {len(files)} files to process")
        return files
    
    def get_repo_metadata(self) -> Dict:
        """Get repository metadata including Git information."""
        metadata = {
            'repo_path': str(self.repo_path),
            'total_files': 0,
            'languages': {},
            'git_info': {}
        }
        
        # Try to get Git information
        try:
            result = subprocess.run(
                ['git', '-C', str(self.repo_path), 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                metadata['git_info']['is_git_repo'] = True
                
                # Get current branch
                branch_result = subprocess.run(
                    ['git', '-C', str(self.repo_path), 'branch', '--show-current'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if branch_result.returncode == 0:
                    metadata['git_info']['branch'] = branch_result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get Git info: {e}")
            metadata['git_info']['is_git_repo'] = False
        
        return metadata

