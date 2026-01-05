"""Question answering engine with intent detection and answer generation."""

from typing import List, Dict, Any, Optional
import json
import logging

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai package not available. Install with: pip install openai")

logger = logging.getLogger(__name__)


class QAEngine:
    """Question answering engine that generates grounded answers."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the QA engine.
        
        Args:
            api_key: OpenAI API key (or DeepSeek API key)
            base_url: Base URL for API (for DeepSeek)
            model: Model name to use
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")
        
        if not api_key:
            raise ValueError("API key required for QA engine")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info(f"QA Engine initialized with model: {model}")
    
    def detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect the intent of a query.
        
        Args:
            query: User query
        
        Returns:
            Dictionary with intent classification
        """
        intent_prompt = f"""Classify the following developer question into one of these categories:
- architecture: Questions about system design, architecture, flow
- location: Questions asking "where" something is implemented
- explanation: Questions asking "how" something works
- dependency: Questions about dependencies, relationships
- code_search: Questions about finding specific code patterns

Question: {query}

Respond with JSON only:
{{"intent": "<category>", "keywords": ["keyword1", "keyword2"]}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intent classifier. Respond with valid JSON only."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            # Clean JSON response
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            intent_data = json.loads(result.strip())
            return intent_data
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}. Using default.")
            return {"intent": "code_search", "keywords": query.split()}
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                       context_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved chunks and context.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved code chunks
            context_summary: Optional MCP context summary
        
        Returns:
            Dictionary with answer and metadata
        """
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant code to answer your question. Please try rephrasing or check if the repository has been indexed.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks[:5], 1):  # Use top 5 chunks
            file_path = chunk.get('file_path', 'unknown')
            start_line = chunk.get('start_line', 0)
            end_line = chunk.get('end_line', 0)
            content = chunk.get('content', '')
            similarity = chunk.get('similarity_score', 0.0)
            
            context_parts.append(f"[Source {i}] File: {file_path} (lines {start_line}-{end_line}, similarity: {similarity:.2f})\n{content}\n")
            sources.append({
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'similarity': similarity
            })
        
        context_text = "\n".join(context_parts)
        
        # Add MCP context if available
        mcp_context = ""
        if context_summary:
            mcp_context = f"\n\nRepository Context:\n- Total files: {context_summary.get('ast_summaries_count', 0)}\n"
            if context_summary.get('architecture_overview'):
                mcp_context += f"- Architecture: {str(context_summary['architecture_overview'])[:200]}\n"
        
        prompt = f"""You are an expert codebase analyst. Answer the following question based on the provided code context.

Question: {query}

Relevant Code Context:
{context_text}
{mcp_context}

Instructions:
1. Provide a clear, concise answer
2. Reference specific files and line numbers when relevant
3. If the answer requires information not in the context, say so
4. Include code snippets if helpful
5. Explain the flow or architecture if asked

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert codebase analyst. Provide clear, accurate answers based on the code context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(s.get('similarity', 0.0) for s in sources) / len(sources) if sources else 0.0
            confidence = min(1.0, avg_similarity * 1.2)  # Scale confidence
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'num_chunks_used': len(sources)
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': sources,
                'confidence': 0.0
            }
    
    def answer_question(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                       context_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete QA pipeline: intent detection + answer generation.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved code chunks
            context_summary: Optional MCP context
        
        Returns:
            Complete answer with metadata
        """
        intent = self.detect_intent(query)
        answer_data = self.generate_answer(query, retrieved_chunks, context_summary)
        
        return {
            **answer_data,
            'intent': intent.get('intent', 'unknown'),
            'keywords': intent.get('keywords', [])
        }

