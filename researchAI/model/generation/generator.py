from typing import List, Dict, Any
import logging
import google.generativeai as genai

from config.settings import config
from config.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses using Gemini with retrieved context"""
    
    def __init__(self):
        self.config = config.generation
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini client
        if self.config.api_key:
            genai.configure(api_key=self.config.api_key)
            self.model = genai.GenerativeModel(self.config.model_name)
            self.logger.info(f"Initialized Gemini client with model: {self.config.model_name}")
        else:
            self.logger.warning("No Gemini API key provided. Using mock responses.")
            self.model = None
    
    def prepare_context(self, retrieved_docs: List[Dict[str, Any]], 
                       max_tokens: int = 3000) -> str:
        """
        Prepare context from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved documents
            max_tokens: Maximum tokens for context (approximate)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4
        
        for idx, doc in enumerate(retrieved_docs):
            meta = doc['metadata']
            
            # Format based on document type
            if meta.get('source') == 'arxiv' or 'arxiv_id' in meta:
                # Research paper
                authors = meta.get('authors', 'Unknown')
                if len(authors) > 100:
                    authors = authors[:100] + "..."
                
                citation = (
                    f"[{idx+1}] {meta.get('title', 'Untitled')} "
                    f"({authors}, {meta.get('published_date', 'N/A')[:4]})"
                )
                source_type = "Research Paper"
                url = meta.get('html_url', meta.get('pdf_url', ''))
            else:
                # News article
                citation = (
                    f"[{idx+1}] {meta.get('title', 'Untitled')} - "
                    f"{meta.get('source_name', 'Unknown')} "
                    f"({meta.get('published_at', 'N/A')[:10]})"
                )
                source_type = "News Article"
                url = meta.get('url', '')
            
            # Get text content
            text = meta.get('text', '')
            
            # Format section
            section = f"""
[{idx+1}] {source_type}
Title: {meta.get('title', 'Untitled')}
Content: {text}
Citation: {citation}
URL: {url}
Categories: {', '.join(meta.get('categories', [])[:3])}
---
"""
            # Check if we're exceeding limit
            if total_chars + len(section) > max_chars:
                self.logger.warning(f"Context truncated at {idx+1} documents")
                break
            
            context_parts.append(section)
            total_chars += len(section)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, 
                         retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using Gemini
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Returns:
            Response dictionary with text and metadata
        """
        if not retrieved_docs:
            return {
                'response': "I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different topic.",
                'sources': [],
                'error': 'no_results'
            }
        
        # Prepare context
        context = self.prepare_context(retrieved_docs)
        
        # Build prompt
        prompt = f"""{SYSTEM_PROMPT}

Retrieved Context:
{context}

User Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Synthesizes information from multiple sources
3. Identifies key trends and patterns
4. Provides forward-looking insights
5. Includes proper citations [1], [2], etc.

Answer:"""
        
        # Generate response
        if self.model:
            try:
                response = self._call_gemini(prompt)
            except Exception as e:
                self.logger.error(f"Error calling Gemini API: {e}")
                response = self._generate_fallback_response(query, retrieved_docs)
        else:
            response = self._generate_fallback_response(query, retrieved_docs)
        
        # Extract citations from response
        citations = self._extract_citations(response, retrieved_docs)
        
        return {
            'response': response,
            'sources': citations,
            'num_sources': len(retrieved_docs),
            'context_length': len(context)
        }
    
    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini API
        
        Args:
            prompt: User prompt
            
        Returns:
            Generated response
        """
        generation_config = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": self.config.max_tokens,
        }
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def _generate_fallback_response(self, query: str, 
                                   retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a simple fallback response without LLM
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Returns:
            Fallback response
        """
        # Simple summary of top documents
        response_parts = [
            f"Based on {len(retrieved_docs)} sources, here are the key findings:\n"
        ]
        
        for idx, doc in enumerate(retrieved_docs[:3]):
            meta = doc['metadata']
            title = meta.get('title', 'Untitled')
            text = meta.get('text', '')[:200]
            
            response_parts.append(f"[{idx+1}] {title}: {text}...")
        
        response_parts.append(
            "\n\nNote: Full response generation requires Gemini API key."
        )
        
        return "\n\n".join(response_parts)
    
    def _extract_citations(self, response: str, 
                          retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation information from response
        
        Args:
            response: Generated response
            retrieved_docs: Retrieved documents
            
        Returns:
            List of cited sources. If the response contains no explicit citations,
            fall back to the top retrieved documents.
        """
        import re
        
        # Find all citation numbers in response
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(re.findall(citation_pattern, response))
        
        citations = []
        for num_str in sorted(cited_numbers, key=int):
            idx = int(num_str) - 1
            if 0 <= idx < len(retrieved_docs):
                doc = retrieved_docs[idx]
                meta = doc['metadata']
                
                citation = {
                    'number': int(num_str),
                    'title': meta.get('title', 'Untitled'),
                    'source': meta.get('source_name') or 'arXiv',
                    'url': meta.get('url') or meta.get('html_url', ''),
                    'date': meta.get('published_at') or meta.get('published_date', '')
                }
                citations.append(citation)
        
        return citations


class StreamingGenerator(ResponseGenerator):
    """Generator with streaming support for Gemini"""
    
    def generate_response_stream(self, query: str, 
                                 retrieved_docs: List[Dict[str, Any]]):
        """
        Generate response with streaming
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Yields:
            Response chunks
        """
        if not self.model:
            # Fall back to non-streaming
            result = self.generate_response(query, retrieved_docs)
            yield result['response']
            return
        
        # Prepare context and prompt
        context = self.prepare_context(retrieved_docs)
        
        prompt = f"""{SYSTEM_PROMPT}

Retrieved Context:
{context}

User Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Synthesizes information from multiple sources
3. Identifies key trends and patterns
4. Provides forward-looking insights
5. Includes proper citations [1], [2], etc.

Answer:"""
        
        try:
            generation_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_output_tokens": self.config.max_tokens,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            self.logger.error(f"Error in streaming: {e}")
            yield f"Error generating response: {str(e)}"