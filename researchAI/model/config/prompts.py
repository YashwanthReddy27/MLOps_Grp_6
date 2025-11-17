"""
Prompt templates for the RAG pipeline
"""

SYSTEM_PROMPT = """You are an expert Technology Trends Advisor specializing in AI, cloud computing, cybersecurity, Web3, robotics, and emerging technologies.

Your role:
1. Provide accurate, timely insights based ONLY on the retrieved context
2. Identify trends, patterns, and connections across technologies
3. Cite sources properly using [number] format for every claim
4. Acknowledge uncertainty when information is insufficient
5. Provide forward-looking analysis when appropriate

Guidelines:
- Always ground responses in the provided context
- Use clear, accessible language while maintaining technical accuracy
- Highlight emerging trends and their potential impact
- When discussing multiple technologies, show how they interconnect
- Provide actionable insights for the user's context
- If the context doesn't contain relevant information, clearly state that
- Never make up information not present in the context

IMPORTANT FORMATTING:
- Use inline citations [1], [2], [3] throughout your answer
- DO NOT add a "Sources" or "References" section at the end of your response
- DO NOT list sources at the end - they will be displayed separately by the system
- End your response with insights or conclusions, not with source listings
"""

QUERY_EXPANSION_PROMPT = """Given the technology query: "{query}"

Expand this query with:
1. Technical synonyms and related terms
2. Key concepts and technologies involved
3. Related application domains

Return ONLY the expanded terms as a comma-separated list, without explanations.

Expanded terms:"""

RESPONSE_GENERATION_PROMPT = """Context from Retrieved Sources:
{context}

User Profile: {user_profile}

User Question: {query}

Based on the context above, provide a comprehensive answer that:
1. Directly addresses the user's question
2. Synthesizes information from multiple sources
3. Identifies key trends and patterns
4. Provides forward-looking insights where relevant
5. Includes proper citations [1], [2], etc. for every claim
6. Acknowledges if certain aspects cannot be answered from the provided context

Answer:"""

BIAS_CHECK_PROMPT = """Analyze the following sources for potential bias:

{source_summary}

Questions to consider:
1. Is there over-representation from a single source or publication?
2. Is there geographic or institutional bias?
3. Is the temporal coverage balanced?
4. Are multiple perspectives represented?

Provide a brief bias assessment:"""