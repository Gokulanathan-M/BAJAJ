"""
LLM Query Processor for parsing and understanding natural language queries
"""
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import logging

import openai
from openai import OpenAI

from .models import Query, QueryType, Domain, RetrievedChunk
from .config import config, DOMAIN_CONFIGS

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


class TokenCounter:
    """Token usage tracking"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def add_usage(self, usage_dict: Dict[str, int]):
        """Add token usage from OpenAI response"""
        if usage_dict:
            self.prompt_tokens += usage_dict.get('prompt_tokens', 0)
            self.completion_tokens += usage_dict.get('completion_tokens', 0)
            self.total_tokens += usage_dict.get('total_tokens', 0)
    
    def get_usage(self) -> Dict[str, int]:
        """Get current token usage"""
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }


class QueryAnalyzer:
    """Analyzes and categorizes user queries"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.token_counter = TokenCounter()
    
    def analyze_query(self, query_text: str) -> Tuple[QueryType, Domain, Dict[str, Any]]:
        """Analyze query to determine type, domain, and extract key information"""
        
        system_prompt = """You are an expert query analyzer for a document retrieval system. 
        Analyze the user query and respond with a JSON object containing:
        
        1. "query_type": One of [coverage_analysis, clause_matching, policy_lookup, compliance_check, general_search]
        2. "domain": One of [insurance, legal, hr, compliance] 
        3. "key_entities": List of important entities/terms mentioned
        4. "intent": Brief description of what the user wants to know
        5. "complexity": One of [simple, moderate, complex]
        
        Query Types:
        - coverage_analysis: Questions about what is covered by policies
        - clause_matching: Finding specific clauses or contract terms
        - policy_lookup: Looking up specific policy information
        - compliance_check: Checking compliance with regulations/rules
        - general_search: General information retrieval
        
        Domains:
        - insurance: Insurance policies, claims, coverage
        - legal: Contracts, agreements, legal documents
        - hr: Employee policies, benefits, procedures
        - compliance: Regulatory requirements, audits, standards
        """
        
        user_prompt = f"Analyze this query: {query_text}"
        
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.add_usage(response.usage.model_dump())
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            analysis = json.loads(content)
            
            query_type = QueryType(analysis.get('query_type', 'general_search'))
            domain = Domain(analysis.get('domain', 'legal'))
            metadata = {
                'key_entities': analysis.get('key_entities', []),
                'intent': analysis.get('intent', ''),
                'complexity': analysis.get('complexity', 'moderate'),
                'analysis_raw': analysis
            }
            
            return query_type, domain, metadata
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Fallback to rule-based analysis
            return self._fallback_analysis(query_text)
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Fallback to rule-based analysis
            return self._fallback_analysis(query_text)
    
    def _fallback_analysis(self, query_text: str) -> Tuple[QueryType, Domain, Dict[str, Any]]:
        """Fallback rule-based query analysis"""
        query_lower = query_text.lower()
        
        # Determine query type based on keywords
        if any(word in query_lower for word in ['cover', 'coverage', 'covered', 'benefit']):
            query_type = QueryType.COVERAGE_ANALYSIS
        elif any(word in query_lower for word in ['clause', 'section', 'term', 'condition']):
            query_type = QueryType.CLAUSE_MATCHING
        elif any(word in query_lower for word in ['policy', 'document', 'contract']):
            query_type = QueryType.POLICY_LOOKUP
        elif any(word in query_lower for word in ['comply', 'compliance', 'regulation', 'requirement']):
            query_type = QueryType.COMPLIANCE_CHECK
        else:
            query_type = QueryType.GENERAL_SEARCH
        
        # Determine domain based on keywords
        domain_scores = {}
        for domain_name, domain_config in DOMAIN_CONFIGS.items():
            score = sum(1 for term in domain_config['key_terms'] if term in query_lower)
            domain_scores[domain_name] = score
        
        best_domain = max(domain_scores, key=domain_scores.get)
        domain = Domain(best_domain) if domain_scores[best_domain] > 0 else Domain.LEGAL
        
        metadata = {
            'key_entities': self._extract_entities(query_text),
            'intent': f"Query about {query_type.value} in {domain.value} domain",
            'complexity': 'moderate',
            'fallback_used': True
        }
        
        return query_type, domain, metadata
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction"""
        # Extract quoted phrases
        entities = re.findall(r'"([^"]*)"', text)
        
        # Extract capitalized words (potential proper nouns)
        entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        
        # Remove duplicates and filter short words
        entities = list(set([e for e in entities if len(e) > 2]))
        
        return entities[:10]  # Limit to 10 entities


class QueryEnhancer:
    """Enhances queries for better retrieval"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.token_counter = TokenCounter()
    
    def enhance_query(self, query: Query, domain_context: Optional[str] = None) -> str:
        """Enhance query with domain-specific terms and synonyms"""
        
        domain_config = DOMAIN_CONFIGS.get(query.domain.value, {})
        key_terms = domain_config.get('key_terms', [])
        
        system_prompt = f"""You are an expert in {query.domain.value} domain. 
        Enhance the user query by:
        1. Adding relevant synonyms and related terms
        2. Including domain-specific terminology
        3. Expanding abbreviations
        4. Making the query more specific for document retrieval
        
        Key terms for this domain: {', '.join(key_terms)}
        
        Return only the enhanced query text, no explanations."""
        
        user_prompt = f"Original query: {query.text}"
        if domain_context:
            user_prompt += f"\nDomain context: {domain_context}"
        
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.add_usage(response.usage.model_dump())
            
            enhanced_query = response.choices[0].message.content.strip()
            
            # Ensure the enhanced query is reasonable
            if len(enhanced_query) > len(query.text) * 3:
                logger.warning("Enhanced query too long, using original")
                return query.text
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query.text


class ContextualAnswerer:
    """Generates contextual answers based on retrieved chunks"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.token_counter = TokenCounter()
    
    def generate_answer(self, query: Query, retrieved_chunks: List[RetrievedChunk],
                       max_context_length: int = 3000) -> Tuple[str, List[str], List[str]]:
        """Generate answer with supporting and contradicting evidence"""
        
        # Prepare context from retrieved chunks
        context_parts = []
        total_length = 0
        
        for chunk in retrieved_chunks:
            chunk_text = f"Source: {chunk.chunk.document_id}\n{chunk.chunk.content}\n"
            if total_length + len(chunk_text) <= max_context_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                break
        
        context = "\n--- Document Chunk ---\n".join(context_parts)
        
        system_prompt = f"""You are an expert analyst in the {query.domain.value} domain.
        Based on the provided document chunks, answer the user's question thoroughly and accurately.
        
        Instructions:
        1. Provide a clear, direct answer to the question
        2. Base your answer strictly on the provided context
        3. If information is insufficient, state what's missing
        4. Cite specific parts of the documents when possible
        5. Be precise about conditions, limitations, or exceptions
        6. If there are contradictions in the sources, highlight them
        
        Format your response as a JSON object with:
        {{
            "answer": "Direct answer to the question",
            "supporting_evidence": ["Evidence supporting the answer"],
            "contradicting_evidence": ["Any contradicting information found"],
            "confidence": "high/medium/low",
            "citations": ["Specific document references"]
        }}"""
        
        user_prompt = f"""Question: {query.text}
        
        Document Context:
        {context}
        
        Please analyze and provide a comprehensive answer."""
        
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.token_counter.add_usage(response.usage.model_dump())
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(content)
                answer = result.get('answer', 'Unable to generate answer')
                supporting = result.get('supporting_evidence', [])
                contradicting = result.get('contradicting_evidence', [])
                
                return answer, supporting, contradicting
                
            except json.JSONDecodeError:
                # If JSON parsing fails, treat the whole response as the answer
                return content, [], []
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}", [], []


class LLMProcessor:
    """Main LLM processor that coordinates all LLM-based operations"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise LLMError("OpenAI API key not provided")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer(self.client)
        self.query_enhancer = QueryEnhancer(self.client)
        self.contextual_answerer = ContextualAnswerer(self.client)
        
        # Overall token counter
        self.token_counter = TokenCounter()
    
    def process_query(self, query_text: str, query_id: str = None) -> Query:
        """Process and analyze a natural language query"""
        start_time = time.time()
        
        try:
            # Analyze the query
            query_type, domain, metadata = self.query_analyzer.analyze_query(query_text)
            
            # Create Query object
            query = Query(
                id=query_id or f"query_{int(time.time())}",
                text=query_text,
                query_type=query_type,
                domain=domain,
                filters=metadata
            )
            
            processing_time = time.time() - start_time
            metadata['processing_time'] = processing_time
            
            logger.info(f"Processed query: {query_type.value} in {domain.value} domain")
            return query
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise LLMError(f"Failed to process query: {str(e)}")
    
    def enhance_query_for_retrieval(self, query: Query) -> str:
        """Enhance query text for better document retrieval"""
        try:
            enhanced_text = self.query_enhancer.enhance_query(query)
            logger.info(f"Enhanced query: '{query.text}' -> '{enhanced_text}'")
            return enhanced_text
        except Exception as e:
            logger.warning(f"Failed to enhance query, using original: {e}")
            return query.text
    
    def generate_contextual_answer(self, query: Query, 
                                 retrieved_chunks: List[RetrievedChunk]) -> Tuple[str, List[str], List[str]]:
        """Generate a contextual answer based on retrieved documents"""
        return self.contextual_answerer.generate_answer(query, retrieved_chunks)
    
    def get_total_token_usage(self) -> Dict[str, int]:
        """Get total token usage across all components"""
        total_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        
        # Aggregate from all components
        for component in [self.query_analyzer, self.query_enhancer, self.contextual_answerer]:
            usage = component.token_counter.get_usage()
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
        
        return total_usage
    
    def reset_token_counters(self):
        """Reset all token counters"""
        for component in [self.query_analyzer, self.query_enhancer, self.contextual_answerer]:
            component.token_counter.reset()
        self.token_counter.reset()
    
    def validate_api_connection(self) -> bool:
        """Validate OpenAI API connection"""
        try:
            response = self.client.chat.completions.create(
                model=config.openai_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return False