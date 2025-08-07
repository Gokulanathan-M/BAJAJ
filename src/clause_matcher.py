"""
Clause matching and semantic similarity system for document analysis
"""
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Document, DocumentChunk, ClauseMatch, RetrievedChunk, Query, Domain
from .config import config, DOMAIN_CONFIGS

logger = logging.getLogger(__name__)


class ClauseMatchingError(Exception):
    """Custom exception for clause matching errors"""
    pass


class ClauseExtractor:
    """Extracts clauses from documents using pattern matching"""
    
    def __init__(self):
        # Define clause patterns for different domains
        self.clause_patterns = {
            Domain.INSURANCE: {
                'coverage_clause': [
                    r'coverage includes?:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'this policy covers:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'covered (?:expenses?|services?|treatments?):?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'exclusion_clause': [
                    r'exclusions?:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'not covered:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'this policy does not cover:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'deductible_clause': [
                    r'deductible:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'deductible amount:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'premium_clause': [
                    r'premium:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'premium amount:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ]
            },
            Domain.LEGAL: {
                'termination_clause': [
                    r'termination:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'this agreement (?:may be|shall be) terminated:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'liability_clause': [
                    r'liability:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'limitation of liability:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'governing_law_clause': [
                    r'governing law:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'this agreement (?:shall be|is) governed by:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'confidentiality_clause': [
                    r'confidentiality:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'non-disclosure:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ]
            },
            Domain.HR: {
                'compensation_clause': [
                    r'compensation:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'salary:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'wage:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'benefits_clause': [
                    r'benefits?:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'employee benefits?:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'leave_clause': [
                    r'leave:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'vacation:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'sick leave:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'termination_clause': [
                    r'termination of employment:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'employment may be terminated:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ]
            },
            Domain.COMPLIANCE: {
                'compliance_requirement': [
                    r'compliance requirement:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'must comply with:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'audit_clause': [
                    r'audit:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'audit requirement:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'reporting_clause': [
                    r'reporting:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'report:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ],
                'penalty_clause': [
                    r'penalty:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'violation:?\s*(.*?)(?:\n\n|\. [A-Z]|$)',
                    r'non-compliance:?\s*(.*?)(?:\n\n|\. [A-Z]|$)'
                ]
            }
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for domain, clause_types in self.clause_patterns.items():
            self.compiled_patterns[domain] = {}
            for clause_type, patterns in clause_types.items():
                self.compiled_patterns[domain][clause_type] = [
                    re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                    for pattern in patterns
                ]
    
    def extract_clauses(self, document: Document) -> List[ClauseMatch]:
        """Extract clauses from a document"""
        clauses = []
        content = document.content.lower()
        
        # Get patterns for the document's domain
        domain_patterns = self.compiled_patterns.get(document.domain, {})
        
        for clause_type, patterns in domain_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(document.content)
                
                for match in matches:
                    clause_text = match.group(1).strip() if match.groups() else match.group(0).strip()
                    
                    # Skip very short matches
                    if len(clause_text) < 20:
                        continue
                    
                    # Get context around the match
                    start_pos = max(0, match.start() - 100)
                    end_pos = min(len(document.content), match.end() + 100)
                    context = document.content[start_pos:end_pos]
                    
                    clause = ClauseMatch(
                        clause_text=clause_text,
                        clause_type=clause_type,
                        document_id=document.id,
                        confidence_score=self._calculate_confidence(clause_text, clause_type),
                        start_position=match.start(),
                        end_position=match.end(),
                        context=context
                    )
                    
                    clauses.append(clause)
        
        # Remove duplicates and sort by confidence
        clauses = self._deduplicate_clauses(clauses)
        clauses.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return clauses
    
    def _calculate_confidence(self, clause_text: str, clause_type: str) -> float:
        """Calculate confidence score for a clause match"""
        base_score = 0.7
        
        # Add points for length (longer clauses tend to be more complete)
        length_score = min(0.2, len(clause_text) / 500)
        
        # Add points for domain-specific keywords
        keyword_score = 0.0
        clause_lower = clause_text.lower()
        
        if clause_type in ['coverage_clause', 'exclusion_clause']:
            keywords = ['coverage', 'benefit', 'limit', 'maximum', 'minimum']
            keyword_score = sum(0.02 for keyword in keywords if keyword in clause_lower)
        elif clause_type in ['termination_clause']:
            keywords = ['notice', 'cause', 'without cause', 'immediate']
            keyword_score = sum(0.02 for keyword in keywords if keyword in clause_lower)
        elif clause_type in ['liability_clause']:
            keywords = ['liable', 'damages', 'limitation', 'maximum', 'aggregate']
            keyword_score = sum(0.02 for keyword in keywords if keyword in clause_lower)
        
        return min(1.0, base_score + length_score + keyword_score)
    
    def _deduplicate_clauses(self, clauses: List[ClauseMatch]) -> List[ClauseMatch]:
        """Remove duplicate clauses based on text similarity"""
        if len(clauses) <= 1:
            return clauses
        
        unique_clauses = []
        texts = [clause.clause_text for clause in clauses]
        
        # Use TF-IDF for similarity detection
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            processed = set()
            
            for i, clause in enumerate(clauses):
                if i in processed:
                    continue
                
                # Find similar clauses
                similar_indices = np.where(similarity_matrix[i] > 0.8)[0]
                
                # Keep the one with highest confidence
                best_idx = max(similar_indices, key=lambda idx: clauses[idx].confidence_score)
                unique_clauses.append(clauses[best_idx])
                
                # Mark all similar clauses as processed
                processed.update(similar_indices)
        
        except Exception as e:
            logger.warning(f"Error in clause deduplication: {e}")
            return clauses
        
        return unique_clauses


class SemanticClauseMatcher:
    """Semantic matching for finding clauses similar to a query"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.clause_cache = {}  # Cache for computed clause embeddings
    
    def find_similar_clauses(self, query: Query, clause_matches: List[ClauseMatch],
                           top_k: int = 5, threshold: float = 0.7) -> List[ClauseMatch]:
        """Find clauses similar to the query using semantic similarity"""
        if not clause_matches:
            return []
        
        try:
            # Get query embedding
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([query.text])
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            else:
                # Fallback to TF-IDF similarity
                return self._tfidf_similarity_matching(query, clause_matches, top_k, threshold)
            
            # Get clause embeddings
            clause_texts = [clause.clause_text for clause in clause_matches]
            clause_embeddings = self.embedding_model.encode(clause_texts)
            clause_embeddings = clause_embeddings / np.linalg.norm(clause_embeddings, axis=1, keepdims=True)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, clause_embeddings)[0]
            
            # Create scored results
            scored_clauses = []
            for i, (clause, similarity) in enumerate(zip(clause_matches, similarities)):
                if similarity >= threshold:
                    # Update confidence score based on semantic similarity
                    new_confidence = (clause.confidence_score + similarity) / 2
                    updated_clause = ClauseMatch(
                        clause_text=clause.clause_text,
                        clause_type=clause.clause_type,
                        document_id=clause.document_id,
                        confidence_score=new_confidence,
                        start_position=clause.start_position,
                        end_position=clause.end_position,
                        context=clause.context
                    )
                    scored_clauses.append(updated_clause)
            
            # Sort by confidence and return top_k
            scored_clauses.sort(key=lambda x: x.confidence_score, reverse=True)
            return scored_clauses[:top_k]
        
        except Exception as e:
            logger.error(f"Error in semantic clause matching: {e}")
            return self._tfidf_similarity_matching(query, clause_matches, top_k, threshold)
    
    def _tfidf_similarity_matching(self, query: Query, clause_matches: List[ClauseMatch],
                                 top_k: int, threshold: float) -> List[ClauseMatch]:
        """Fallback TF-IDF based similarity matching"""
        try:
            texts = [query.text] + [clause.clause_text for clause in clause_matches]
            
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity between query and clauses
            query_vector = tfidf_matrix[0:1]
            clause_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, clause_vectors)[0]
            
            # Create scored results
            scored_clauses = []
            for i, (clause, similarity) in enumerate(zip(clause_matches, similarities)):
                if similarity >= threshold:
                    new_confidence = (clause.confidence_score + similarity) / 2
                    updated_clause = ClauseMatch(
                        clause_text=clause.clause_text,
                        clause_type=clause.clause_type,
                        document_id=clause.document_id,
                        confidence_score=new_confidence,
                        start_position=clause.start_position,
                        end_position=clause.end_position,
                        context=clause.context
                    )
                    scored_clauses.append(updated_clause)
            
            # Sort by confidence and return top_k
            scored_clauses.sort(key=lambda x: x.confidence_score, reverse=True)
            return scored_clauses[:top_k]
        
        except Exception as e:
            logger.error(f"Error in TF-IDF clause matching: {e}")
            return clause_matches[:top_k]


class ClauseAnalyzer:
    """Analyzes clauses for specific patterns and extracts key information"""
    
    def __init__(self):
        # Define key information patterns
        self.info_patterns = {
            'amounts': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*\s*(?:dollars?|USD)',
            'percentages': r'\d+(?:\.\d+)?%',
            'dates': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            'timeframes': r'\d+\s*(?:days?|weeks?|months?|years?)',
            'conditions': r'(?:if|when|unless|provided that|subject to)\s+[^.]+',
            'requirements': r'(?:must|shall|required to|obligated to)\s+[^.]+',
            'exceptions': r'(?:except|excluding|but not|however)\s+[^.]+'
        }
        
        self.compiled_info_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.info_patterns.items()
        }
    
    def analyze_clause(self, clause: ClauseMatch) -> Dict[str, Any]:
        """Analyze a clause and extract key information"""
        analysis = {
            'clause_id': f"{clause.document_id}_{clause.start_position}",
            'clause_type': clause.clause_type,
            'extracted_info': {},
            'key_terms': [],
            'complexity_score': 0.0,
            'readability_score': 0.0
        }
        
        text = clause.clause_text
        
        # Extract structured information
        for info_type, pattern in self.compiled_info_patterns.items():
            matches = pattern.findall(text)
            if matches:
                analysis['extracted_info'][info_type] = matches
        
        # Extract key terms (capitalized words, technical terms)
        key_terms = set()
        
        # Capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        key_terms.update(capitalized)
        
        # Technical terms based on domain
        domain_terms = self._extract_domain_terms(text, clause.clause_type)
        key_terms.update(domain_terms)
        
        analysis['key_terms'] = list(key_terms)[:20]  # Limit to 20 terms
        
        # Calculate complexity score
        analysis['complexity_score'] = self._calculate_complexity(text)
        
        # Calculate readability score (simplified)
        analysis['readability_score'] = self._calculate_readability(text)
        
        return analysis
    
    def _extract_domain_terms(self, text: str, clause_type: str) -> Set[str]:
        """Extract domain-specific terms"""
        terms = set()
        text_lower = text.lower()
        
        # Insurance terms
        if 'insurance' in clause_type or 'coverage' in clause_type:
            insurance_terms = ['deductible', 'premium', 'copay', 'coinsurance', 'out-of-pocket', 
                             'network', 'provider', 'claim', 'benefit', 'limit']
            terms.update([term for term in insurance_terms if term in text_lower])
        
        # Legal terms
        elif 'legal' in clause_type or 'liability' in clause_type or 'termination' in clause_type:
            legal_terms = ['breach', 'damages', 'indemnify', 'jurisdiction', 'arbitration',
                          'remedy', 'waiver', 'severability', 'force majeure']
            terms.update([term for term in legal_terms if term in text_lower])
        
        # HR terms
        elif 'compensation' in clause_type or 'benefits' in clause_type or 'leave' in clause_type:
            hr_terms = ['salary', 'wage', 'benefits', 'vacation', 'sick leave', 'bonus',
                       'overtime', 'commission', 'stock options', 'retirement']
            terms.update([term for term in hr_terms if term in text_lower])
        
        # Compliance terms
        elif 'compliance' in clause_type or 'audit' in clause_type:
            compliance_terms = ['regulation', 'standard', 'requirement', 'audit', 'inspection',
                              'certification', 'approval', 'violation', 'penalty', 'fine']
            terms.update([term for term in compliance_terms if term in text_lower])
        
        return terms
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate clause complexity score (0-1)"""
        # Simple complexity based on sentence length, nested clauses, etc.
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()])
        
        # Count nested structures
        nested_count = text.count('(') + text.count('[') + text.count('{')
        conditional_count = len(re.findall(r'\b(?:if|when|unless|provided|subject)\b', text, re.IGNORECASE))
        
        # Normalize scores
        length_score = min(1.0, avg_sentence_length / 30)  # 30 words = complex
        nested_score = min(1.0, nested_count / 10)  # 10 nested structures = complex
        conditional_score = min(1.0, conditional_count / 5)  # 5 conditions = complex
        
        return (length_score + nested_score + conditional_score) / 3
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch formula
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        
        # Normalize to 0-1 scale (higher = more readable)
        return max(0.0, min(1.0, score / 100))
    
    def _count_syllables(self, text: str) -> int:
        """Simple syllable counting"""
        # Very basic syllable counting - count vowel groups
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in text.lower():
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Adjust for silent 'e'
        if text.lower().endswith('e'):
            syllables -= 1
        
        return max(1, syllables)  # At least one syllable per word


class ClauseMatcher:
    """Main clause matching system that coordinates extraction, analysis, and semantic matching"""
    
    def __init__(self, embedding_model=None):
        self.extractor = ClauseExtractor()
        self.semantic_matcher = SemanticClauseMatcher(embedding_model)
        self.analyzer = ClauseAnalyzer()
        
        # Cache for document clauses
        self.document_clauses = {}
    
    def process_document(self, document: Document) -> List[ClauseMatch]:
        """Process a document and extract clauses"""
        try:
            clauses = self.extractor.extract_clauses(document)
            self.document_clauses[document.id] = clauses
            
            logger.info(f"Extracted {len(clauses)} clauses from document {document.id}")
            return clauses
        
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            raise ClauseMatchingError(f"Failed to process document: {str(e)}")
    
    def match_clauses_for_query(self, query: Query, documents: List[Document] = None,
                              top_k: int = 5) -> List[ClauseMatch]:
        """Find clauses that match a specific query"""
        try:
            # Collect all clauses from documents
            all_clauses = []
            
            if documents:
                for document in documents:
                    if document.id not in self.document_clauses:
                        clauses = self.process_document(document)
                    else:
                        clauses = self.document_clauses[document.id]
                    all_clauses.extend(clauses)
            else:
                # Use all cached clauses
                for clauses in self.document_clauses.values():
                    all_clauses.extend(clauses)
            
            if not all_clauses:
                return []
            
            # Filter by domain if specified
            if query.domain:
                # Would need document domain info in clauses for filtering
                pass
            
            # Use semantic matching to find similar clauses
            matched_clauses = self.semantic_matcher.find_similar_clauses(
                query, all_clauses, top_k
            )
            
            logger.info(f"Found {len(matched_clauses)} matching clauses for query")
            return matched_clauses
        
        except Exception as e:
            logger.error(f"Error matching clauses for query: {e}")
            raise ClauseMatchingError(f"Failed to match clauses: {str(e)}")
    
    def analyze_matched_clauses(self, clauses: List[ClauseMatch]) -> List[Dict[str, Any]]:
        """Analyze matched clauses for detailed information"""
        analyses = []
        
        for clause in clauses:
            try:
                analysis = self.analyzer.analyze_clause(clause)
                analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Error analyzing clause: {e}")
                continue
        
        return analyses
    
    def get_clause_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed clauses"""
        total_clauses = sum(len(clauses) for clauses in self.document_clauses.values())
        
        clause_types = defaultdict(int)
        for clauses in self.document_clauses.values():
            for clause in clauses:
                clause_types[clause.clause_type] += 1
        
        return {
            'total_documents': len(self.document_clauses),
            'total_clauses': total_clauses,
            'clause_types': dict(clause_types),
            'average_clauses_per_document': total_clauses / len(self.document_clauses) if self.document_clauses else 0
        }