"""
Logic evaluation and decision engine for analyzing retrieved information
"""
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .models import (
    Query, QueryResponse, RetrievedChunk, ClauseMatch, 
    DecisionReasoning, Domain, QueryType
)
from .config import config

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceType(Enum):
    """Types of evidence"""
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


@dataclass
class Evidence:
    """Evidence structure for decision making"""
    text: str
    source: str
    evidence_type: EvidenceType
    weight: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context: str = ""


class DecisionEngineError(Exception):
    """Custom exception for decision engine errors"""
    pass


class EvidenceAnalyzer:
    """Analyzes evidence for decision making"""
    
    def __init__(self):
        # Define domain-specific decision patterns
        self.decision_patterns = {
            Domain.INSURANCE: {
                'coverage_keywords': [
                    'covered', 'includes', 'benefits', 'eligible', 'provided',
                    'reimbursed', 'paid', 'compensated'
                ],
                'exclusion_keywords': [
                    'excluded', 'not covered', 'except', 'limitation', 'restriction',
                    'does not include', 'not eligible', 'not provided'
                ],
                'condition_keywords': [
                    'if', 'when', 'provided that', 'subject to', 'only if',
                    'must', 'required', 'necessary', 'conditional'
                ]
            },
            Domain.LEGAL: {
                'affirmative_keywords': [
                    'shall', 'must', 'required', 'obligated', 'bound',
                    'entitled', 'authorized', 'permitted'
                ],
                'negative_keywords': [
                    'shall not', 'prohibited', 'forbidden', 'not permitted',
                    'not authorized', 'not entitled', 'excluded'
                ],
                'conditional_keywords': [
                    'if', 'unless', 'provided', 'subject to', 'in case of',
                    'upon', 'when', 'where', 'conditional'
                ]
            },
            Domain.HR: {
                'entitlement_keywords': [
                    'entitled', 'eligible', 'qualified', 'benefit',
                    'receive', 'provided', 'granted'
                ],
                'restriction_keywords': [
                    'not entitled', 'not eligible', 'restricted', 'limited',
                    'subject to', 'conditional', 'probationary'
                ],
                'requirement_keywords': [
                    'must', 'required', 'necessary', 'mandatory',
                    'obligated', 'expected', 'responsible'
                ]
            },
            Domain.COMPLIANCE: {
                'compliant_keywords': [
                    'complies', 'meets', 'satisfies', 'adheres',
                    'conforms', 'fulfills', 'in accordance'
                ],
                'non_compliant_keywords': [
                    'violates', 'breaches', 'fails to meet', 'non-compliant',
                    'does not comply', 'insufficient', 'inadequate'
                ],
                'requirement_keywords': [
                    'must', 'shall', 'required', 'mandatory',
                    'obligated', 'necessary', 'compulsory'
                ]
            }
        }
    
    def analyze_evidence(self, retrieved_chunks: List[RetrievedChunk], 
                        matched_clauses: List[ClauseMatch], 
                        query: Query) -> List[Evidence]:
        """Analyze retrieved chunks and clauses to extract evidence"""
        evidence_list = []
        
        # Analyze retrieved chunks
        for chunk in retrieved_chunks:
            evidence_type, weight, confidence = self._classify_chunk_evidence(
                chunk, query
            )
            
            evidence = Evidence(
                text=chunk.chunk.content,
                source=f"Document: {chunk.chunk.document_id}",
                evidence_type=evidence_type,
                weight=weight,
                confidence=confidence,
                context=f"Similarity: {chunk.similarity_score:.3f}"
            )
            evidence_list.append(evidence)
        
        # Analyze matched clauses
        for clause in matched_clauses:
            evidence_type, weight, confidence = self._classify_clause_evidence(
                clause, query
            )
            
            evidence = Evidence(
                text=clause.clause_text,
                source=f"Clause: {clause.clause_type} in {clause.document_id}",
                evidence_type=evidence_type,
                weight=weight,
                confidence=confidence,
                context=clause.context
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def _classify_chunk_evidence(self, chunk: RetrievedChunk, 
                               query: Query) -> Tuple[EvidenceType, float, float]:
        """Classify a chunk as supporting, contradicting, or neutral evidence"""
        content = chunk.chunk.content.lower()
        query_text = query.text.lower()
        
        # Get domain-specific patterns
        patterns = self.decision_patterns.get(query.domain, {})
        
        # Calculate evidence type based on query type and content
        if query.query_type == QueryType.COVERAGE_ANALYSIS:
            return self._analyze_coverage_evidence(content, query_text, patterns)
        elif query.query_type == QueryType.CLAUSE_MATCHING:
            return self._analyze_clause_evidence(content, query_text, patterns)
        elif query.query_type == QueryType.COMPLIANCE_CHECK:
            return self._analyze_compliance_evidence(content, query_text, patterns)
        else:
            return self._analyze_general_evidence(content, query_text, chunk.similarity_score)
    
    def _classify_clause_evidence(self, clause: ClauseMatch, 
                                query: Query) -> Tuple[EvidenceType, float, float]:
        """Classify a clause as supporting, contradicting, or neutral evidence"""
        content = clause.clause_text.lower()
        query_text = query.text.lower()
        
        # Higher weight for clauses due to their structured nature
        base_weight = 0.8
        base_confidence = clause.confidence_score
        
        patterns = self.decision_patterns.get(query.domain, {})
        evidence_type, weight_mod, conf_mod = self._analyze_general_evidence(
            content, query_text, base_confidence
        )
        
        return evidence_type, base_weight * weight_mod, base_confidence * conf_mod
    
    def _analyze_coverage_evidence(self, content: str, query_text: str, 
                                 patterns: Dict[str, List[str]]) -> Tuple[EvidenceType, float, float]:
        """Analyze evidence for coverage queries"""
        coverage_keywords = patterns.get('coverage_keywords', [])
        exclusion_keywords = patterns.get('exclusion_keywords', [])
        
        coverage_score = sum(1 for keyword in coverage_keywords if keyword in content)
        exclusion_score = sum(1 for keyword in exclusion_keywords if keyword in content)
        
        if coverage_score > exclusion_score:
            return EvidenceType.SUPPORTING, 0.8, 0.8
        elif exclusion_score > coverage_score:
            return EvidenceType.CONTRADICTING, 0.8, 0.8
        else:
            return EvidenceType.NEUTRAL, 0.5, 0.6
    
    def _analyze_clause_evidence(self, content: str, query_text: str,
                               patterns: Dict[str, List[str]]) -> Tuple[EvidenceType, float, float]:
        """Analyze evidence for clause matching queries"""
        # Check for direct matches with query terms
        query_words = set(query_text.split())
        content_words = set(content.split())
        overlap = len(query_words.intersection(content_words))
        
        if overlap > len(query_words) * 0.5:
            return EvidenceType.SUPPORTING, 0.9, 0.9
        elif overlap > len(query_words) * 0.2:
            return EvidenceType.SUPPORTING, 0.6, 0.7
        else:
            return EvidenceType.NEUTRAL, 0.4, 0.5
    
    def _analyze_compliance_evidence(self, content: str, query_text: str,
                                   patterns: Dict[str, List[str]]) -> Tuple[EvidenceType, float, float]:
        """Analyze evidence for compliance queries"""
        compliant_keywords = patterns.get('compliant_keywords', [])
        non_compliant_keywords = patterns.get('non_compliant_keywords', [])
        
        compliant_score = sum(1 for keyword in compliant_keywords if keyword in content)
        non_compliant_score = sum(1 for keyword in non_compliant_keywords if keyword in content)
        
        if compliant_score > non_compliant_score:
            return EvidenceType.SUPPORTING, 0.8, 0.8
        elif non_compliant_score > compliant_score:
            return EvidenceType.CONTRADICTING, 0.8, 0.8
        else:
            return EvidenceType.NEUTRAL, 0.5, 0.6
    
    def _analyze_general_evidence(self, content: str, query_text: str, 
                                similarity_score: float) -> Tuple[EvidenceType, float, float]:
        """General evidence analysis based on similarity"""
        if similarity_score > 0.8:
            return EvidenceType.SUPPORTING, 0.9, 0.9
        elif similarity_score > 0.6:
            return EvidenceType.SUPPORTING, 0.7, 0.8
        elif similarity_score > 0.4:
            return EvidenceType.NEUTRAL, 0.5, 0.6
        else:
            return EvidenceType.NEUTRAL, 0.3, 0.4


class LogicEngine:
    """Core logic engine for decision making"""
    
    def __init__(self):
        self.evidence_analyzer = EvidenceAnalyzer()
    
    def evaluate_decision(self, query: Query, evidence_list: List[Evidence]) -> DecisionReasoning:
        """Evaluate evidence and make a decision"""
        
        # Separate evidence by type
        supporting_evidence = [e for e in evidence_list if e.evidence_type == EvidenceType.SUPPORTING]
        contradicting_evidence = [e for e in evidence_list if e.evidence_type == EvidenceType.CONTRADICTING]
        neutral_evidence = [e for e in evidence_list if e.evidence_type == EvidenceType.NEUTRAL]
        
        # Calculate weighted scores
        supporting_score = self._calculate_evidence_score(supporting_evidence)
        contradicting_score = self._calculate_evidence_score(contradicting_evidence)
        neutral_score = self._calculate_evidence_score(neutral_evidence)
        
        # Determine decision based on query type
        decision, confidence = self._make_decision(
            query, supporting_score, contradicting_score, neutral_score
        )
        
        # Generate reasoning
        reasoning = DecisionReasoning(
            decision=decision,
            confidence=confidence,
            supporting_evidence=self._format_evidence_list(supporting_evidence),
            contradicting_evidence=self._format_evidence_list(contradicting_evidence),
            assumptions=self._identify_assumptions(query, evidence_list),
            limitations=self._identify_limitations(evidence_list)
        )
        
        return reasoning
    
    def _calculate_evidence_score(self, evidence_list: List[Evidence]) -> float:
        """Calculate weighted score for a list of evidence"""
        if not evidence_list:
            return 0.0
        
        total_score = sum(e.weight * e.confidence for e in evidence_list)
        max_possible_score = sum(e.weight for e in evidence_list)
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0
    
    def _make_decision(self, query: Query, supporting_score: float, 
                      contradicting_score: float, neutral_score: float) -> Tuple[str, float]:
        """Make a decision based on evidence scores"""
        
        # Decision logic based on query type
        if query.query_type == QueryType.COVERAGE_ANALYSIS:
            return self._coverage_decision(supporting_score, contradicting_score, neutral_score)
        elif query.query_type == QueryType.COMPLIANCE_CHECK:
            return self._compliance_decision(supporting_score, contradicting_score, neutral_score)
        elif query.query_type == QueryType.CLAUSE_MATCHING:
            return self._clause_decision(supporting_score, contradicting_score, neutral_score)
        else:
            return self._general_decision(supporting_score, contradicting_score, neutral_score)
    
    def _coverage_decision(self, supporting: float, contradicting: float, 
                         neutral: float) -> Tuple[str, float]:
        """Make coverage analysis decision"""
        if supporting > 0.7 and contradicting < 0.3:
            return "Yes, this appears to be covered based on the available information.", supporting
        elif contradicting > 0.7 and supporting < 0.3:
            return "No, this does not appear to be covered based on the available information.", contradicting
        elif supporting > contradicting:
            return f"Likely covered, but with some uncertainty. Supporting evidence score: {supporting:.2f}", supporting * 0.8
        elif contradicting > supporting:
            return f"Likely not covered, but with some uncertainty. Contradicting evidence score: {contradicting:.2f}", contradicting * 0.8
        else:
            return "Unclear based on available information. Additional documentation may be needed.", 0.3
    
    def _compliance_decision(self, supporting: float, contradicting: float,
                           neutral: float) -> Tuple[str, float]:
        """Make compliance check decision"""
        if supporting > 0.8:
            return "Compliant based on available evidence.", supporting
        elif contradicting > 0.6:
            return "Non-compliant based on available evidence.", contradicting
        else:
            return "Compliance status unclear - requires further review.", max(supporting, contradicting) * 0.7
    
    def _clause_decision(self, supporting: float, contradicting: float,
                        neutral: float) -> Tuple[str, float]:
        """Make clause matching decision"""
        if supporting > 0.6:
            return "Relevant clauses found that address the query.", supporting
        else:
            return "No clearly relevant clauses found in the available documents.", neutral
    
    def _general_decision(self, supporting: float, contradicting: float,
                         neutral: float) -> Tuple[str, float]:
        """Make general decision"""
        total_score = supporting + neutral * 0.5
        if total_score > 0.7:
            return "Information found that addresses the query.", total_score
        elif total_score > 0.4:
            return "Some relevant information found, but may be incomplete.", total_score
        else:
            return "Limited relevant information found in the available documents.", total_score
    
    def _format_evidence_list(self, evidence_list: List[Evidence]) -> List[str]:
        """Format evidence list for output"""
        formatted = []
        for evidence in evidence_list[:5]:  # Limit to top 5 pieces of evidence
            formatted.append(
                f"{evidence.source}: {evidence.text[:200]}..." 
                if len(evidence.text) > 200 else f"{evidence.source}: {evidence.text}"
            )
        return formatted
    
    def _identify_assumptions(self, query: Query, evidence_list: List[Evidence]) -> List[str]:
        """Identify assumptions made in the analysis"""
        assumptions = []
        
        # Check for assumptions based on query type
        if query.query_type == QueryType.COVERAGE_ANALYSIS:
            assumptions.append("Assumes current policy terms and conditions apply")
            assumptions.append("Assumes no recent policy changes not reflected in documents")
        
        if query.query_type == QueryType.COMPLIANCE_CHECK:
            assumptions.append("Assumes current regulatory requirements apply")
            assumptions.append("Assumes no recent regulatory changes")
        
        # Check for data quality assumptions
        if len(evidence_list) < 3:
            assumptions.append("Limited evidence available - conclusions may not be comprehensive")
        
        if all(e.confidence < 0.7 for e in evidence_list):
            assumptions.append("Evidence confidence is moderate - conclusions should be verified")
        
        return assumptions
    
    def _identify_limitations(self, evidence_list: List[Evidence]) -> List[str]:
        """Identify limitations of the analysis"""
        limitations = []
        
        if len(evidence_list) == 0:
            limitations.append("No relevant evidence found in available documents")
        elif len(evidence_list) < 5:
            limitations.append("Limited evidence available - analysis may not be comprehensive")
        
        if all(e.confidence < 0.6 for e in evidence_list):
            limitations.append("Low confidence in available evidence")
        
        if len(set(e.source for e in evidence_list)) == 1:
            limitations.append("Evidence from limited number of sources")
        
        return limitations


class DecisionEngine:
    """Main decision engine that coordinates analysis and reasoning"""
    
    def __init__(self):
        self.evidence_analyzer = EvidenceAnalyzer()
        self.logic_engine = LogicEngine()
    
    def process_query_results(self, query: Query, retrieved_chunks: List[RetrievedChunk],
                            matched_clauses: List[ClauseMatch]) -> DecisionReasoning:
        """Process query results and generate decision reasoning"""
        start_time = time.time()
        
        try:
            # Analyze evidence
            evidence_list = self.evidence_analyzer.analyze_evidence(
                retrieved_chunks, matched_clauses, query
            )
            
            # Evaluate decision
            reasoning = self.logic_engine.evaluate_decision(query, evidence_list)
            
            processing_time = time.time() - start_time
            logger.info(f"Decision processing completed in {processing_time:.2f} seconds")
            
            return reasoning
        
        except Exception as e:
            logger.error(f"Error in decision processing: {e}")
            raise DecisionEngineError(f"Failed to process decision: {str(e)}")
    
    def validate_decision(self, reasoning: DecisionReasoning) -> Dict[str, Any]:
        """Validate decision reasoning and provide quality metrics"""
        validation = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check confidence level
        if reasoning.confidence < 0.3:
            validation['issues'].append("Very low confidence in decision")
            validation['recommendations'].append("Seek additional documentation or expert review")
        
        # Check evidence balance
        supporting_count = len(reasoning.supporting_evidence)
        contradicting_count = len(reasoning.contradicting_evidence)
        
        if supporting_count == 0 and contradicting_count == 0:
            validation['issues'].append("No evidence found to support decision")
            validation['is_valid'] = False
        
        if supporting_count > 0 and contradicting_count > 0:
            if abs(supporting_count - contradicting_count) > 3:
                validation['issues'].append("Highly imbalanced evidence")
        
        # Calculate quality score
        evidence_quality = min(1.0, (supporting_count + contradicting_count) / 5)
        confidence_quality = reasoning.confidence
        assumption_penalty = min(0.3, len(reasoning.assumptions) * 0.1)
        limitation_penalty = min(0.3, len(reasoning.limitations) * 0.1)
        
        validation['quality_score'] = max(0.0, 
            (evidence_quality + confidence_quality) / 2 - assumption_penalty - limitation_penalty
        )
        
        return validation
    
    def explain_decision_process(self, reasoning: DecisionReasoning) -> Dict[str, Any]:
        """Generate explanation of the decision process"""
        explanation = {
            'process_steps': [
                "1. Analyzed retrieved document chunks for relevant information",
                "2. Identified and classified matched clauses",
                "3. Evaluated evidence as supporting, contradicting, or neutral",
                "4. Applied domain-specific decision logic",
                "5. Generated confidence score based on evidence quality",
                "6. Identified assumptions and limitations"
            ],
            'evidence_summary': {
                'supporting_count': len(reasoning.supporting_evidence),
                'contradicting_count': len(reasoning.contradicting_evidence),
                'confidence_level': reasoning.confidence
            },
            'decision_factors': {
                'primary_factors': reasoning.supporting_evidence[:3],
                'limiting_factors': reasoning.limitations,
                'key_assumptions': reasoning.assumptions
            }
        }
        
        return explanation