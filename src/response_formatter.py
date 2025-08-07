"""
JSON response formatter with explainability features
"""
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .models import (
    Query, QueryResponse, RetrievedChunk, ClauseMatch, 
    DecisionReasoning, SystemMetrics
)
from .config import config

logger = logging.getLogger(__name__)


class ResponseFormatterError(Exception):
    """Custom exception for response formatting errors"""
    pass


class ExplainabilityFormatter:
    """Formats explainability information for responses"""
    
    def format_decision_explanation(self, reasoning: DecisionReasoning) -> Dict[str, Any]:
        """Format decision reasoning for JSON output"""
        return {
            "decision": reasoning.decision,
            "confidence_score": round(reasoning.confidence, 3),
            "confidence_level": self._get_confidence_level(reasoning.confidence),
            "supporting_evidence": [
                self._format_evidence(evidence) for evidence in reasoning.supporting_evidence
            ],
            "contradicting_evidence": [
                self._format_evidence(evidence) for evidence in reasoning.contradicting_evidence
            ],
            "key_assumptions": reasoning.assumptions,
            "analysis_limitations": reasoning.limitations,
            "evidence_summary": {
                "total_supporting": len(reasoning.supporting_evidence),
                "total_contradicting": len(reasoning.contradicting_evidence),
                "evidence_balance": self._calculate_evidence_balance(reasoning)
            }
        }
    
    def format_retrieval_explanation(self, retrieved_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Format retrieval process explanation"""
        if not retrieved_chunks:
            return {
                "retrieval_method": "semantic_search",
                "total_chunks_found": 0,
                "search_effectiveness": "no_results",
                "chunks": []
            }
        
        return {
            "retrieval_method": "semantic_search",
            "total_chunks_found": len(retrieved_chunks),
            "search_effectiveness": self._assess_search_effectiveness(retrieved_chunks),
            "average_similarity": round(
                sum(chunk.similarity_score for chunk in retrieved_chunks) / len(retrieved_chunks), 3
            ),
            "chunks": [
                {
                    "document_id": chunk.chunk.document_id,
                    "similarity_score": round(chunk.similarity_score, 3),
                    "relevance_explanation": chunk.relevance_explanation,
                    "content_preview": chunk.chunk.content[:200] + "..." if len(chunk.chunk.content) > 200 else chunk.chunk.content,
                    "chunk_metadata": {
                        "word_count": chunk.chunk.metadata.get("word_count", 0),
                        "start_position": chunk.chunk.start_index,
                        "end_position": chunk.chunk.end_index
                    }
                }
                for chunk in retrieved_chunks
            ]
        }
    
    def format_clause_explanation(self, matched_clauses: List[ClauseMatch]) -> Dict[str, Any]:
        """Format clause matching explanation"""
        if not matched_clauses:
            return {
                "clause_matching_method": "pattern_based",
                "total_clauses_found": 0,
                "matching_effectiveness": "no_results",
                "clauses": []
            }
        
        return {
            "clause_matching_method": "pattern_based_and_semantic",
            "total_clauses_found": len(matched_clauses),
            "matching_effectiveness": self._assess_clause_effectiveness(matched_clauses),
            "average_confidence": round(
                sum(clause.confidence_score for clause in matched_clauses) / len(matched_clauses), 3
            ),
            "clauses": [
                {
                    "clause_type": clause.clause_type,
                    "document_id": clause.document_id,
                    "confidence_score": round(clause.confidence_score, 3),
                    "clause_text": clause.clause_text,
                    "context_preview": clause.context[:200] + "..." if len(clause.context) > 200 else clause.context,
                    "position": {
                        "start": clause.start_position,
                        "end": clause.end_position
                    }
                }
                for clause in matched_clauses
            ]
        }
    
    def _format_evidence(self, evidence: str) -> Dict[str, Any]:
        """Format a single piece of evidence"""
        # Extract source information from evidence string
        if ": " in evidence:
            source, content = evidence.split(": ", 1)
        else:
            source = "Unknown"
            content = evidence
        
        return {
            "source": source,
            "content": content,
            "content_length": len(content),
            "excerpt": content[:150] + "..." if len(content) > 150 else content
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to categorical level"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "moderate"
        else:
            return "low"
    
    def _calculate_evidence_balance(self, reasoning: DecisionReasoning) -> str:
        """Calculate evidence balance assessment"""
        supporting = len(reasoning.supporting_evidence)
        contradicting = len(reasoning.contradicting_evidence)
        
        if supporting == 0 and contradicting == 0:
            return "no_evidence"
        elif contradicting == 0:
            return "strongly_supporting"
        elif supporting == 0:
            return "strongly_contradicting"
        elif abs(supporting - contradicting) <= 1:
            return "balanced"
        elif supporting > contradicting:
            return "mostly_supporting"
        else:
            return "mostly_contradicting"
    
    def _assess_search_effectiveness(self, chunks: List[RetrievedChunk]) -> str:
        """Assess the effectiveness of the search"""
        if not chunks:
            return "no_results"
        
        avg_similarity = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
        
        if avg_similarity >= 0.8:
            return "excellent"
        elif avg_similarity >= 0.6:
            return "good"
        elif avg_similarity >= 0.4:
            return "moderate"
        else:
            return "poor"
    
    def _assess_clause_effectiveness(self, clauses: List[ClauseMatch]) -> str:
        """Assess the effectiveness of clause matching"""
        if not clauses:
            return "no_results"
        
        avg_confidence = sum(clause.confidence_score for clause in clauses) / len(clauses)
        
        if avg_confidence >= 0.8:
            return "excellent"
        elif avg_confidence >= 0.6:
            return "good"
        elif avg_confidence >= 0.4:
            return "moderate"
        else:
            return "poor"


class PerformanceFormatter:
    """Formats performance and metrics information"""
    
    def format_performance_metrics(self, processing_time: float, 
                                 token_usage: Dict[str, int]) -> Dict[str, Any]:
        """Format performance metrics"""
        return {
            "processing_time_seconds": round(processing_time, 3),
            "processing_speed": self._categorize_speed(processing_time),
            "token_usage": {
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "estimated_cost_usd": self._estimate_cost(token_usage)
            },
            "efficiency_score": self._calculate_efficiency_score(processing_time, token_usage)
        }
    
    def _categorize_speed(self, processing_time: float) -> str:
        """Categorize processing speed"""
        if processing_time < 2.0:
            return "fast"
        elif processing_time < 5.0:
            return "moderate"
        elif processing_time < 10.0:
            return "slow"
        else:
            return "very_slow"
    
    def _estimate_cost(self, token_usage: Dict[str, int]) -> float:
        """Estimate cost based on token usage (simplified)"""
        # Simple cost estimation (actual costs vary by model)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        # Rough estimates for GPT-3.5-turbo
        prompt_cost = prompt_tokens * 0.0015 / 1000
        completion_cost = completion_tokens * 0.002 / 1000
        
        return round(prompt_cost + completion_cost, 6)
    
    def _calculate_efficiency_score(self, processing_time: float, 
                                  token_usage: Dict[str, int]) -> float:
        """Calculate efficiency score (0-1)"""
        # Simple efficiency calculation
        time_score = max(0, 1 - (processing_time / 10))  # 10 seconds = 0 score
        token_score = max(0, 1 - (token_usage.get("total_tokens", 0) / 5000))  # 5000 tokens = 0 score
        
        return round((time_score + token_score) / 2, 3)


class ResponseFormatter:
    """Main response formatter that creates structured JSON responses"""
    
    def __init__(self):
        self.explainability_formatter = ExplainabilityFormatter()
        self.performance_formatter = PerformanceFormatter()
    
    def format_query_response(self, query: Query, answer: str, 
                            decision_reasoning: DecisionReasoning,
                            retrieved_chunks: List[RetrievedChunk],
                            matched_clauses: List[ClauseMatch],
                            processing_time: float,
                            token_usage: Dict[str, int],
                            metadata: Dict[str, Any] = None) -> QueryResponse:
        """Format complete query response"""
        
        try:
            # Create the response object
            response = QueryResponse(
                query_id=query.id,
                query_text=query.text,
                answer=answer,
                decision_reasoning=decision_reasoning,
                retrieved_chunks=retrieved_chunks,
                matched_clauses=matched_clauses,
                processing_time=processing_time,
                token_usage=token_usage,
                metadata=metadata or {}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting query response: {e}")
            raise ResponseFormatterError(f"Failed to format response: {str(e)}")
    
    def format_json_response(self, response: QueryResponse, 
                           include_explainability: bool = True,
                           include_performance: bool = True) -> Dict[str, Any]:
        """Format response as detailed JSON with explainability"""
        
        try:
            # Base response structure
            json_response = {
                "query": {
                    "id": response.query_id,
                    "text": response.query_text,
                    "timestamp": response.created_at.isoformat()
                },
                "answer": {
                    "text": response.answer,
                    "confidence": round(response.decision_reasoning.confidence, 3),
                    "decision_summary": response.decision_reasoning.decision
                },
                "evidence": {
                    "supporting_evidence_count": len(response.decision_reasoning.supporting_evidence),
                    "contradicting_evidence_count": len(response.decision_reasoning.contradicting_evidence),
                    "retrieved_chunks_count": len(response.retrieved_chunks),
                    "matched_clauses_count": len(response.matched_clauses)
                }
            }
            
            # Add explainability information
            if include_explainability:
                json_response["explainability"] = {
                    "decision_reasoning": self.explainability_formatter.format_decision_explanation(
                        response.decision_reasoning
                    ),
                    "retrieval_process": self.explainability_formatter.format_retrieval_explanation(
                        response.retrieved_chunks
                    ),
                    "clause_matching": self.explainability_formatter.format_clause_explanation(
                        response.matched_clauses
                    )
                }
            
            # Add performance metrics
            if include_performance:
                json_response["performance"] = self.performance_formatter.format_performance_metrics(
                    response.processing_time, response.token_usage
                )
            
            # Add metadata
            if response.metadata:
                json_response["metadata"] = response.metadata
            
            return json_response
            
        except Exception as e:
            logger.error(f"Error formatting JSON response: {e}")
            raise ResponseFormatterError(f"Failed to format JSON response: {str(e)}")
    
    def format_summary_response(self, response: QueryResponse) -> Dict[str, Any]:
        """Format a concise summary response"""
        
        return {
            "query_id": response.query_id,
            "answer": response.answer,
            "confidence": round(response.decision_reasoning.confidence, 3),
            "evidence_summary": {
                "supporting_evidence": len(response.decision_reasoning.supporting_evidence),
                "contradicting_evidence": len(response.decision_reasoning.contradicting_evidence),
                "total_sources": len(response.retrieved_chunks) + len(response.matched_clauses)
            },
            "processing_time": round(response.processing_time, 2),
            "timestamp": response.created_at.isoformat()
        }
    
    def format_error_response(self, error: Exception, query_id: str = None,
                            query_text: str = None) -> Dict[str, Any]:
        """Format error response"""
        
        return {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "query_id": query_id,
            "query_text": query_text,
            "timestamp": datetime.now().isoformat(),
            "suggestions": self._get_error_suggestions(error)
        }
    
    def format_system_status(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Format system status and metrics"""
        
        return {
            "system_status": "operational",
            "metrics": {
                "total_documents_processed": metrics.total_documents,
                "total_chunks_indexed": metrics.total_chunks,
                "total_queries_processed": metrics.total_queries,
                "average_response_time": round(metrics.average_response_time, 3),
                "average_accuracy": round(metrics.average_accuracy, 3),
                "total_tokens_used": metrics.total_tokens_used
            },
            "last_updated": metrics.last_updated.isoformat()
        }
    
    def _get_error_suggestions(self, error: Exception) -> List[str]:
        """Get suggestions based on error type"""
        suggestions = []
        
        error_type = type(error).__name__
        
        if "parsing" in str(error).lower():
            suggestions.extend([
                "Check if the document format is supported (PDF, DOCX, Email)",
                "Ensure the document is not corrupted or password-protected",
                "Try uploading a different document"
            ])
        elif "api" in str(error).lower() or "openai" in str(error).lower():
            suggestions.extend([
                "Check your OpenAI API key configuration",
                "Verify internet connectivity",
                "Try again in a few moments"
            ])
        elif "embedding" in str(error).lower():
            suggestions.extend([
                "The embedding model may be temporarily unavailable",
                "Try restarting the system",
                "Check system resources"
            ])
        else:
            suggestions.extend([
                "Try rephrasing your query",
                "Check if relevant documents have been uploaded",
                "Contact system administrator if the problem persists"
            ])
        
        return suggestions
    
    def validate_response(self, json_response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the JSON response structure"""
        validation = {
            "is_valid": True,
            "missing_fields": [],
            "invalid_fields": [],
            "warnings": []
        }
        
        # Required fields
        required_fields = ["query", "answer", "evidence"]
        for field in required_fields:
            if field not in json_response:
                validation["missing_fields"].append(field)
                validation["is_valid"] = False
        
        # Validate answer structure
        if "answer" in json_response:
            answer = json_response["answer"]
            if "confidence" in answer and not (0 <= answer["confidence"] <= 1):
                validation["invalid_fields"].append("answer.confidence")
        
        # Check for empty results
        if "evidence" in json_response:
            evidence = json_response["evidence"]
            total_evidence = (
                evidence.get("retrieved_chunks_count", 0) + 
                evidence.get("matched_clauses_count", 0)
            )
            if total_evidence == 0:
                validation["warnings"].append("No evidence found - answer may be unreliable")
        
        return validation