"""
Pydantic models for the LLM-Powered Intelligent Query-Retrieval System
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    TEXT = "text"


class Domain(str, Enum):
    """Supported business domains"""
    INSURANCE = "insurance"
    LEGAL = "legal"
    HR = "hr"
    COMPLIANCE = "compliance"


class QueryType(str, Enum):
    """Types of queries the system can handle"""
    COVERAGE_ANALYSIS = "coverage_analysis"
    CLAUSE_MATCHING = "clause_matching"
    POLICY_LOOKUP = "policy_lookup"
    COMPLIANCE_CHECK = "compliance_check"
    GENERAL_SEARCH = "general_search"


class Document(BaseModel):
    """Document metadata and content"""
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    doc_type: DocumentType = Field(..., description="Type of document")
    domain: Domain = Field(..., description="Business domain")
    source_path: Optional[str] = Field(None, description="Original file path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class DocumentChunk(BaseModel):
    """Chunk of a document for embedding"""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    start_index: int = Field(..., description="Start position in original document")
    end_index: int = Field(..., description="End position in original document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    """User query structure"""
    id: str = Field(..., description="Unique query identifier")
    text: str = Field(..., description="Natural language query")
    query_type: QueryType = Field(default=QueryType.GENERAL_SEARCH)
    domain: Optional[Domain] = Field(None, description="Target domain")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    created_at: datetime = Field(default_factory=datetime.now)


class RetrievedChunk(BaseModel):
    """Retrieved document chunk with similarity score"""
    chunk: DocumentChunk = Field(..., description="Document chunk")
    similarity_score: float = Field(..., description="Similarity score")
    relevance_explanation: Optional[str] = Field(None, description="Why this chunk is relevant")


class ClauseMatch(BaseModel):
    """Matched clause information"""
    clause_text: str = Field(..., description="Matched clause content")
    clause_type: str = Field(..., description="Type of clause")
    document_id: str = Field(..., description="Source document ID")
    confidence_score: float = Field(..., description="Confidence in the match")
    start_position: int = Field(..., description="Start position in document")
    end_position: int = Field(..., description="End position in document")
    context: str = Field(..., description="Surrounding context")


class DecisionReasoning(BaseModel):
    """Reasoning behind a decision"""
    decision: str = Field(..., description="Final decision or answer")
    confidence: float = Field(..., description="Confidence level (0-1)")
    supporting_evidence: List[str] = Field(..., description="Supporting evidence")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Contradicting evidence")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the analysis")


class QueryResponse(BaseModel):
    """Complete query response"""
    query_id: str = Field(..., description="Original query ID")
    query_text: str = Field(..., description="Original query text")
    answer: str = Field(..., description="Direct answer to the query")
    decision_reasoning: DecisionReasoning = Field(..., description="Detailed reasoning")
    retrieved_chunks: List[RetrievedChunk] = Field(..., description="Retrieved relevant chunks")
    matched_clauses: List[ClauseMatch] = Field(default_factory=list, description="Matched clauses")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    created_at: datetime = Field(default_factory=datetime.now)


class SystemMetrics(BaseModel):
    """System performance metrics"""
    total_documents: int = Field(default=0)
    total_chunks: int = Field(default=0)
    total_queries: int = Field(default=0)
    average_response_time: float = Field(default=0.0)
    average_accuracy: float = Field(default=0.0)
    total_tokens_used: int = Field(default=0)
    last_updated: datetime = Field(default_factory=datetime.now)


class ProcessingStatus(BaseModel):
    """Status of document processing"""
    document_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    progress: float = Field(default=0.0, description="Progress percentage")
    message: str = Field(default="", description="Status message")
    error: Optional[str] = Field(None, description="Error message if any")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(None)