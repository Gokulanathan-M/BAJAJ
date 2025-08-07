"""
Main Query System Orchestrator - LLM-Powered Intelligent Query-Retrieval System
"""
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .models import (
    Document, Query, QueryResponse, QueryType, Domain, 
    DocumentType, SystemMetrics, ProcessingStatus
)
from .config import config
from .document_parsers import DocumentParserFactory, DocumentParsingError
from .embeddings import EmbeddingManager, EmbeddingError
from .llm_processor import LLMProcessor, LLMError
from .clause_matcher import ClauseMatcher, ClauseMatchingError
from .decision_engine import DecisionEngine, DecisionEngineError
from .response_formatter import ResponseFormatter, ResponseFormatterError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuerySystemError(Exception):
    """Custom exception for query system errors"""
    pass


class QuerySystem:
    """
    Main orchestrator for the LLM-Powered Intelligent Query-Retrieval System
    
    This system coordinates document parsing, embedding generation, semantic search,
    clause matching, and decision making to provide intelligent responses to queries
    about insurance, legal, HR, and compliance documents.
    """
    
    def __init__(self, openai_api_key: str = None, index_path: str = None):
        """Initialize the query system with all components"""
        
        self.openai_api_key = openai_api_key or config.openai_api_key
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. LLM features will be limited.")
        
        # Initialize core components
        self.document_parser = DocumentParserFactory()
        self.embedding_manager = EmbeddingManager(index_path=index_path)
        self.llm_processor = LLMProcessor(api_key=self.openai_api_key) if self.openai_api_key else None
        self.clause_matcher = ClauseMatcher(embedding_model=self.embedding_manager.embedding_model)
        self.decision_engine = DecisionEngine()
        self.response_formatter = ResponseFormatter()
        
        # System state
        self.documents: Dict[str, Document] = {}
        self.processing_statuses: Dict[str, ProcessingStatus] = {}
        self.metrics = SystemMetrics()
        
        # Initialize directories
        self._create_directories()
        
        logger.info("Query System initialized successfully")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [config.data_dir, config.index_dir, config.temp_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def add_document(self, file_path: str, domain: Domain = None, 
                    metadata: Dict[str, Any] = None) -> Document:
        """
        Add a document to the system
        
        Args:
            file_path: Path to the document file
            domain: Business domain (will be auto-detected if not provided)
            metadata: Additional metadata for the document
            
        Returns:
            Document: Parsed document object
        """
        start_time = time.time()
        
        try:
            # Parse the document
            logger.info(f"Parsing document: {file_path}")
            document = self.document_parser.parse_document(file_path, metadata)
            
            # Override domain if provided
            if domain:
                document.domain = domain
            
            # Add to embeddings
            logger.info(f"Adding document to vector store: {document.id}")
            chunks = self.embedding_manager.add_document(document)
            
            # Process clauses
            logger.info(f"Extracting clauses from document: {document.id}")
            clauses = self.clause_matcher.process_document(document)
            
            # Store document
            self.documents[document.id] = document
            
            # Update metrics
            self.metrics.total_documents += 1
            self.metrics.total_chunks += len(chunks)
            
            processing_time = time.time() - start_time
            logger.info(f"Document {document.id} processed successfully in {processing_time:.2f} seconds")
            
            return document
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            raise QuerySystemError(f"Failed to add document: {str(e)}")
    
    def add_documents_from_directory(self, directory_path: str, 
                                   domain: Domain = None) -> List[Document]:
        """
        Add multiple documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            domain: Business domain for all documents
            
        Returns:
            List[Document]: List of processed documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise QuerySystemError(f"Directory not found: {directory_path}")
        
        documents = []
        supported_extensions = ['.pdf', '.docx', '.eml', '.msg', '.txt']
        
        for file_path in directory.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    document = self.add_document(str(file_path), domain)
                    documents.append(document)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue
        
        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents
    
    def query(self, query_text: str, domain: Domain = None, 
             query_type: QueryType = None, filters: Dict[str, Any] = None,
             include_explainability: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query and return structured response
        
        Args:
            query_text: Natural language query
            domain: Target domain (will be auto-detected if not provided)
            query_type: Type of query (will be auto-detected if not provided)
            filters: Additional filters for search
            include_explainability: Whether to include explainability information
            
        Returns:
            Dict: Structured JSON response
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing query: {query_text}")
            
            # Step 1: Process and analyze the query
            if self.llm_processor:
                query = self.llm_processor.process_query(query_text, query_id)
                
                # Override domain and query_type if provided
                if domain:
                    query.domain = domain
                if query_type:
                    query.query_type = query_type
                
                # Enhance query for better retrieval
                enhanced_query_text = self.llm_processor.enhance_query_for_retrieval(query)
            else:
                # Fallback without LLM
                query = Query(
                    id=query_id,
                    text=query_text,
                    query_type=query_type or QueryType.GENERAL_SEARCH,
                    domain=domain or Domain.LEGAL,
                    filters=filters or {}
                )
                enhanced_query_text = query_text
            
            # Step 2: Semantic search for relevant chunks
            logger.info("Performing semantic search")
            retrieved_chunks = self.embedding_manager.search(
                enhanced_query_text,
                top_k=config.max_retrieval_docs,
                threshold=config.similarity_threshold,
                filters=filters
            )
            
            # Step 3: Find matching clauses
            logger.info("Matching clauses")
            relevant_documents = list(self.documents.values())
            matched_clauses = self.clause_matcher.match_clauses_for_query(
                query, relevant_documents, top_k=5
            )
            
            # Step 4: Generate contextual answer and reasoning
            if self.llm_processor and retrieved_chunks:
                logger.info("Generating contextual answer")
                answer, supporting_evidence, contradicting_evidence = self.llm_processor.generate_contextual_answer(
                    query, retrieved_chunks
                )
            else:
                answer = "Based on the available documents, here are the relevant findings:"
                supporting_evidence = []
                contradicting_evidence = []
            
            # Step 5: Decision analysis
            logger.info("Analyzing decision")
            decision_reasoning = self.decision_engine.process_query_results(
                query, retrieved_chunks, matched_clauses
            )
            
            # Use LLM answer if available, otherwise use decision reasoning
            if answer and answer != "Based on the available documents, here are the relevant findings:":
                final_answer = answer
                # Update reasoning with LLM evidence
                if supporting_evidence:
                    decision_reasoning.supporting_evidence.extend(supporting_evidence)
                if contradicting_evidence:
                    decision_reasoning.contradicting_evidence.extend(contradicting_evidence)
            else:
                final_answer = decision_reasoning.decision
            
            # Step 6: Format response
            processing_time = time.time() - start_time
            token_usage = self.llm_processor.get_total_token_usage() if self.llm_processor else {}
            
            # Create response object
            response = self.response_formatter.format_query_response(
                query=query,
                answer=final_answer,
                decision_reasoning=decision_reasoning,
                retrieved_chunks=retrieved_chunks,
                matched_clauses=matched_clauses,
                processing_time=processing_time,
                token_usage=token_usage,
                metadata={
                    'enhanced_query': enhanced_query_text,
                    'system_version': '1.0.0',
                    'processing_steps': 6
                }
            )
            
            # Format as JSON
            json_response = self.response_formatter.format_json_response(
                response, 
                include_explainability=include_explainability,
                include_performance=True
            )
            
            # Update metrics
            self.metrics.total_queries += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_queries - 1) + processing_time) /
                self.metrics.total_queries
            )
            if token_usage:
                self.metrics.total_tokens_used += token_usage.get('total_tokens', 0)
            
            logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
            
            return json_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = self.response_formatter.format_error_response(
                error=e, query_id=query_id, query_text=query_text
            )
            return error_response
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Get a document by its ID"""
        return self.documents.get(document_id)
    
    def list_documents(self, domain: Domain = None) -> List[Dict[str, Any]]:
        """List all documents with basic information"""
        documents = []
        for doc in self.documents.values():
            if domain is None or doc.domain == domain:
                documents.append({
                    'id': doc.id,
                    'title': doc.title,
                    'domain': doc.domain.value,
                    'doc_type': doc.doc_type.value,
                    'created_at': doc.created_at.isoformat(),
                    'word_count': doc.metadata.get('word_count', 0)
                })
        return documents
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        try:
            # Update metrics
            embedding_stats = self.embedding_manager.get_stats()
            clause_stats = self.clause_matcher.get_clause_statistics()
            
            self.metrics.total_chunks = embedding_stats.get('total_chunks', 0)
            
            status = self.response_formatter.format_system_status(self.metrics)
            
            # Add component-specific information
            status['components'] = {
                'document_parser': 'operational',
                'embedding_manager': 'operational',
                'llm_processor': 'operational' if self.llm_processor else 'disabled',
                'clause_matcher': 'operational',
                'decision_engine': 'operational'
            }
            
            status['statistics'] = {
                'documents_by_domain': self._get_documents_by_domain(),
                'documents_by_type': self._get_documents_by_type(),
                'embedding_stats': embedding_stats,
                'clause_stats': clause_stats
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_documents_by_domain(self) -> Dict[str, int]:
        """Get document count by domain"""
        domain_counts = {}
        for doc in self.documents.values():
            domain = doc.domain.value
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts
    
    def _get_documents_by_type(self) -> Dict[str, int]:
        """Get document count by type"""
        type_counts = {}
        for doc in self.documents.values():
            doc_type = doc.doc_type.value
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts
    
    def validate_api_connection(self) -> bool:
        """Validate API connections"""
        try:
            if self.llm_processor:
                return self.llm_processor.validate_api_connection()
            return True  # No API to validate
        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return False
    
    def save_system_state(self, path: str = None) -> None:
        """Save system state to disk"""
        try:
            # Save embedding index
            self.embedding_manager.save(path)
            logger.info("System state saved successfully")
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            raise QuerySystemError(f"Failed to save system state: {str(e)}")
    
    def clear_system(self) -> None:
        """Clear all documents and reset system state"""
        try:
            self.documents.clear()
            self.processing_statuses.clear()
            self.embedding_manager.vector_store.clear()
            self.clause_matcher.document_clauses.clear()
            self.metrics = SystemMetrics()
            
            if self.llm_processor:
                self.llm_processor.reset_token_counters()
            
            logger.info("System cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing system: {e}")
            raise QuerySystemError(f"Failed to clear system: {str(e)}")
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics and performance data"""
        try:
            metrics = {
                'system_metrics': {
                    'total_documents': self.metrics.total_documents,
                    'total_chunks': self.metrics.total_chunks,
                    'total_queries': self.metrics.total_queries,
                    'average_response_time': round(self.metrics.average_response_time, 3),
                    'total_tokens_used': self.metrics.total_tokens_used
                },
                'component_metrics': {
                    'embedding_manager': self.embedding_manager.get_stats(),
                    'clause_matcher': self.clause_matcher.get_clause_statistics()
                },
                'performance_metrics': {
                    'documents_by_domain': self._get_documents_by_domain(),
                    'documents_by_type': self._get_documents_by_type(),
                    'memory_usage': self._get_memory_usage()
                }
            }
            
            if self.llm_processor:
                metrics['component_metrics']['llm_processor'] = self.llm_processor.get_total_token_usage()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting detailed metrics: {e}")
            return {'error': str(e)}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get approximate memory usage information"""
        import sys
        
        # Simple memory usage estimation
        total_documents_size = sum(
            len(doc.content) for doc in self.documents.values()
        )
        
        return {
            'total_documents_chars': total_documents_size,
            'total_chunks': len(self.embedding_manager.vector_store.chunks),
            'python_objects': len(self.documents)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            'status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check document parser
            health['components']['document_parser'] = 'healthy'
            
            # Check embedding manager
            if self.embedding_manager.vector_store.size > 0:
                health['components']['embedding_manager'] = 'healthy'
            else:
                health['components']['embedding_manager'] = 'empty'
                health['issues'].append('No documents in embedding index')
                health['recommendations'].append('Add documents to the system')
            
            # Check LLM processor
            if self.llm_processor:
                if self.validate_api_connection():
                    health['components']['llm_processor'] = 'healthy'
                else:
                    health['components']['llm_processor'] = 'unhealthy'
                    health['issues'].append('OpenAI API connection failed')
                    health['recommendations'].append('Check API key and internet connection')
            else:
                health['components']['llm_processor'] = 'disabled'
                health['issues'].append('LLM processor not available')
                health['recommendations'].append('Provide OpenAI API key for full functionality')
            
            # Check clause matcher
            if len(self.clause_matcher.document_clauses) > 0:
                health['components']['clause_matcher'] = 'healthy'
            else:
                health['components']['clause_matcher'] = 'empty'
            
            # Check decision engine
            health['components']['decision_engine'] = 'healthy'
            
            # Overall status
            if health['issues']:
                if any('unhealthy' in status for status in health['components'].values()):
                    health['status'] = 'unhealthy'
                else:
                    health['status'] = 'degraded'
            
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
        
        return health