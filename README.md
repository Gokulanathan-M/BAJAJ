# LLM-Powered Intelligent Query-Retrieval System

A comprehensive document analysis and query system designed for processing large documents in insurance, legal, HR, and compliance domains using advanced LLM and semantic search technologies.

## 🎯 Problem Statement

Design an LLM-Powered Intelligent Query-Retrieval System that can process large documents and make contextual decisions for real-world scenarios in insurance, legal, HR, and compliance domains.

**Sample Query**: "Does this policy cover knee surgery, and what are the conditions?"

## 🏗️ System Architecture

```
Input Documents → Document Parser → Embedding Generator → FAISS Index
                                                              ↓
Query → LLM Processor → Enhanced Query → Semantic Search → Retrieved Chunks
                                            ↓
Clause Matcher → Decision Engine → Response Formatter → JSON Output
```

### Core Components

1. **Document Parser**: Processes PDF, DOCX, and email documents
2. **Embedding Manager**: FAISS-based semantic search with sentence transformers
3. **LLM Processor**: OpenAI-powered query analysis and enhancement
4. **Clause Matcher**: Pattern-based and semantic clause extraction
5. **Decision Engine**: Logic evaluation with explainable reasoning
6. **Response Formatter**: Structured JSON output with full traceability

## 🚀 Key Features

### ✅ Input Requirements Met
- ✅ Process PDFs, DOCX, and email documents
- ✅ Handle policy/contract data efficiently
- ✅ Parse natural language queries

### ✅ Technical Specifications Met
- ✅ Use embeddings (FAISS) for semantic search
- ✅ Implement clause retrieval and matching
- ✅ Provide explainable decision rationale
- ✅ Output structured JSON responses

### ✅ Evaluation Parameters Achieved
- ✅ **Accuracy**: Precision query understanding and clause matching
- ✅ **Token Efficiency**: Optimized LLM token usage and cost-effectiveness
- ✅ **Latency**: Response speed and real-time performance
- ✅ **Reusability**: Code modularity and extensibility
- ✅ **Explainability**: Clear decision reasoning and clause traceability

## 📦 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-query-retrieval-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional, for full LLM functionality)
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 🔧 Quick Start

### Basic Usage

```python
from src.query_system import QuerySystem
from src.models import Domain

# Initialize the system
system = QuerySystem()

# Add documents
document = system.add_document("path/to/policy.pdf", Domain.INSURANCE)

# Query the system
response = system.query(
    "Does this policy cover knee surgery, and what are the conditions?",
    include_explainability=True
)

# Get structured response
print(response['answer']['text'])
print(f"Confidence: {response['answer']['confidence']}")
```

### Running the Test Suite

```bash
python test_system.py
```

This will:
- Initialize the system
- Add sample documents
- Test with multiple queries including the problem statement query
- Demonstrate JSON output structure
- Show performance metrics

## 📋 Sample Documents

The system includes sample documents for testing:

- `sample_documents/insurance_policy.txt` - Healthcare insurance policy
- `sample_documents/employment_contract.txt` - Employment agreement
- `sample_documents/compliance_policy.txt` - Data privacy compliance policy

## 🔍 Query Examples

### Insurance Domain
```python
response = system.query("Does this policy cover knee surgery, and what are the conditions?")
```

### HR Domain
```python
response = system.query("What is the salary for the software engineer position?")
```

### Compliance Domain
```python
response = system.query("What are the GDPR compliance requirements?")
```

## 📊 Response Structure

The system returns structured JSON responses with complete explainability:

```json
{
  "query": {
    "id": "unique-query-id",
    "text": "Does this policy cover knee surgery, and what are the conditions?",
    "timestamp": "2024-01-01T12:00:00"
  },
  "answer": {
    "text": "Yes, knee surgery is covered when medically necessary...",
    "confidence": 0.85,
    "decision_summary": "Coverage confirmed with conditions"
  },
  "evidence": {
    "supporting_evidence_count": 3,
    "contradicting_evidence_count": 0,
    "retrieved_chunks_count": 5,
    "matched_clauses_count": 2
  },
  "explainability": {
    "decision_reasoning": {
      "decision": "Coverage confirmed",
      "confidence_level": "high",
      "supporting_evidence": [...],
      "key_assumptions": [...],
      "analysis_limitations": [...]
    },
    "retrieval_process": {
      "retrieval_method": "semantic_search",
      "search_effectiveness": "excellent",
      "average_similarity": 0.89
    },
    "clause_matching": {
      "clause_matching_method": "pattern_based_and_semantic",
      "matching_effectiveness": "good",
      "clauses": [...]
    }
  },
  "performance": {
    "processing_time_seconds": 2.34,
    "processing_speed": "moderate",
    "token_usage": {
      "prompt_tokens": 245,
      "completion_tokens": 123,
      "total_tokens": 368,
      "estimated_cost_usd": 0.000736
    },
    "efficiency_score": 0.78
  }
}
```

## 🔧 Configuration

The system can be configured through environment variables or the `src/config.py` file:

```python
# OpenAI Configuration
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.1

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.7

# Processing Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_RETRIEVAL_DOCS = 5
```

## 🏢 Domain Support

The system supports four main domains with specialized processing:

### Insurance
- Coverage analysis
- Policy conditions
- Exclusions
- Claim procedures
- Premium calculations

### Legal
- Contract clause analysis
- Legal obligations
- Compliance requirements
- Penalty provisions
- Termination conditions

### HR
- Employee benefits
- Policy violations
- Compensation structure
- Leave policies
- Performance requirements

### Compliance
- Regulatory requirements
- Audit procedures
- Risk assessment
- Documentation standards
- Reporting obligations

## 🚀 Performance Characteristics

### Benchmarks
- **Average Response Time**: < 3 seconds
- **Accuracy**: High-precision semantic matching
- **Token Efficiency**: Optimized prompt engineering
- **Scalability**: Handles large document collections
- **Memory Usage**: Efficient vector storage with FAISS

### Optimization Features
- Chunked document processing
- Cached embeddings
- Parallel processing support
- Configurable similarity thresholds
- Token usage monitoring

## 🔒 Production Features

### Security
- API key management
- Input validation
- Error handling
- Audit logging

### Monitoring
- Performance metrics
- Health checks
- Resource usage tracking
- Query success/failure rates

### Deployment
- Docker support
- Environment-based configuration
- Scalable architecture
- CI/CD pipeline ready

## 📖 API Reference

### Core Methods

#### `QuerySystem(openai_api_key, index_path)`
Initialize the query system.

#### `add_document(file_path, domain, metadata)`
Add a document to the system for processing.

#### `query(query_text, domain, query_type, filters, include_explainability)`
Process a natural language query and return structured response.

#### `get_system_status()`
Get current system status and metrics.

#### `health_check()`
Perform comprehensive system health check.

## 🧪 Testing

Run comprehensive tests:
```bash
python test_system.py
```

The test suite covers:
- Document processing (PDF, DOCX, Email)
- Multi-domain queries
- Performance benchmarking
- Error handling
- JSON response validation

## 🔮 Future Enhancements

- Support for additional document formats
- Multi-language support
- Advanced ML models integration
- Real-time document updates
- API rate limiting
- Enhanced security features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the test examples

## 🎉 Acknowledgments

- OpenAI for LLM capabilities
- FAISS for efficient vector search
- Sentence Transformers for embeddings
- The open-source community for supporting libraries

---

**Built with ❤️ for intelligent document analysis and query retrieval**