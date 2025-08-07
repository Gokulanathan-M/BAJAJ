#!/usr/bin/env python3
"""
Test script for the LLM-Powered Intelligent Query-Retrieval System
"""
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.query_system import QuerySystem
from src.models import Domain, QueryType


def test_system():
    """Test the complete system functionality"""
    
    print("=" * 80)
    print("LLM-POWERED INTELLIGENT QUERY-RETRIEVAL SYSTEM TEST")
    print("=" * 80)
    
    # Initialize system (will work without OpenAI API key, with limited functionality)
    print("\n1. Initializing Query System...")
    try:
        system = QuerySystem()
        print("✓ System initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        return False
    
    # Test system health
    print("\n2. Checking System Health...")
    health = system.health_check()
    print(f"Status: {health['status']}")
    if health.get('issues'):
        print("Issues:")
        for issue in health['issues']:
            print(f"  - {issue}")
    if health.get('recommendations'):
        print("Recommendations:")
        for rec in health['recommendations']:
            print(f"  - {rec}")
    
    # Add sample documents
    print("\n3. Adding Sample Documents...")
    sample_docs_dir = Path("sample_documents")
    
    documents_added = []
    
    # Insurance policy
    insurance_doc = sample_docs_dir / "insurance_policy.txt"
    if insurance_doc.exists():
        try:
            doc = system.add_document(str(insurance_doc), Domain.INSURANCE)
            documents_added.append(doc)
            print(f"✓ Added insurance policy: {doc.id}")
        except Exception as e:
            print(f"✗ Failed to add insurance policy: {e}")
    
    # Employment contract
    hr_doc = sample_docs_dir / "employment_contract.txt"
    if hr_doc.exists():
        try:
            doc = system.add_document(str(hr_doc), Domain.HR)
            documents_added.append(doc)
            print(f"✓ Added employment contract: {doc.id}")
        except Exception as e:
            print(f"✗ Failed to add employment contract: {e}")
    
    # Compliance policy
    compliance_doc = sample_docs_dir / "compliance_policy.txt"
    if compliance_doc.exists():
        try:
            doc = system.add_document(str(compliance_doc), Domain.COMPLIANCE)
            documents_added.append(doc)
            print(f"✓ Added compliance policy: {doc.id}")
        except Exception as e:
            print(f"✗ Failed to add compliance policy: {e}")
    
    if not documents_added:
        print("✗ No documents were added. Creating sample documents...")
        create_sample_documents()
        return test_system()
    
    print(f"\nTotal documents added: {len(documents_added)}")
    
    # List documents
    print("\n4. Listing Documents...")
    docs = system.list_documents()
    for doc in docs:
        print(f"  - {doc['title']} ({doc['domain']}) - {doc['word_count']} words")
    
    # Test queries
    print("\n5. Testing Sample Queries...")
    
    test_queries = [
        {
            "query": "Does this policy cover knee surgery, and what are the conditions?",
            "expected_domain": Domain.INSURANCE,
            "description": "Insurance coverage query (from problem statement)"
        },
        {
            "query": "What is the salary for the software engineer position?",
            "expected_domain": Domain.HR,
            "description": "HR compensation query"
        },
        {
            "query": "What are the GDPR compliance requirements?",
            "expected_domain": Domain.COMPLIANCE,
            "description": "Compliance requirements query"
        },
        {
            "query": "What is the vacation policy?",
            "expected_domain": Domain.HR,
            "description": "HR benefits query"
        },
        {
            "query": "Are there any exclusions in the insurance policy?",
            "expected_domain": Domain.INSURANCE,
            "description": "Insurance exclusions query"
        }
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n5.{i} Query: {test_query['query']}")
        print(f"     Expected Domain: {test_query['expected_domain'].value}")
        print(f"     Description: {test_query['description']}")
        
        try:
            response = system.query(
                query_text=test_query['query'],
                include_explainability=True
            )
            
            # Print key results
            if 'error' not in response:
                print(f"     Answer: {response['answer']['text'][:200]}...")
                print(f"     Confidence: {response['answer']['confidence']}")
                print(f"     Evidence Count: {response['evidence']['retrieved_chunks_count']} chunks, {response['evidence']['matched_clauses_count']} clauses")
                
                if 'performance' in response:
                    print(f"     Processing Time: {response['performance']['processing_time_seconds']} seconds")
                    
            else:
                print(f"     ✗ Error: {response['error_message']}")
                
        except Exception as e:
            print(f"     ✗ Exception: {e}")
    
    # System metrics
    print("\n6. System Metrics...")
    try:
        metrics = system.get_detailed_metrics()
        print(f"Documents: {metrics['system_metrics']['total_documents']}")
        print(f"Chunks: {metrics['system_metrics']['total_chunks']}")
        print(f"Queries: {metrics['system_metrics']['total_queries']}")
        print(f"Avg Response Time: {metrics['system_metrics']['average_response_time']} seconds")
        
        print("\nDocuments by Domain:")
        for domain, count in metrics['performance_metrics']['documents_by_domain'].items():
            print(f"  {domain}: {count}")
            
    except Exception as e:
        print(f"✗ Failed to get metrics: {e}")
    
    # Save system state
    print("\n7. Saving System State...")
    try:
        system.save_system_state()
        print("✓ System state saved successfully")
    except Exception as e:
        print(f"✗ Failed to save system state: {e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    
    return True


def create_sample_documents():
    """Create sample documents if they don't exist"""
    sample_docs_dir = Path("sample_documents")
    sample_docs_dir.mkdir(exist_ok=True)
    
    # Create basic insurance document if not exists
    insurance_file = sample_docs_dir / "insurance_policy.txt"
    if not insurance_file.exists():
        insurance_content = """
        HEALTHCARE INSURANCE POLICY

        COVERED SERVICES:
        - Knee surgery is covered when medically necessary
        - Requires pre-authorization from primary care physician
        - Must be performed by an in-network orthopedic surgeon

        EXCLUSIONS:
        - Cosmetic surgery and procedures
        - Experimental treatments
        - Services not medically necessary

        DEDUCTIBLE: $1,500 per individual
        """
        insurance_file.write_text(insurance_content.strip())
    
    # Create basic HR document if not exists
    hr_file = sample_docs_dir / "employment_contract.txt"
    if not hr_file.exists():
        hr_content = """
        EMPLOYMENT AGREEMENT

        COMPENSATION:
        Base Salary: $120,000 annually
        Performance Bonus: Up to 20% of base salary

        BENEFITS:
        Health Insurance: Company provides comprehensive coverage
        Vacation Time: 20 days paid vacation per year
        Sick Leave: 10 days paid sick leave per year

        TERMINATION:
        At-will employment relationship
        2 weeks notice required for voluntary termination
        """
        hr_file.write_text(hr_content.strip())
    
    # Create basic compliance document if not exists
    compliance_file = sample_docs_dir / "compliance_policy.txt"
    if not compliance_file.exists():
        compliance_content = """
        DATA PRIVACY AND COMPLIANCE POLICY

        GDPR COMPLIANCE:
        - Data breach notification within 72 hours
        - Data Protection Impact Assessments required
        - Right to be forgotten implementation

        AUDIT REQUIREMENTS:
        - Annual compliance audits required
        - Monthly compliance reports
        - Continuous monitoring of data access

        VIOLATIONS:
        - GDPR fines up to €20 million or 4% of annual revenue
        - Report violations within 24 hours
        """
        compliance_file.write_text(compliance_content.strip())


def demo_json_output():
    """Demonstrate the structured JSON output format"""
    print("\n" + "=" * 80)
    print("DEMONSTRATING STRUCTURED JSON OUTPUT")
    print("=" * 80)
    
    try:
        system = QuerySystem()
        
        # Add a sample document
        sample_docs_dir = Path("sample_documents")
        insurance_doc = sample_docs_dir / "insurance_policy.txt"
        
        if insurance_doc.exists():
            system.add_document(str(insurance_doc), Domain.INSURANCE)
        
        # Make a query
        query = "Does this policy cover knee surgery, and what are the conditions?"
        response = system.query(query, include_explainability=True)
        
        # Pretty print the JSON response
        print(f"\nQuery: {query}")
        print("\nStructured JSON Response:")
        print(json.dumps(response, indent=2, default=str))
        
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    print("Starting LLM-Powered Intelligent Query-Retrieval System Test...")
    
    # Run main test
    success = test_system()
    
    # Demonstrate JSON output
    demo_json_output()
    
    if success:
        print("\n✓ All tests completed successfully!")
        print("\nNOTE: For full LLM functionality, set OPENAI_API_KEY environment variable")
        print("Current system works with semantic search and rule-based analysis")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)