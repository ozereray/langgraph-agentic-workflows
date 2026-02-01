import os
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Initialize Tavily Client
# Ensure TAVILY_API_KEY is set in your .env file
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def run_agentic_search(query: str):
    """
    Performs an agentic search using Tavily.
    Unlike regular search, this is optimized for AI agents to retrieve 
    clean, relevant information directly.
    """
    print(f"--- Executing Agentic Search for: {query} ---")
    
    # Executing the search
    # include_answer=True provides a direct LLM-generated answer based on results
    result = tavily.search(query=query, include_answer=True, max_results=3)
    
    # Output the direct answer
    if 'answer' in result:
        print("\n[AI Contextual Answer]:")
        print(result['answer'])
    
    # Output the structured results
    print("\n[Source Results]:")
    for i, res in enumerate(result['results']):
        print(f"{i+1}. {res['title']}")
        print(f"   URL: {res['url']}")
        print(f"   Content Snippet: {res['content'][:200]}...")
        print("-" * 20)

# Example: Manual Web Scraping comparison (Traditional vs Agentic)
# In the lecture, traditional scraping involves requests and BeautifulSoup.
# Agentic search replaces that complexity with a single API call.

if __name__ == "__main__":
    test_query = "What is the current stock price of NVIDIA and recent news?"
    run_agentic_search(test_query)