import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Define the State of the graph
class AgentState(TypedDict):
    """
    Represents the state of our agent. 
    The 'messages' key uses operator.add to append new messages 
    rather than overwriting the existing list.
    """
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    """
    A LangGraph-based agent that integrates an LLM with external tools.
    """
    def __init__(self, model, tools, system=""):
        self.system = system
        
        # Initialize the graph with our state definition
        graph = StateGraph(AgentState)
        
        # Define nodes: the LLM call and the Tool execution
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        
        # Define conditional edges to determine if we should continue or stop
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        
        # After taking an action, always return to the LLM to process results
        graph.add_edge("action", "llm")
        
        # Set the starting point of the graph
        graph.set_entry_point("llm")
        
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        """Check if the last message contains tool calls."""
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        """Invoke the LLM with the current state."""
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        """Execute the requested tools and return the observations."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"--- Calling tool: {t['name']} ---")
            if t['name'] not in self.tools:
                # Handle hallucinated tool names
                print(f"...Bad tool name: {t['name']}...")
                result = "Error: Tool does not exist. Please try again."
            else:
                # Execute the actual tool
                result = self.tools[t['name']].invoke(t['args'])
            
            # Create a ToolMessage to feed back into the graph
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
        print("--- Back to the model ---")
        return {'messages': results}

# --- Initialization and Execution ---
if __name__ == "__main__":
    # Initialize tools and model
    tool = TavilySearchResults(max_results=2)
    model = ChatOpenAI(model="gpt-4o") # Using gpt-4o for better tool performance
    
    system_prompt = """You are a smart research assistant. 
    Use the search engine to look up information when needed."""
    
    abot = Agent(model, [tool], system=system_prompt)

    # Example Query
    query = "What is the current weather in San Francisco?"
    messages = [HumanMessage(content=query)]
    
    # Run the graph
    result = abot.graph.invoke({"messages": messages})
    
    # Print the final response
    print("\nFinal Result:")
    print(result['messages'][-1].content)