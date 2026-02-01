import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver # Persistence layer
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """The state now persists via the checkpointer."""
    messages: Annotated[list[AnyMessage], operator.add]

class PersistentAgent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        # Compile the graph with a checkpointer for persistence
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"--- Executing Tool: {t['name']} ---")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

# --- Execution with Memory and Streaming ---
if __name__ == "__main__":
    # 1. Setup Persistence (In-memory SQLite for this example)
    memory = SqliteSaver.from_conn_string(":memory:")
    
    # 2. Setup Agent
    tool = TavilySearchResults(max_results=2)
    model = ChatOpenAI(model="gpt-4o")
    abot = PersistentAgent(model, [tool], checkpointer=memory)

    # 3. Thread configuration (This ID is the key to memory)
    thread_config = {"configurable": {"thread_id": "user_123"}}

    # First Interaction
    print("--- First Interaction ---")
    req1 = [HumanMessage(content="What is the weather in San Francisco?")]
    for event in abot.graph.stream({"messages": req1}, thread_config):
        for value in event.values():
            print("Current State Message:", value["messages"][-1].content)

    # Second Interaction (Agent remembers the context)
    print("\n--- Second Interaction (Memory Check) ---")
    req2 = [HumanMessage(content="What about in London?")]
    for event in abot.graph.stream({"messages": req2}, thread_config):
        for value in event.values():
            print("Current State Message:", value["messages"][-1].content)