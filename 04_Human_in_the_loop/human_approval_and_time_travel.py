import operator
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv
from uuid import uuid4

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    """
    Advanced reducer that merges messages. If a message has the same ID, 
    it replaces the old one, enabling 'Time Travel' and state editing.
    """
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

class HumanInTheLoopAgent:
    def __init__(self, model, tools, checkpointer):
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
        
        # KEY FEATURE: interrupt_before specifies nodes that require human intervention
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        message = self.model.invoke(state['messages'])
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"--- Action Approved! Executing Tool: {t['name']} ---")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

# --- Demo: Approval and State Editing ---
if __name__ == "__main__":
    memory = SqliteSaver.from_conn_string(":memory:")
    abot = HumanInTheLoopAgent(ChatOpenAI(model="gpt-4o"), [TavilySearchResults(max_results=1)], memory)
    config = {"configurable": {"thread_id": "hr_001"}}

    # 1. Trigger an Interrupt
    print("--- 1. Sending Query ---")
    inputs = [HumanMessage(content="What's the weather in SF?")]
    for event in abot.graph.stream({"messages": inputs}, config):
        print(event)

    # The graph is now paused before 'action'
    snapshot = abot.graph.get_state(config)
    print("\n--- Currently Waiting at Node:", snapshot.next, "---")

    # 2. Time Travel / State Modification
    # Let's say we want to change the query to 'London' instead of 'SF' before approving
    last_message = snapshot.values['messages'][-1]
    last_message.tool_calls[0]['args']['query'] = "current weather in London"
    
    print("\n--- 2. Modifying State (Time Travel / Branching) ---")
    abot.graph.update_state(config, {"messages": [last_message]})

    # 3. Resume Execution
    print("\n--- 3. Resuming with Approved/Modified State ---")
    for event in abot.graph.stream(None, config):
        print(event)