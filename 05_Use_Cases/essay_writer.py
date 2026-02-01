import operator
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel

# Load environment variables
load_dotenv()

# --- Schema Definitions ---
class Queries(BaseModel):
    """Structured output for research queries."""
    queries: List[str]

class AgentState(TypedDict):
    """The complex state required for an iterative writing process."""
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# --- Prompts ---
PLAN_PROMPT = "You are an expert writer. Write a high-level outline for an essay based on the user request."
WRITER_PROMPT = "You are an essay assistant. Generate the best 5-paragraph essay possible. Use the provided research content."
REFLECTION_PROMPT = "You are a teacher grading an essay. Provide detailed critique and recommendations for improvement."
RESEARCH_PLAN_PROMPT = "You are a researcher. Generate 3 search queries to gather information for the following essay plan."

# --- Node Functions ---
class EssayAgent:
    def __init__(self, model, search_tool):
        self.model = model
        self.search = search_tool

    def plan_node(self, state: AgentState):
        messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state['task'])]
        response = self.model.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState):
        # Using structured output to get clean search queries
        structured_llm = self.model.with_structured_output(Queries)
        queries = structured_llm.invoke([SystemMessage(content=RESEARCH_PLAN_PROMPT), HumanMessage(content=state['plan'])])
        
        content = state.get('content', [])
        for q in queries.queries:
            res = self.search.invoke(q)
            content.append(str(res))
        return {"content": content}

    def generation_node(self, state: AgentState):
        content_str = "\n".join(state['content'])
        user_msg = f"Task: {state['task']}\nPlan: {state['plan']}\nResearch: {content_str}"
        response = self.model.invoke([SystemMessage(content=WRITER_PROMPT), HumanMessage(content=user_msg)])
        return {"draft": response.content, "revision_number": state.get("revision_number", 0) + 1}

    def reflection_node(self, state: AgentState):
        response = self.model.invoke([SystemMessage(content=REFLECTION_PROMPT), HumanMessage(content=state['draft'])])
        return {"critique": response.content}

    def should_continue(self, state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

# --- Graph Construction ---
def build_essay_graph():
    memory = SqliteSaver.from_conn_string(":memory:")
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    search = TavilySearchResults(max_results=2)
    
    essay_bot = EssayAgent(model, search)
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("planner", essay_bot.plan_node)
    builder.add_node("researcher", essay_bot.research_plan_node)
    builder.add_node("generator", essay_bot.generation_node)
    builder.add_node("reflect", essay_bot.reflection_node)

    # Define Edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "generator")
    
    builder.add_conditional_edges(
        "generator",
        essay_bot.should_continue,
        {"reflect": "reflect", END: END}
    )
    builder.add_edge("reflect", "researcher") # Loop back for more research based on critique

    return builder.compile(checkpointer=memory)

# --- Execution ---
if __name__ == "__main__":
    graph = build_essay_graph()
    thread = {"configurable": {"thread_id": "essay_001"}}
    
    initial_state = {
        "task": "Write an essay about the impact of LangGraph on AI Agent development.",
        "max_revisions": 2,
        "revision_number": 0,
        "content": []
    }

    for event in graph.stream(initial_state, thread):
        print(f"\n--- Node: {list(event.keys())[0]} ---")
        # To avoid massive console output, we just print the event keys