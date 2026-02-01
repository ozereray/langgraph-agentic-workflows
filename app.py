import os
from typing import List, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel, Field

load_dotenv()

# ------------------ ŞEMA ------------------

class Queries(BaseModel):
    queries: List[str] = Field(description="Search queries list")

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

# ------------------ AGENT ------------------

class MasterEssayAgent:
    def __init__(self, model, search_tool):
        self.model = model
        self.search = search_tool

    def plan_node(self, state: AgentState):
        prompt = "You are an expert writer. Create an outline for the essay."
        res = self.model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["task"])
        ])
        return {"plan": res.content}

    def research_node(self, state: AgentState):
        prompt = "Generate 3 research queries based on the plan."

        structured_llm = self.model.with_structured_output(Queries)
        query_results = structured_llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=state["plan"])
        ])

        current_content = state.get("content", [])
        for q in query_results.queries:
            result = self.search.invoke({"query": q})
            current_content.append(str(result))

        return {"content": current_content}

    def generation_node(self, state: AgentState):
        prompt = "Write a high quality essay using the research."

        research_blob = "\n".join(state["content"])
        msg = f"""
TASK:
{state['task']}

PLAN:
{state['plan']}

RESEARCH:
{research_blob}
"""

        res = self.model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=msg)
        ])

        return {
            "draft": res.content,
            "revision_number": state.get("revision_number", 0) + 1
        }

# ------------------ GRAPH ------------------

def build_graph():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    search = TavilySearchResults(max_results=2)

    agent = MasterEssayAgent(model, search)

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", agent.plan_node)
    workflow.add_node("researcher", agent.research_node)
    workflow.add_node("generator", agent.generation_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "generator")

    def router(state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return "end"
        return "researcher"

    workflow.add_conditional_edges(
        "generator",
        router,
        {"researcher": "researcher", "end": END}
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# ------------------ RUN ------------------

if __name__ == "__main__":
    app = build_graph()

    config = RunnableConfig(configurable={"thread_id": "essay_run_01"})

    initial_input: AgentState = {
        "task": "Future of AI Agents and Software Engineering",
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
        "revision_number": 0,
        "max_revisions": 1,
    }

    print("\n🚀 Essay Agent Başlatıldı...\n")

    for event in app.stream(initial_input, config=config):
        node = list(event.keys())[0]
        print(f"✅ Node completed: {node}")

    print("\n🧠 Final Draft:\n")
    final_state = app.get_state(config)
    print(final_state.values["draft"])
