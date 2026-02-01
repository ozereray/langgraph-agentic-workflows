# LangGraph AI Agents Mastery 🚀

This repository is a comprehensive guide and portfolio project based on the **"AI Agents in LangGraph"** course by DeepLearning.AI, featuring Harrison Chase (LangChain) and Rotem Weiss (Tavily). It explores the transition from simple LLM chains to complex, stateful, and cyclic agentic workflows.

## 🛠 Project Structure

The project is organized into progressive modules:

- **`01_Basics/`**: Building a ReAct agent from scratch and introducing LangGraph components (Nodes, Edges, State).
- **`02_State_Management/`**: Advanced persistence using SQLite, conversation threads, and real-time token streaming.
- **`03_Tool_Integration/`**: Leveraging Agentic Search (Tavily) for LLM-optimized information retrieval.
- **`04_Human_in_the_loop/`**: Implementing human-approval gates, state editing, and "Time Travel" debugging.
- **`05_Use_Cases/`**: A full-scale **Essay Writer Agent** utilizing a cyclic Reflection workflow.

## 🌟 Key Concepts Implemented

### 1. Agentic Workflows

Unlike linear chains, these workflows are **iterative**. Agents plan, act, reflect, and use tools to track progress over multiple steps.

### 2. State Management & Persistence

Using `Annotated` types and `SqliteSaver` to give agents long-term memory and the ability to resume conversations across different threads.

### 3. Human-in-the-Loop

Strategic `interrupt_before` points that allow humans to approve or modify agent actions (e.g., confirming a search query or a financial transaction).

### 4. Time Travel (State Manipulation)

The ability to go back in history, fork the state, and re-run agentic logic from a specific point in time for debugging and steering.

## 🚀 Getting Started

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yourusername/LangGraph-Agent-Mastery.git](https://github.com/yourusername/LangGraph-Agent-Mastery.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    Create a `.env` file and add:
    ```env
    OPENAI_API_KEY=your_key
    TAVILY_API_KEY=your_key
    ```

## 📚 Acknowledgments

Special thanks to **Harrison Chase** and **Rotem Weiss** for the amazing insights provided in the DeepLearning.AI course.
