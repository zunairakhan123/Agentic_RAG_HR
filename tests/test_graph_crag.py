import sys
import os
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import builder # Import the uncompiled builder

async def run_crag_selfrag_demonstration():
    print("======================================================")
    print(" CRAG & SELF-RAG PASS/FAIL DEMONSTRATION")
    print("======================================================\n")

    # Connect to your SQLite checkpointer
    async with AsyncSqliteSaver.from_conn_string("hr_agent_memory.db") as memory:
        
        # Compile the graph with persistent memory
        graph = builder.compile(checkpointer=memory)
        
        # We assign a strict thread ID to track this specific conversation
        config = {"configurable": {"thread_id": "test_crag_001"}}
        
        # A query designed to test the adaptive router and retrieval grading
        # We will use informal wording to see if the grader forces a rewrite
        query = "What happens if I don't use my PTO by December?"
        print(f"[User]: {query}\n")
        
        inputs = {"messages": [HumanMessage(content=query)]}
        
        print("--- GRAPH EXECUTION TRACE ---")
        # Stream the node transitions so we can watch the loops happen
        async for event in graph.astream(inputs, config, stream_mode="updates"):
            for node_name, state_update in event.items():
                print(f"\n[Node Execution]: {node_name.upper()}")
                
                # Print specific telemetry based on which node just ran
                if node_name == "router":
                    print(f"  -> Classified as: {state_update.get('query_type')}")
                elif node_name == "grader":
                    print(f"  -> Relevant: {state_update.get('documents_relevant')}")
                    print(f"  -> Reason: {state_update.get('retrieval_grade_reason')}")
                elif node_name == "rewrite":
                    print(f"  -> New Query: '{state_update.get('rewritten_query')}'")
                elif node_name == "reflection":
                    print(f"  -> Grounded: {state_update.get('answer_grounded')}")
                    print(f"  -> Error Type: {state_update.get('reflection_error_type', 'none')}")
        
        # Fetch the final generated state from SQLite memory
        final_state = await graph.aget_state(config)
        final_message = final_state.values["messages"][-1].content
        
        print("\n======================================================")
        print(" FINAL AI RESPONSE:")
        print("======================================================")
        print(final_message)
        print("\n[✓] State successfully persisted to hr_agent_memory.db")

if __name__ == "__main__":
    # Windows asyncio policy fix to prevent EventLoop errors
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_crag_selfrag_demonstration())