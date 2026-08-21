import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import builder

async def test_single_turn():
    async with AsyncSqliteSaver.from_conn_string("test_memory.db") as memory:
        await memory.setup()
        hr_graph = builder.compile(checkpointer=memory)

        query = "How many sick or casual leaves are permanent employees entitled to in a year?"
        print(f"\n[Test Query]: {query}\n")

        config = {"configurable": {"thread_id": "test_thread_001"}}
        result = await hr_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        messages = result["messages"]
        print("--- Message Flow ---")
        for msg in messages:
            msg_type = getattr(msg, "type", type(msg).__name__)
            name_attr = f" ({getattr(msg, 'name', '')})" if hasattr(msg, "name") and msg.name else ""
            print(f"- {msg_type}{name_attr}: {str(msg.content)[:100]}...")

        print("\n--- Final Agent Response ---")
        print(messages[-1].content)

if __name__ == "__main__":
    asyncio.run(test_single_turn())