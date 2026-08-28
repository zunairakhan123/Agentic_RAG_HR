import os
from src.graph import supervisor_builder, rag_builder  # Import your compiled graphs or builders

# 1. Compile the graphs (without checkpointer needed just for visualization)
super_graph = supervisor_builder.compile()
rag_subgraph = rag_builder.compile()

# --- Export Parent Super-Graph (High-Level Overview) ---
with open("super_graph.png", "wb") as f:
    f.write(super_graph.get_graph().draw_mermaid_png())
print("Saved: super_graph.png")

# --- Export Full Nested Graph (X-Ray Mode - Shows Child Subgraphs Inside) ---
with open("super_graph_xray.png", "wb") as f:
    # xray=True expands the RAG subgraph nodes inside the parent graph
    f.write(super_graph.get_graph(xray=True).draw_mermaid_png())
print("Saved: super_graph_xray.png")

# --- Export Isolated RAG Subgraph ---
with open("rag_subgraph.png", "wb") as f:
    f.write(rag_subgraph.get_graph().draw_mermaid_png())
print("Saved: rag_subgraph.png")