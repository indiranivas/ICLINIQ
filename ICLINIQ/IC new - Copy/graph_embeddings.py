import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0 = all logs, 3 = no logs)

import torch
from torch_geometric.data import Data #type: ignore
from torch_geometric.nn import Node2Vec #type: ignore

def generate_graph_embeddings():
    # Example: Load graph data (replace with actual Neo4j data)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    # Train Node2Vec embeddings
    model = Node2Vec(data.edge_index, embedding_dim=768, walk_length=20, context_size=10, walks_per_node=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = model.loader(batch_size=128, shuffle=True)
    
    for epoch in range(10):
        model.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
    # Create the 'recommendation' directory if it doesn't exist
    os.makedirs('recommendation', exist_ok=True)
    
    # Save embeddings
    embeddings = model.embedding.weight.data
    torch.save(embeddings, 'graph_embeddings.pt')
    print("Graph embeddings saved.")

if __name__ == "__main__":
    generate_graph_embeddings()