
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://4b4a985d-d035-4433-82f6-ebf5363dbb5c.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1jmgNue36VmR6woI6TCB7QzftYCa3G7VfVZ7sTHGZx4",
)

print(qdrant_client.get_collections())