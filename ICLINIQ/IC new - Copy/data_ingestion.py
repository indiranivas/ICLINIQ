import pandas as pd
from py2neo import Graph, Node, Relationship #type: ignore

def ingest_data():
    # Load datasets
    dataset = pd.read_csv('dataset.csv')
    symptom_description = pd.read_csv('symptom_description.csv')
    symptom_precaution = pd.read_csv('symptom_precaution.csv')
    symptom_severity = pd.read_csv('symptom_severity.csv')

    # Connect to Neo4j
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

    # Clear existing data
    graph.run("MATCH (n) DETACH DELETE n")

    # Ingest diseases and symptoms
    for _, row in dataset.iterrows():
        disease = Node("Disease", name=row['Disease'])  # Create Disease node
        graph.merge(disease, "Disease", "name")

        for symptom in row[1:]:
            if pd.notna(symptom) and symptom.strip():  # Skip NaN and empty strings
                symptom_node = Node("Symptom", name=symptom.strip())  # Create Symptom node
                graph.merge(symptom_node, "Symptom", "name")
                rel = Relationship(disease, "HAS_SYMPTOM", symptom_node)  # Create relationship
                graph.create(rel)

    # Ingest symptom descriptions
    for _, row in symptom_description.iterrows():
        disease = Node("Disease", name=row['Disease'])
        description = Node("Description", text=row['Description'])
        graph.merge(disease, "Disease", "name")
        graph.merge(description, "Description", "text")
        rel = Relationship(disease, "HAS_DESCRIPTION", description)
        graph.create(rel)

    # Ingest symptom precautions
    for _, row in symptom_precaution.iterrows():
        disease = Node("Disease", name=row['Disease'])
        for i in range(1, 5):
            precaution = row[f'Precaution_{i}']
            if pd.notna(precaution) and precaution.strip():  # Skip NaN and empty strings
                precaution_node = Node("Precaution", text=precaution.strip())
                graph.merge(precaution_node, "Precaution", "text")
                rel = Relationship(disease, "HAS_PRECAUTION", precaution_node)
                graph.create(rel)

    # Ingest symptom severity
    for _, row in symptom_severity.iterrows():
        symptom = Node("Symptom", name=row['Symptom'])
        severity = Node("Severity", weight=row['weight'])
        graph.merge(symptom, "Symptom", "name")
        graph.merge(severity, "Severity", "weight")
        rel = Relationship(symptom, "HAS_SEVERITY", severity)
        graph.create(rel)

    print("Data ingested into Neo4j.")

if __name__ == "__main__":
    ingest_data()