import re
from py2neo import Graph #type: ignore
from fuzzywuzzy import process #type: ignore

class RecommendationEngine:
    def __init__(self):
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
        self.symptom_mapping = {
            'fever': ['high_fever', 'mild_fever'],
            'cold': ['cold'],
            'headache': ['headache'],
            'cough': ['dry_cough', 'cough'],
            'body pain': ['body_pain', 'back_pain']
        }

    def recommend(self, query):
        # Extract symptoms using advanced matching
        symptoms = self.extract_symptoms(query)
        print("Extracted Symptoms:", symptoms)

        # Recommend diseases based on symptoms
        recommendations = self.recommend_diseases(symptoms)
        print("Recommended Diseases:", recommendations)

        # Get details for the top recommended disease
        if recommendations:
            top_disease = recommendations[0]['disease']
            details = self.get_disease_details(top_disease)
            diagnosis = self.generate_diagnosis(top_disease, symptoms)
            return top_disease, details, diagnosis
        return None, None, None

    def extract_symptoms(self, query):
        # Get full symptom list from Neo4j
        symptom_list = self.get_symptom_list()
        
        # Normalize query
        query_lower = query.lower()
        
        # Extract symptoms using multiple techniques
        extracted_symptoms = set()
        
        # Direct mapping
        for key, mapped_symptoms in self.symptom_mapping.items():
            if key in query_lower:
                extracted_symptoms.update(mapped_symptoms)
        
        # Fuzzy matching
        for symptom in symptom_list:
            # Check if the full symptom is in the query
            if symptom.lower() in query_lower:
                extracted_symptoms.add(symptom)
            
            # Fuzzy matching with minimum score
            match = process.extractOne(symptom, [query_lower], score_cutoff=80)
            if match:
                extracted_symptoms.add(symptom)
        
        return list(extracted_symptoms)

    def get_symptom_list(self):
        query = """
        MATCH (s:Symptom)
        RETURN s.name AS symptom
        """
        result = self.graph.run(query).data()
        return [row['symptom'] for row in result]

    def recommend_diseases(self, symptoms):
        if not symptoms:
            return []
        
        query = """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH d, COUNT(DISTINCT s) AS matched_symptoms
        MATCH (d)-[:HAS_SYMPTOM]->(total:Symptom)
        WITH d, matched_symptoms, COUNT(DISTINCT total) AS total_symptoms
        WHERE total_symptoms > 0
        RETURN d.name AS disease, 
               matched_symptoms, 
               total_symptoms, 
               (matched_symptoms * 1.0 / total_symptoms) AS match_percentage
        ORDER BY match_percentage DESC
        LIMIT 5
        """
        result = self.graph.run(query, symptoms=symptoms).data()
        return result

    def get_disease_details(self, disease_name):
        query = """
        MATCH (d:Disease {name: $disease_name})
        OPTIONAL MATCH (d)-[:HAS_DESCRIPTION]->(desc:Description)
        OPTIONAL MATCH (d)-[:HAS_PRECAUTION]->(prec:Precaution)
        RETURN 
            COALESCE(desc.text, 'No description available') AS description,
            COLLECT(DISTINCT prec.text) AS precautions
        """
        result = self.graph.run(query, disease_name=disease_name).data()
        return result

    def generate_diagnosis(self, disease, symptoms):
        query = """
        MATCH (d:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        RETURN COLLECT(s.name) AS matched_symptoms
        """
        matched = self.graph.run(query, disease=disease, symptoms=symptoms).data()[0]['matched_symptoms']
        
        # Diagnostic confidence calculation
        confidence = len(matched) / len(symptoms) * 100 if symptoms else 0
        
        # Prepare diagnosis explanation
        diagnosis = {
            'disease': disease,
            'matched_symptoms': matched,
            'confidence_percentage': round(confidence, 2),
            'diagnostic_statement': self._create_diagnostic_statement(disease, matched, confidence)
        }
        
        return diagnosis

    def _create_diagnostic_statement(self, disease, matched_symptoms, confidence):
        confidence_level = (
            "High" if confidence > 80 else
            "Moderate" if confidence > 50 else
            "Low"
        )
        
        return (
            f"Based on the symptoms provided, there is a {confidence_level} likelihood "
            f"of {disease}. Matched symptoms include: {', '.join(matched_symptoms)}. "
            f"Diagnostic confidence: {confidence:.2f}%"
        )

if __name__ == "__main__":
    engine = RecommendationEngine()
    query = input("Enter your symptoms: ")
    disease, details, diagnosis = engine.recommend(query)
    
    if disease and details and diagnosis:
        print(f"\nDisease: {disease}")
        print(f"Description: {details[0]['description']}")
        
        print("\nDiagnosis Details:")
        print(f"Diagnostic Statement: {diagnosis['diagnostic_statement']}")
        print(f"Matched Symptoms: {', '.join(diagnosis['matched_symptoms'])}")
        
        print("\nPrecautions:")
        precautions = details[0]['precautions']
        if precautions and precautions[0]:
            for prec in precautions:
                print(f"- {prec}")
        else:
            print("No precautions found.")