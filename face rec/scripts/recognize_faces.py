import pandas as pd
from deepface import DeepFace
import numpy as np
import ast  


csv_path = "dataset/embeddings.csv"
df = pd.read_csv(csv_path)


def recognize_face(image_path):
    try:
        new_embedding = DeepFace.represent(image_path, model_name="Facenet")[0]['embedding']
        new_embedding = np.array(new_embedding)

        best_match = None
        min_distance = float("inf")

        for index, row in df.iterrows():
            existing_embedding = np.array(ast.literal_eval(row["embedding"]))
            distance = np.linalg.norm(existing_embedding - new_embedding)

            print(f"Comparing with {row['person_name']}, Distance: {distance}")

            if distance < min_distance:
                min_distance = distance
                best_match = row["person_name"]

        threshold = 3.0  
        print(f"Best match: {best_match}, Distance: {min_distance}")

        if min_distance < threshold:
            return best_match
        else:
            return "Unknown"

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error"

    

image_path = "test.jpg"  
result = recognize_face(image_path)
print("Recognized as:", result)