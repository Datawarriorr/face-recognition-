import cv2
import os
import pandas as pd
from deepface import DeepFace

# Paths
original_path = "Original Image/"  # Your dataset with subfolders
face_path = "Face/"  # Store cropped face images
csv_path = "dataset.csv"  # Read dataset.csv
csv_output = "dataset/embeddings.csv"  # Save embeddings

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ensure face directory exists
os.makedirs(face_path, exist_ok=True)

# Load dataset.csv
df = pd.read_csv(csv_path, header=None, names=["image_name", "person_name"])

data = []

# Iterate through dataset.csv entries
for _, row in df.iterrows():
    image_name, person_name = row["image_name"], row["person_name"]
    img_path = os.path.join(original_path, person_name, image_name)

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {img_path}, unable to read.")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face = img[y:y+h, x:x+w]

        # Resize to 160x160
        face_resized = cv2.resize(face, (160, 160))

        # Save cropped face
        save_path = os.path.join(face_path, f"{person_name}_{image_name}")
        cv2.imwrite(save_path, face_resized)

        # Extract embeddings using DeepFace
        try:
            embedding = DeepFace.represent(save_path, model_name="Facenet")[0]['embedding']
            data.append([image_name, person_name, embedding])
            print(f"Processed: {image_name}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
    else:
        print(f"No face found in {img_path}")

# Save embeddings to CSV
df_output = pd.DataFrame(data, columns=["image_name", "person_name", "embedding"])
df_output.to_csv(csv_output, index=False)
print("Embeddings saved successfully!")
