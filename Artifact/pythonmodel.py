from sentence_transformers import SentenceTransformer
import pickle

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure the directory exists
import os
os.makedirs("Artifacts", exist_ok=True)

# Save the model as a pickle file
with open("Artifacts/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully to Artifacts/model.pkl")
