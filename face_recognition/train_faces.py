# train_faces.py

import face_recognition
import os
import logging
import pickle

def train_images(image_dir="TrainingImage"):
    known_face_encodings = []
    known_face_names = []

    logging.info(f"Training Images Directory: {image_dir}")

    if not os.path.exists(image_dir):
        logging.error(f"Training image directory not found: {image_dir}")
        return [], []

    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            try:
                img_path = os.path.join(image_dir, filename)
                image = face_recognition.load_image_file(img_path)

                # Get face encodings
                encodings = face_recognition.face_encodings(image)
                
                if not encodings:
                    logging.warning(f"No face detected in image: {filename}")
                    continue

                encoding = encodings[0]
                known_face_encodings.append(encoding)
                
                # Extract name from filename (assuming format: userID_name.jpg)
                name = filename.split('_')[0]
                known_face_names.append(name)
                
                logging.info(f"Processed Image: {filename} - Name: {name}")

            except Exception as img_error:
                logging.error(f"Error processing image {filename}: {str(img_error)}")

    logging.info(f"Face Training Completed. Total Faces Trained: {len(known_face_names)}")

    # Save encodings and names to a file for later use
    with open("trained_faces.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    return known_face_encodings, known_face_names

def load_pretrained_encodings():
    """Load pre-trained face encodings from a file."""
    if os.path.exists("trained_faces.pkl"):
        with open("trained_faces.pkl", "rb") as f:
            return pickle.load(f)
    else:
        logging.error("No trained faces found!")
        return [], []
