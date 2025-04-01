import cv2
import os
import numpy as np

# Load OpenCV's Face Recognizer and Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def train_voter_recognizer(database_path="voter_database"):
    """
    Train the face recognizer using stored voter images.
    """
    faces = []
    labels = []
    label_map = {}  # Mapping between labels and voter IDs

    # Scan through each voter folder
    for label, voter_id in enumerate(os.listdir(database_path)):
        voter_folder = os.path.join(database_path, voter_id)
        if not os.path.isdir(voter_folder):
            continue

        label_map[label] = voter_id  # Store mapping: Label -> Voter ID

        # Read each face image inside voter's folder
        for img_name in os.listdir(voter_folder):
            img_path = os.path.join(voter_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

            # Detect face
            faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces_detected:
                face = img[y:y+h, x:x+w]  # Crop detected face
                faces.append(face)
                labels.append(label)

    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    
    # Save the trained model and label map
    recognizer.save("voter_recognizer.yml")
    np.save("voter_labels.npy", label_map)

    print("✅ Training complete! Model saved as voter_recognizer.yml")

# Call function to train the model
train_voter_recognizer()
