import cv2
import os
import numpy as np
import json
import subprocess  # To call train_model.py automatically

VOTER_DB_PATH = "voter_database"  # Store images in voter-specific folders
if not os.path.exists(VOTER_DB_PATH):
    os.makedirs(VOTER_DB_PATH)

label_map_file = "voter_labels.npy"

# Load existing label mapping or create a new one
if os.path.exists(label_map_file):
    label_map = np.load(label_map_file, allow_pickle=True).item()
else:
    label_map = {}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_voter():
    voter_id = input("Enter Aadhar Number (as ID): ")
    
    voter_folder = os.path.join(VOTER_DB_PATH, voter_id)  # Create folder for this voter
    if not os.path.exists(voter_folder):
        os.makedirs(voter_folder)
    else:
        print("⚠️ Voter already registered!")
        return

    print(f"\n📸 Capturing face data for Aadhar ID: {voter_id}... Look at the camera.")

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 20:  # Capture 20 images
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            filename = f"{voter_folder}/{count}.jpg"  # Save images inside voter folder
            cv2.imwrite(filename, face)
            print(f"✅ Captured {count}/20")

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    label_map[len(label_map)] = voter_id  # Assign an index label
    np.save(label_map_file, label_map)  # Save label mapping

    print("\n✅ Voter Registered Successfully!")

    # ✅ Call train_model.py automatically
    print("\n🔄 Training model with new data...")
    subprocess.run(["python", "train_model.py"])  # Calls the training script

# Run voter registration
register_voter()