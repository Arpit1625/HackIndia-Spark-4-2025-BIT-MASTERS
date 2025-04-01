import cv2
import numpy as np
import os
import json

# Load trained model and label mappings
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = "voter_recognizer.yml"
if os.path.exists(model_path):
    try:
        recognizer.read(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file '{model_path}' does not exist. Please train the model first.")

label_map = np.load("voter_labels.npy", allow_pickle=True).item()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# File to store users who have voted
VOTED_FILE = "voted_list.json"

# Check if voted_list.json exists, if not create an empty file
if not os.path.exists(VOTED_FILE):
    with open(VOTED_FILE, "w") as f:
        json.dump({}, f)

def load_voted_users():
    """Load users who have already voted"""
    with open(VOTED_FILE, "r") as f:
        return json.load(f)

def save_voted_user(voter_id):
    """Save the voter as 'already voted'"""
    voted_users = load_voted_users()
    voted_users[voter_id] = True
    with open(VOTED_FILE, "w") as f:
        json.dump(voted_users, f)

def recognize_voter():
    """
    Recognizes a voter using Aadhar Number & Face Recognition.
    """
    aadhar_number = input("Enter Aadhar Number: ")

    # Check if Aadhar exists in registered voters
    voter_id = None
    for key, value in label_map.items():
        if value == aadhar_number:
            voter_id = key
            break

    if voter_id is None:
        print("❌ Aadhar Number not found! Try again.")
        return

    # Check if the voter has already voted
    voted_users = load_voted_users()
    if aadhar_number in voted_users:
        print(f"⚠️ Aadhar {aadhar_number} has already voted! ❌")
        return

    cap = cv2.VideoCapture(0)  # Open webcam

    print("📸 Look at the camera for Face Verification...")

    # Counter to limit the total time for face recognition
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Check if faces are detected
        if len(faces) == 0:
            print("❌ No face detected. Please try again.")
            cv2.putText(frame, "No face detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Prioritize the largest face detected (if multiple faces detected)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # Extract detected face

            # Recognize the face
            label, confidence = recognizer.predict(face)

            if label == voter_id and confidence < 50:  # Lower confidence = better match
                print(f"✅ Voter Verified: {aadhar_number}")
                cap.release()
                cv2.destroyAllWindows()

                # Allow Voting
                vote_now(aadhar_number)
                return
            else:
                print("❌ Invalid User! Face does not match.")
                cv2.putText(frame, "Invalid User", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show face detection result on the screen
        cv2.imshow("Voter Verification", frame)

        # Check for a timeout (5 seconds of no face detection or invalid face)
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time > 30:  # 30 seconds timeout for verification
            print("⏳ Timeout reached. No face detected or invalid face!")
            cap.release()
            cv2.destroyAllWindows()
            return

        # Wait for user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def vote_now(voter_id):
    """
    Allows the verified voter to cast their vote.
    """
    print("\n🗳️ VOTING OPTIONS:")
    print("1️⃣ Candidate A")
    print("2️⃣ Candidate B")
    print("3️⃣ Candidate C")

    choice = input("Enter your choice (1/2/3): ")

    if choice not in ["1", "2", "3"]:
        print("❌ Invalid choice! Try again.")
        return

    # Save voter as 'already voted'
    save_voted_user(voter_id)
    print(f"\n✅ Vote Registered for Voter: {voter_id}")
    print("🎉 Thank you for voting!")

# Start voter recognition
recognize_voter()