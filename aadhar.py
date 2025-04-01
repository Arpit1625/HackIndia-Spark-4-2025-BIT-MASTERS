import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_from_aadhar(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read the image!")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0] 
        face_image = img[y:y+h, x:x+w]  
        cv2.imwrite(output_path, face_image)
        
        print(f"Face extracted and saved to {output_path}")
        return face_image
    else:
        print("No face detected in the Aadhar card image!")
        return None

image_path = r"C:\Users\shrey\Downloads\amanadhar.jpg"  
output_path = r"extracted_face.jpg"   
extracted_face = extract_face_from_aadhar(image_path, output_path)
if extracted_face is not None:
    cv2.imshow("Extracted Face", extracted_face)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()