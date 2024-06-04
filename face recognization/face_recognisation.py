import cv2
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images for face recognition
image_paths_sreemanta = [
   
     # Add system paths to  images for person 2
]

image_paths_person2 = [
 
   
    # Add system paths to  images for person 2
]

image_paths_person3 = [
    #r'C:\path\to\your\images\person3_img1.jpg',
    #r'C:\path\to\your\images\person3_img2.jpg',
    # Add more images for person 3
]

# Combine all image paths and create labels
image_paths = image_paths_sreemanta + image_paths_person2 + image_paths_person3
labels = [1] * len(image_paths_sreemanta) + [2] * len(image_paths_person2) + [3] * len(image_paths_person3)

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer with multiple images and labels
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
recognizer.train(images, np.array(labels))

# Capture video from the default camera (you can also load a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Horizontally flip the frame
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) from the face
        face_roi = gray[y:y+h, x:x+w]

        # Perform face recognition
        label, confidence = recognizer.predict(face_roi)

        # Display the face rectangle and recognition result
        if confidence < 100:
            # Display correct face ID in green
            color = (0, 255, 0)
            if label == 1:
                cv2.putText(frame, "Sreemanta", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            elif label == 2:
                cv2.putText(frame, "Susumita ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            elif label == 3:
                cv2.putText(frame, "Person 3", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            # Display "Unknown" in red for unrecognized faces
            color = (0, 0, 255)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
