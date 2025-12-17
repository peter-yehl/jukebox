import cv2

# Load Haar cascade (included with opencv-python)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detectMultiScale returns (x, y, w, h) and also confidence when scaleFactor >= 1.05
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # We need the version that gives confidence values (OpenCV 4.5+)
    # Try this API:
    faces2 = face_cascade.detectMultiScale3(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        outputRejectLevels=True
    )

    if faces2[0] is not None:
        boxes, rejectLevels, levelWeights = faces2

        for (box, weight) in zip(boxes, levelWeights):
            # Upgrade Confidence from 10 to 100
            confidence = float(weight) * 10

            # If Confidence Above 10%, Create rectangle around face
            if confidence > 10:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f"{confidence:.1f}%",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2)

    cv2.imshow("Face Detection (press q to quit)", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
