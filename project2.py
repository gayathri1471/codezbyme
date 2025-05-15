import cv2
import torch
import webbrowser

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Object Info Mapping (Example: Book → Wikipedia)
object_info = {
    "bottle": "https://en.wikipedia.org/wiki/Bottle",
    "cell phone": "https://en.wikipedia.org/wiki/Mobile_phone",
    "book": "https://en.wikipedia.org/wiki/Book"
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show AR text overlay
        if label in object_info:
            cv2.putText(frame, "Press 'O' for Info", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("AR Smart Object Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o') and label in object_info:
        webbrowser.open(object_info[label])

cap.release()
cv2.destroyAllWindows()
