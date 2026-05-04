

import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

camera = cv2.VideoCapture(0)

initial_speed = None

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to open the camera...")
        break

    # pose estimation
    results = model(frame, verbose=False)

    if initial_speed is None:
        speed = results[0].speed
        initial_speed = (
            speed["preprocess"], 
            speed["inference"],
            speed["postprocess"]
        )


    output_frame = results[0].plot()

    cv2.putText(output_frame, f"Speed: {initial_speed[0]:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Pose Detection", output_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
