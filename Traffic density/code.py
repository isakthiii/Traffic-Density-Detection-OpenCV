import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 model (Nano version for fast real-time performance)
model = YOLO('yolov8n.pt') 

# Define COCO dataset class IDs for target vehicles
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# 2. Setup Video Capture (0 for default webcam, or put path to a video file like 'traffic.mp4')
video_source = 0 
cap = cv2.VideoCapture(video_source)

# Check if camera/video opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("Processing traffic video feed... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video stream ended or failed to read frame.")
        break
        
    # 3. Run YOLO inference on the current frame
    # verbose=False keeps the VS Code terminal clean
    results = model(frame, verbose=False)
    
    vehicle_count = 0
    
    # 4. Parse the detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract class ID
            class_id = int(box.cls[0])
            
            # Check if detected object belongs to our vehicle classes
            if class_id in VEHICLE_CLASSES:
                vehicle_count += 1
                
                # Get coordinates for bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                confidence = float(box.conf[0])
                label = f"{model.names[class_id]} {confidence:.2f}"
                
                # Draw the bounding box around the vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Put the vehicle type text label above the box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 2)

    # 5. Determine Traffic Density Status based on presentation thresholds
    if vehicle_count <= 10:
        status = "LOW"
        status_color = (0, 255, 0)     # Green
    elif 11 <= vehicle_count <= 20:
        status = "MEDIUM"
        status_color = (0, 255, 255)   # Yellow
    else:
        status = "HIGH"
        status_color = (0, 0, 255)     # Red

    # 6. Display Analytics Overlay Dashboard on the window
    # Semi-transparent background block for metrics dashboard
    cv2.rectangle(frame, (20, 20), (320, 120), (0, 0, 0), -1)
    
    # Render texts for Total Counts and Density Status
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (35, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Density: {status}", (35, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # 7. Render and output window frame
    cv2.imshow("Intelligent Traffic Density Detector Dashboard", frame)
    
    # Break loop cleanly when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
