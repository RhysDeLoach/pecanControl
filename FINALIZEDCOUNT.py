####################################################################################
# Load Libraries

from ultralytics import YOLO
import cvzone
import cv2
import threading
import queue
import paho.mqtt.client as mqtt
import time
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
import pandas as pd

####################################################################################
# Initialize Variables

# Load YOLOv11 model
device = 'cuda'
model = YOLO("yolo11n-pecan.pt")  # Update with correct model path
model.to("cuda")

cy1 = 475 # Count Threshold (Center Y-coordinate)
offset = 60 # Offset for Center Y-coordinate
ids = set()
pecanCount = 0
samplePeriod = 0.25 # seconds
refThroughput = 0 # Count / Second

measuredList = []   # Temp
filteredList = []   # Temp
sampleList = []    # Temp
sampleCount = 0     # Temp

####################################################################################
# Initialize Camera

cap = cv2.VideoCapture(0)  # Ensure that you have the correct camera index

# Set the resolution and other properties for each camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 120)

####################################################################################
# Unscented Kalman Filter

# State transition function (identity - no control input)
def fx(x, dt):
    """ State transition function (no external control) """
    return x  # No change in state without an input

# Measurement function (identity)
def hx(x):
    """ Measurement function (direct observation) """
    return x  # We measure the state directly

points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2, kappa=0)# Define sigma points

ukf = UKF(dim_x=1, dim_z=1, fx=fx, hx=hx, points=points, dt=samplePeriod) # Initial State Estimate
ukf.x = np.array([refThroughput]) # Initial state estimate
ukf.Q = np.array([[0.02]]) # Process noise covariance (Q) - controls how much the state changes naturally
ukf.R = np.array([[1]]) # Measurement noise covariance (R) - how noisy the measurements are
ukf.P = np.eye(1) * 0.1 # Initial state covariance (P) - initial uncertainty

####################################################################################
# MQTT

# Mqtt Configuration
MQTT_BROKER = "192.168.1.110"
MQTT_PORT = 1883
MQTT_TOPIC = "/jc/feedrate/"
MQTT_COUNT_TOPIC = "/jc/feedrate/count/"

# MQTT Handling
def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, message):
    global refThroughput  # Count / Second
    refThroughput = float(message.payload.decode())
    with refThroughput_lock:
        with ukf_lock:
            ukf.x = np.array([refThroughput])

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()  # Starts the loop in the background

####################################################################################
# Thread Initialization

# Queue to hold frames captured by the capture thread
frame_queue = queue.Queue(maxsize=1)  # Limit to 1 frames in the queue

# Locks for thread synchronization
refThroughput_lock = threading.Lock()
ukf_lock = threading.Lock()

####################################################################################
# Camera Thread

def capture_thread():

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the black box position (adjust as needed)
        top_left = (0, 600)
        bottom_right = (1280, 720)

        # Draw a black rectangle (filled)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), thickness=-1)

        # Flip Frame
        frame = cv2.flip(frame, 0)

        # Put the frame into the queue
        if not frame_queue.full():
            frame_queue.put(frame)

####################################################################################
# Processing Thread

def processing_thread():
    global pecanCount, ids, sampleCount, measuredList, filteredList, sampleList

    sampleStart = time.time()

    while True:
        # Check if there are any frames in the queue
        if not frame_queue.empty():
            frame = frame_queue.get()

            results = model.track(frame, persist=True, classes=0, device=device, stream=True, tracker='bytetrack_custom.yaml', iou = 0.6)

            for result in results:
                # Check if there are any boxes in the result
                if result.boxes is not None and result.boxes.id is not None:
                    # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
                    boxes = result.boxes.xyxy.int().tolist()  # Bounding boxes
                    track_ids = result.boxes.id.int().tolist()  # Track IDs
                    confidence = result.boxes.conf.tolist()

                    # Iterate over each detected object
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        cy = int((y1 + y2) // 2)  # Center Y-coordinate

                        if cy < (cy1 + offset) and cy > (cy1 - offset) and track_id not in ids:
                            pecanCount += 1
                            ids.add(track_id)
    
        sampleEnd = time.time()
        if (sampleEnd - sampleStart) > samplePeriod:
            measuredCount = pecanCount / samplePeriod

            with ukf_lock:
                ukf.predict()
                ukf.update(np.array([measuredCount]))

            filteredCount = ukf.x[0]

            mqtt_client.publish(MQTT_COUNT_TOPIC, str(filteredCount))

            sampleCount += 0.25 # Temp
            sampleList.append(sampleCount)  # Temp
            measuredList.append(measuredCount) # Temp
            filteredList.append(filteredCount)    # Temp

            pecanCount = 0 # Reset count for next sample period
            sampleStart = time.time() # Reset start time for next sample period

####################################################################################
# Threading

# Start capture and processing threads
capture_thread = threading.Thread(target=capture_thread)
capture_thread.daemon = True  # Ensures the thread exits when the main program exits
capture_thread.start()

processing_thread = threading.Thread(target=processing_thread)
processing_thread.daemon = True  # Ensures the thread exits when the main program exits
processing_thread.start()

####################################################################################
# Clean Up

try:
    while True:
        # Keep the main loop alive
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
except KeyboardInterrupt:
    print("Process interrupted")

# Release resources when exiting
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
print("Capture Released")

data = pd.DataFrame(list(zip(sampleList, measuredList, filteredList)), columns=['Time', 'Measured', 'Filtered'])
data.to_csv('throughput80IC34.csv', index=False)

####################################################################################


