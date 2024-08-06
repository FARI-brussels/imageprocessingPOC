import cv2
import socket
import pickle
import struct
import argparse
import os
import time

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Webcam streaming client.")
parser.add_argument('--record', type=bool, default=False, help='Record one frame per hour in /home/fari/images')
args = parser.parse_args()

# Ensure the output directory exists if recording is enabled
if args.record:
    output_dir = "/home/fari/images"
    os.makedirs(output_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.10', 8080))  # Replace with the backend's IP address

last_record_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    # Serialize frame
    data = pickle.dumps(img)

    # Send message length first
    message_size = struct.pack("Q", len(data))

    # Then data
    client_socket.sendall(message_size + data)

    # Receive bounding box data from backend
    data = b""
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        packet = client_socket.recv(4 * 1024)
        if not packet:
            break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4 * 1024)
    bbox_data = data[:msg_size]
    bboxes = pickle.loads(bbox_data)

    # Draw bounding boxes on the frame
    for (x1, y1, x2, y2, cls, confidence) in bboxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        org = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(img, cls, org, font, fontScale, color, thickness)

    # Display the frame
    cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Webcam', img)

    # Record one frame per hour if the --record flag is set
    if args.record and (time.time() - last_record_time) >= 3600:
        frame_time = time.strftime("%Y%m%d-%H%M%S")
        frame_path = os.path.join(output_dir, f"frame_{frame_time}.jpg")
        cv2.imwrite(frame_path, img)
        last_record_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client_socket.close()
