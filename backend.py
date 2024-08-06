from ultralytics import YOLO
import socket
import pickle
import struct

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8080))
server_socket.listen(5)

while True:
    client_socket, addr = server_socket.accept()
    data = b""
    payload_size = struct.calcsize("Q")
    
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet:
                break
            data += packet
        if not data:
            break
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            packet = client_socket.recv(4*1024)
            if not packet:
                break
            data += packet
        if not data:
            break
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize frame
        frame = pickle.loads(frame_data)

        # Perform object detection
        results = model(frame, stream=True)
        bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = round(float(box.conf[0]) * 100, 2)  # Convert tensor to float
                cls = int(box.cls[0])
                bboxes.append((x1, y1, x2, y2, classNames[cls], confidence))

        # Serialize bounding boxes
        bbox_data = pickle.dumps(bboxes)
        bbox_size = struct.pack("Q", len(bbox_data))
        client_socket.sendall(bbox_size + bbox_data)

    client_socket.close()