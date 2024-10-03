import cv2
from ultralytics import YOLO
import numpy as np
import random
import torch


# เปิดใช้การปรับแต่งของ OpenCV
cv2.setUseOptimized(True)

# โหลดโมเดล YOLOv8 ที่ทำ instance segmentation
model = YOLO('yolov9e-seg.pt')  # หรือ 'cuda:0' สำหรับ GPU ตัวแรก

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

# สร้าง mapping สีสำหรับแต่ละ label
label_colors = {}

# ฟังก์ชันสุ่มสี
def get_label_color(label):
    if label not in label_colors:
        # สุ่มสีสำหรับ label ที่ยังไม่เคยถูกสร้าง
        label_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return label_colors[label]

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# ตั้งค่าขนาดเฟรม
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ทำการตรวจจับ instance segmentation บน frame จากกล้อง
    results = model(frame, batch=1)
    

    # คัดลอก frame
    annotated_frame = frame.copy()

    # ตรวจสอบว่ามีการตรวจจับวัตถุและ mask หรือไม่
    if results[0].masks is not None:
        # วาด segmentation mask
        for i, mask in enumerate(results[0].masks.data):
            binary_mask = mask.cpu().numpy().astype('uint8') * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ดึงสีสำหรับ label ปัจจุบัน
            label = results[0].names[results[0].boxes.cls[i].item()]
            label_color = get_label_color(label)

            # วาดเฉพาะเส้นรอบวง (contours) ของ mask ไม่เติมสีภายใน
            cv2.drawContours(annotated_frame, contours, -1, label_color, 4)

            # แสดง label และ score พร้อมพื้นหลัง
            score = results[0].boxes.conf[i].item() * 100  # เปลี่ยนเป็น %
            label_text = f'{label} {score:.2f}%'

            # ตำแหน่งของ label และขนาด text
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            x, y = int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1])

            # วาดพื้นหลังสีตามกรอบ
            cv2.rectangle(annotated_frame, (x, y - text_height - baseline), (x + text_width, y), label_color, -1)

            # วาด label ด้วยตัวหนังสือสีขาว
            cv2.putText(annotated_frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # แสดงผล
    cv2.imshow("YOLOv9e Segmentation", annotated_frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
