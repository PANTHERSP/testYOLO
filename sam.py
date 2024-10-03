import cv2
import os
from segment_anything import SamPredictor, sam_model_registry

# โหลดโมเดล SAM
sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\supha\Documents\65011048\sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# โหลดวิดีโอ
video = cv2.VideoCapture(r"C:\Users\supha\Downloads\video_526807873905230339-rVKi7taA.mov")

# สร้างโฟลเดอร์เพื่อเก็บภาพ
output_folder = r"C:\Users\supha\Documents\65011048\result"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_number = 0

# รัน SAM สำหรับแต่ละเฟรม
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # ทำ segmentation บนเฟรมปัจจุบัน
    predictor.set_image(frame)
    masks, scores, _ = predictor.predict()
    
    # นำ mask มาซ้อนทับกับเฟรมต้นฉบับเพื่อแสดงผล
    for i, mask in enumerate(masks):
        frame[mask] = [0, 255, 0]  # เปลี่ยนสีส่วนที่เป็น mask เป็นสีเขียว
    
    # บันทึกภาพเฟรมที่ถูก segment ลงในโฟลเดอร์
    output_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
    cv2.imwrite(output_path, frame)
    
    print(f"บันทึกภาพเฟรม {frame_number} ลงใน {output_path}")
    frame_number += 1

# ปิดวิดีโอ
video.release()
