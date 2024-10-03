# import cv2
# import os
# from segment_anything import SamPredictor, sam_model_registry

# # โหลดโมเดล SAM
# sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\supha\Documents\65011048\sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)

# # โหลดวิดีโอ
# image = cv2.imread("https://t1.blockdit.com/photos/2021/12/61c9be00112c9421e4ff0ca0_800x0xcover_TcXIogxn.jpg")

# # สร้างโฟลเดอร์เพื่อเก็บภาพ
# output_folder = r"C:\Users\supha\Documents\65011048\result"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)


# # รัน SAM สำหรับแต่ละเฟรม
# for i in range(image.shape[0]):
#     # ทำ segmentation บนเฟรมปัจจุบัน
#     predictor.set_image(image[i])
#     masks, scores, _ = predictor.predict()
    
#     # นำ mask มาซ้อนทับกับเฟรมต้นฉบับเพื่อแสดงผล
#     for j, mask in enumerate(masks):
#         image[i][mask] = [0, 255, 0]  # เปลี่ยนสีส่วนที่เป็น mask เป็นสีเขียว
    
#     # บันทึกภาพเฟรมที่ถูก segment ลงในโฟลเดอร์
#     output_path = os.path.join(output_folder, f"frame_{i}.jpg")
#     cv2.imwrite(output_path, image[i])

#     print(f"บันทึกภาพเฟรม {i} ลงใน {output_path}")



import cv2
import os
from segment_anything import SamPredictor, sam_model_registry

# โหลดโมเดล SAM
print("เริ่มต้นการโหลดโมเดล SAM...")
try:
    sam = sam_model_registry["vit_h"](checkpoint=r"C:\Users\supha\Documents\65011048\sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    print("โมเดล SAM โหลดสำเร็จ")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    exit()

# โหลดภาพนิ่ง
image_path = "https://t1.blockdit.com/photos/2021/12/61c9be00112c9421e4ff0ca0_800x0xcover_TcXIogxn.jpg"  # ระบุเส้นทางของภาพนิ่ง
print(f"กำลังโหลดภาพจากเส้นทาง: {image_path}")
image = cv2.imread(image_path)

if image is None:
    print(f"ไม่สามารถโหลดภาพได้จากเส้นทาง: {image_path}")
    exit()
else:
    print(f"โหลดภาพสำเร็จ: {image_path}")

# สร้างโฟลเดอร์เพื่อเก็บผลลัพธ์
output_folder = r"C:\Users\supha\Documents\65011048\result"
print(f"กำลังตรวจสอบโฟลเดอร์สำหรับเก็บผลลัพธ์: {output_folder}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"โฟลเดอร์ {output_folder} ถูกสร้างเรียบร้อย")
else:
    print(f"โฟลเดอร์ {output_folder} มีอยู่แล้ว")

# ทำ segmentation บนภาพนิ่ง
print("กำลังทำการ segmentation บนภาพ...")
try:
    predictor.set_image(image)
    masks, scores, _ = predictor.predict()
    print(f"การ segmentation สำเร็จ: พบ {len(masks)} mask(s) ในภาพ")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการทำ segmentation: {e}")
    exit()

# นำ mask มาซ้อนทับกับภาพต้นฉบับเพื่อแสดงผล
print("กำลังนำ mask มาซ้อนทับกับภาพต้นฉบับ...")
for i, mask in enumerate(masks):
    image[mask] = [0, 255, 0]  # เปลี่ยนสีส่วนที่เป็น mask เป็นสีเขียว
print(f"ซ้อนทับ mask ลงบนภาพสำเร็จ")

# บันทึกภาพที่ถูก segment ลงในโฟลเดอร์
output_path = os.path.join(output_folder, "segmented_image.jpg")
print(f"กำลังบันทึกภาพที่ถูก segment ลงใน: {output_path}")
cv2.imwrite(output_path, image)
print(f"บันทึกภาพสำเร็จที่: {output_path}")
