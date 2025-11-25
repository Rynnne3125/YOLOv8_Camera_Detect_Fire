import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

# --- CẤU HÌNH MẶC ĐỊNH (BẠN CHỈNH SỬA Ở ĐÂY) ---
# Thay 'best.pt' bằng đường dẫn file model train dataset của bạn
# Ví dụ: 'runs/detect/train/weights/best.pt' hoặc 'yolov8n.pt'
DEFAULT_MODEL_PATH = 'best.pt' 

# Mặc định là '0' để dùng Webcam laptop/USB. 
# Nếu dùng ảnh/video thì điền đường dẫn file vào đây.
DEFAULT_SOURCE = '0' 

# --------------------------------------------------

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                    help='Path to YOLO model file (.pt)')
parser.add_argument('--source', type=str, default=DEFAULT_SOURCE,
                    help='0 for webcam, or path to image/video')
parser.add_argument('--thresh', type=float, default=0.5,
                    help='Minimum confidence threshold')
parser.add_argument('--resolution', default=None,
                    help='Resolution in WxH (example: 640x480)')
parser.add_argument('--record', action='store_true',
                    help='Record results to demo.avi')

args = parser.parse_args()

# Gán biến từ arguments
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# --- KIỂM TRA MODEL ---
if not os.path.exists(model_path):
    # Nếu không tìm thấy model custom, thử load model chuẩn yolov8n.pt để test
    print(f"⚠️ Cảnh báo: Không tìm thấy file model tại '{model_path}'.")
    print("👉 Đang thử tải model mặc định 'yolov8n.pt' để chạy thử...")
    model_path = 'yolov8n.pt'

# --- LOAD YOLO MODEL ---
print(f"🔥 Đang load model: {model_path}...")
try:
    model = YOLO(model_path, task='detect')
    # Tự động lấy danh sách class từ dataset đã train
    labels = model.names 
    print(f"✅ Đã load thành công! Dataset gồm {len(labels)} classes: {labels}")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    sys.exit(0)

# --- XÁC ĐỊNH NGUỒN (SOURCE TYPE) ---
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']
source_type = None

# Xử lý input
if img_source == '0':
    source_type = 'usb'
    usb_idx = 0
elif img_source.isdigit(): # Nếu là số khác (1, 2...)
    source_type = 'usb'
    usb_idx = int(img_source)
elif os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
elif 'picamera' in img_source:
    source_type = 'picamera'
else:
    # Mặc định thử coi như là webcam nếu không tìm thấy file
    print(f"⚠️ Không tìm thấy file '{img_source}', thử mở Webcam 0...")
    source_type = 'usb'
    usb_idx = 0

# --- CẤU HÌNH ĐỘ PHÂN GIẢI ---
resW, resH = 640, 480 # Default
resize = False
if user_res:
    try:
        parts = user_res.split('x')
        resW, resH = int(parts[0]), int(parts[1])
        resize = True
    except:
        print("Lỗi format resolution. Dùng mặc định 640x480.")

# --- KHỞI TẠO CAMERA / SOURCE ---
cap = None
imgs_list = []
img_count = 0

if source_type == 'usb':
    print(f"📷 Đang mở Webcam index {usb_idx}...")
    cap = cv2.VideoCapture(usb_idx)
    # Cố gắng set độ phân giải phần cứng để nét hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)

elif source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = []
    for ext in img_ext_list:
        imgs_list.extend(glob.glob(os.path.join(img_source, f'*{ext}')))
    print(f"📁 Tìm thấy {len(imgs_list)} ảnh trong folder.")

elif source_type == 'picamera':
    try:
        from picamera2 import Picamera2
        cap = Picamera2()
        config = cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)})
        cap.configure(config)
        cap.start()
        print("Picamera2 started.")
    except ImportError:
        print("❌ Lỗi: Không có thư viện 'picamera2'. (Chỉ chạy trên Raspberry Pi)")
        print("👉 Đang chuyển sang chế độ Webcam USB...")
        source_type = 'usb'
        usb_idx = 0
        cap = cv2.VideoCapture(0)

# --- SETUP GHI HÌNH (RECORD) ---
recorder = None
if record and source_type in ['usb', 'video', 'picamera']:
    record_name = 'demo_output.avi'
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))
    print(f"🔴 Đang ghi hình vào: {record_name}")

# --- BẢNG MÀU (Tableau 10) ---
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# --- VÒNG LẶP CHÍNH (INFERENCE LOOP) ---
print("\n🚀 Bắt đầu nhận diện... Nhấn 'q' để thoát, 'p' để tạm dừng.\n")

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

while True:
    t_start = time.perf_counter()

    # 1. ĐỌC FRAME
    frame = None
    
    if source_type in ['usb', 'video']:
        ret, frame = cap.read()
        if not ret:
            print("Kết thúc video hoặc mất kết nối camera.")
            break
            
    elif source_type == 'picamera':
        frame = cap.capture_array()
        
    elif source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print("Đã xử lý hết ảnh.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
        if frame is None: continue

    if frame is None: break

    # 2. RESIZE (Nếu cần thiết để hiển thị chuẩn)
    if resize or source_type in ['image', 'folder']:
        frame = cv2.resize(frame, (resW, resH))

    # 3. AI NHẬN DIỆN (YOLO)
    # verbose=False để đỡ spam terminal
    results = model(frame, verbose=False, conf=min_thresh)
    detections = results[0].boxes

    # 4. VẼ KHUNG VÀ THÔNG TIN
    object_count = 0
    
    for i in range(len(detections)):
        # Lấy tọa độ
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Lấy Class ID và Tên Class
        classidx = int(detections[i].cls.item())
        if classidx < len(labels):
            classname = labels[classidx]
        else:
            classname = f"Class {classidx}"

        conf = detections[i].conf.item()

        # Chỉ vẽ nếu độ tin cậy > ngưỡng
        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            
            # Vẽ hình chữ nhật
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Tạo nhãn (Label)
            label = f'{classname} {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            
            # Vẽ nền chữ
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), 
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            # Viết chữ
            cv2.putText(frame, label, (xmin, label_ymin - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            object_count += 1

    # 5. HIỂN THỊ FPS
    t_stop = time.perf_counter()
    fps_curr = 1 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0
    
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps_curr)
    avg_frame_rate = np.mean(frame_rate_buffer)

    cv2.putText(frame, f'FPS: {avg_frame_rate:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 6. SHOW HÌNH ẢNH
    cv2.imshow('YOLO Custom Dataset', frame)
    
    if recorder:
        recorder.write(frame)

    # 7. PHÍM ĐIỀU KHIỂN
    wait_ms = 1
    if source_type in ['image', 'folder']: wait_ms = 0 # Dừng lại ở ảnh để xem
    
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q'): # Thoát
        break
    elif key == ord('p'): # Pause
        cv2.waitKey(0)

# --- CLEAN UP ---
if cap:
    if hasattr(cap, 'release'): cap.release()
    elif hasattr(cap, 'stop'): cap.stop()
if recorder:
    recorder.release()
cv2.destroyAllWindows()
print("👋 Chương trình đã tắt.")