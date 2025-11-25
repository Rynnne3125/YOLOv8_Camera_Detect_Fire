import cv2
from ultralytics import YOLO
import math

# --- CẤU HÌNH ---
# Đường dẫn model (Nếu chưa train xong thì dùng tạm yolov8n.pt để test code)
MODEL_PATH = 'runs/detect/fire_detection_fast/weights/best.pt' 

# --- MAIN ---
def main():
    # 1. Load Model
    print(f"🔥 Đang tải model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except:
        print("⚠️ Lỗi load model custom, đang dùng model mặc định để test code...")
        model = YOLO("yolov8n.pt")

    # In ra tên các class mà model này học được để debug
    print("📋 Model Classes:", model.names)

    # 2. Mở Camera (0 là webcam laptop, 1 là webcam rời nếu có)
    cap = cv2.VideoCapture(0)
    
    # Set độ phân giải (thấp một chút để FPS cao)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Không mở được Webcam!")
        return

    print("\n🚀 ĐANG CHẠY! Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. NHẬN DIỆN (Quan trọng)
        # conf=0.25: Chỉ cần chắc chắn 25% là lửa cũng sẽ hiện (giúp bắt lửa nhỏ)
        # iou=0.5: Giúp loại bỏ các khung trùng nhau
        results = model(frame, stream=True, conf=0.25, iou=0.5, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy thông tin bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Lấy độ tự tin (Confidence)
                conf = math.ceil((box.conf[0] * 100)) / 100
                
                # Lấy tên class
                cls = int(box.cls[0])
                current_class = model.names[cls]

                # --- LỌC CLASS (Tùy chọn) ---
                # Nếu model bạn train chỉ có 1 class là lửa, thì không cần if này.
                # Nếu dùng model gốc (có người, xe, v.v) thì cần lọc.
                # if current_class not in ['fire', 'flame']: continue 

                # Vẽ khung chữ nhật
                if conf > 0.4: # Màu đỏ đậm nếu chắc chắn
                    color = (0, 0, 255) 
                else: # Màu cam nếu chưa chắc chắn lắm (lửa nhỏ/mờ)
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Viết chữ lên trên
                label = f'{current_class} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)  # Nền chữ
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Hiển thị
        cv2.imshow("Fire Detection Test", frame)

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()