from ultralytics import YOLO
import os
import argparse
import torch

# Dọn bộ nhớ GPU trước khi train
torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/data.yaml', help='path to data.yaml')
    parser.add_argument('--model', default='yolov8n.pt', help='base model name or path')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs') # Tăng lên 100
    parser.add_argument('--device', default='0', help="'0' for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    print("🔥 Starting YOLOv8 Fire Detection Training (Optimized)...")

    data_yaml = os.path.abspath(args.data)
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        exit(1)

    print(f"📦 Loading base model: {args.model}")
    model = YOLO(args.model)

    print("🚀 Training...")

    # ✅ Train model - Cấu hình tối ưu cho lửa nhỏ
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        
        # --- CẤU HÌNH QUAN TRỌNG ---
        imgsz=640,              # 🔹 QUAN TRỌNG NHẤT: Tăng độ phân giải để nhìn thấy lửa nhỏ
        batch=20,               # 🔹 Tăng lên 20 (hạ xuống 8 nếu báo lỗi Out of Memory)
        device=args.device,
        name='fire_detect_optimized',
        project='runs/detect',
        exist_ok=True,
        save=True,
        workers=4,              # Tăng worker để load dữ liệu nhanh hơn
        patience=30,            # Dừng sớm nếu 30 epoch không tiến bộ
        
        # --- AUGMENTATION (TỰ ĐỘNG HÓA TỐT HƠN) ---
        # Ta bỏ các thông số lr0, momentum thủ công để YOLO dùng mặc định "smart"
        # Chỉ chỉnh nhẹ các thông số augment để tránh méo hình lửa quá mức
        degrees=10.0,           # Xoay nhẹ ảnh +/- 10 độ
        fliplr=0.5,             # Lật ảnh trái phải (hợp lý với lửa)
        mosaic=1.0,             # Giữ mosaic bật để model học ngữ cảnh tốt
        close_mosaic=10,        # Tắt mosaic 10 epoch cuối để tinh chỉnh chính xác
        
        # rect=True,            # 💡 Mẹo: Bỏ comment dòng này nếu ảnh dataset của bạn là hình chữ nhật (không vuông)
        verbose=True
    )

    print("\n📊 Evaluating model...")
    try:
        metrics = model.val()
        print(f"✅ Training completed!")
        print(f"mAP50: {metrics.box.map50:.3f}")
    except Exception as e:
        print(f"⚠️ Validation warning: {e}")

    # ✅ Xuất model sang ONNX (tốt cho deploy nhúng)
    try:
        print("📦 Exporting to ONNX...")
        model.export(format='onnx', simplify=True)
    except Exception as e:
        print(f"⚠️ ONNX export failed: {e}")

    print(f"\n🎉 Done! Best model: runs/detect/fire_detect_optimized/weights/best.pt")

if __name__ == "__main__":
    main()