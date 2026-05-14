"""
交通标志检测实验 - 完整训练+推理脚本
适用于百度 AI Studio / Kaggle GPU 环境

使用方法：
1. 将整个项目数据集上传到平台的工作目录
2. 运行本脚本即可完成训练、评估、推理和提交文件生成
"""

import os
import csv
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def check_gpu():
    print("=" * 60)
    print("GPU 环境检查")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到 GPU，训练将非常缓慢！")
    print("=" * 60)


def train_model(data_yaml, model_name="yolov8s.pt", epochs=80, imgsz=416, batch=16):
    print("\n" + "=" * 60)
    print("开始模型训练")
    print("=" * 60)

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=4,
        project="runs/detect",
        name="traffic_sign",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
        crop_fraction=1.0,
        patience=20,
        save=True,
        save_period=-1,
        cache=False,
        amp=True,
        fraction=1.0,
        close_mosaic=10,
        resume=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split="val",
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n训练完成！")
    print(f"模型保存目录: {save_dir}")
    print(f"best.pt 实际路径: {best_pt}")
    print(f"best.pt 是否存在: {best_pt.exists()}")

    return model, results, str(best_pt)


def validate_model(model, data_yaml):
    print("\n" + "=" * 60)
    print("模型验证")
    print("=" * 60)

    metrics = model.val(data=data_yaml, split="val", device=0)

    print(f"\nmAP@0.5:    {metrics.box.map50:.4f}")
    print(f"mAP@0.5:95: {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

    class_names = [
        "Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100",
        "Speed Limit 110", "Speed Limit 120", "Speed Limit 20", "Speed Limit 30",
        "Speed Limit 40", "Speed Limit 50", "Speed Limit 60", "Speed Limit 70",
        "Speed Limit 80", "Speed Limit 90", "Stop"
    ]

    print("\n各类别 mAP@0.5:")
    for i, name in enumerate(class_names):
        if i < len(metrics.box.maps):
            print(f"  {name}: {metrics.box.maps[i]:.4f}")

    return metrics


def generate_submission(model, test_dir, output_path="submission.csv", conf=0.001):
    print("\n" + "=" * 60)
    print("生成提交文件")
    print("=" * 60)

    image_paths = sorted(
        [p for p in Path(test_dir).iterdir() if p.is_file()]
    )
    print(f"测试图片数量: {len(image_paths)}")
    if len(image_paths) == 0:
        print("错误: 测试目录中没有图片！请检查路径是否正确。")
        return

    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        total_boxes = 0
        results = model.predict(
            source=[str(p) for p in image_paths],
            conf=conf,
            save=False,
            verbose=False,
            device=0,
        )
        for idx, result in enumerate(results):
            image_id = image_paths[idx].name
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow(
                    {
                        "image_id": image_id,
                        "class_id": int(box.cls[0].item()),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": float(box.conf[0].item()),
                    }
                )
                total_boxes += 1

    print(f"提交文件已生成: {output_path}")
    print(f"总检测框数: {total_boxes}")

    if total_boxes == 0:
        print("警告: 提交文件中没有检测框！模型可能未正确训练或推理出错。")
    else:
        with Path(output_path).open("r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"\n提交文件前5行预览:")
        for line in lines[:5]:
            print(f"  {line.strip()}")


def predict_and_save_images(model, test_dir, output_dir="predict_results", num_images=8):
    print("\n" + "=" * 60)
    print("预测并保存可视化结果")
    print("=" * 60)

    image_paths = sorted(
        [p for p in Path(test_dir).iterdir() if p.is_file()]
    )

    os.makedirs(output_dir, exist_ok=True)

    selected = image_paths[:num_images]

    for img_path in selected:
        model.predict(
            source=str(img_path),
            conf=0.25,
            save=True,
            project=output_dir,
            name="vis",
            exist_ok=True,
            device=0,
            verbose=False,
        )

    vis_dir = os.path.join(output_dir, "vis")
    if os.path.exists(vis_dir):
        print(f"预测可视化图片已保存到: {vis_dir}")


def collect_training_artifacts(run_dir):
    print("\n" + "=" * 60)
    print("收集训练产物（用于实验报告）")
    print("=" * 60)

    artifacts_dir = "experiment_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    important_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "P_curve.png",
        "R_curve.png",
        "PR_curve.png",
        "F1_curve.png",
    ]

    for fname in important_files:
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(artifacts_dir, fname)
            shutil.copy2(src, dst)
            print(f"  已复制: {fname}")

    csv_src = os.path.join(run_dir, "results.csv")
    if os.path.exists(csv_src):
        shutil.copy2(csv_src, os.path.join(artifacts_dir, "results.csv"))
        print(f"  已复制: results.csv")

    weights_dir = os.path.join(run_dir, "weights")
    if os.path.exists(weights_dir):
        for w in os.listdir(weights_dir):
            src = os.path.join(weights_dir, w)
            dst = os.path.join(artifacts_dir, w)
            shutil.copy2(src, dst)
            print(f"  已复制: weights/{w}")

    print(f"\n所有训练产物已保存到: {artifacts_dir}/")


def print_experiment_info():
    print("\n" + "=" * 60)
    print("实验信息（用于填写实验报告）")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("YOLO 版本: Ultralytics YOLOv8")
    print("模型: YOLOv8s")
    print("训练轮数: 80")
    print("图像尺寸: 416")
    print("Batch Size: 16")
    print("优化器: SGD")
    print("初始学习率: 0.01")
    print("数据增强: HSV增强、平移、缩放、水平翻转、Mosaic、Random Erasing")


def main():
    DATA_DIR = "/kaggle/input/datasets/avivali/kagglefour/kagglefour"
    DATA_YAML = os.path.join(DATA_DIR, "data.yaml")
    TEST_DIR = os.path.join(DATA_DIR, "test", "images")

    print(f"数据集目录: {DATA_DIR}")
    print(f"data.yaml: {DATA_YAML}")
    print(f"data.yaml 是否存在: {os.path.exists(DATA_YAML)}")
    print(f"测试图片目录: {TEST_DIR}")
    print(f"测试图片目录是否存在: {os.path.exists(TEST_DIR)}")

    if not os.path.exists(DATA_YAML):
        print(f"错误: 找不到数据配置文件 {DATA_YAML}")
        print("请确保数据集已上传到平台工作目录")
        return

    if not os.path.exists(TEST_DIR):
        print(f"错误: 找不到测试图片目录 {TEST_DIR}")
        return

    check_gpu()
    print_experiment_info()

    model, train_results, best_model_path = train_model(
        data_yaml=DATA_YAML,
        model_name="yolov8s.pt",
        epochs=80,
        imgsz=416,
        batch=16,
    )

    print(f"\n将使用训练好的模型直接进行验证和推理（避免路径错误）")

    metrics = validate_model(model, DATA_YAML)

    generate_submission(
        model=model,
        test_dir=TEST_DIR,
        output_path="submission.csv",
        conf=0.001,
    )

    predict_and_save_images(
        model=model,
        test_dir=TEST_DIR,
        output_dir="predict_results",
        num_images=8,
    )

    collect_training_artifacts(run_dir=str(Path(train_results.save_dir)))

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print("生成的文件：")
    print("  1. submission.csv          - 提交文件")
    print("  2. predict_results/vis/    - 预测可视化图片")
    print("  3. experiment_artifacts/   - 训练产物（损失曲线、混淆矩阵等）")
    print(f"  4. {train_results.save_dir} - 完整训练输出目录")
    print(f"  5. best.pt 路径: {best_model_path}")


if __name__ == "__main__":
    main()