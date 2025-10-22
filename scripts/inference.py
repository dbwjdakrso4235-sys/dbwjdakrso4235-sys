"""
YOLOv11 Segmentation Inference Script
학습된 모델로 추론 실행 및 결과 시각화
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def run_inference(
    model_path,
    source,
    output_dir="runs/predict",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,
    save_conf=True,
    show_labels=True,
    show_conf=True,
    device=0
):
    """
    모델 추론 실행

    Args:
        model_path (str): 모델 가중치 경로
        source (str): 입력 소스 (이미지/폴더/비디오)
        output_dir (str): 결과 저장 디렉토리
        imgsz (int): 이미지 크기
        conf (float): Confidence threshold
        iou (float): IoU threshold (NMS)
        save (bool): 결과 이미지 저장
        save_txt (bool): 라벨 저장
        save_conf (bool): Confidence 저장
        show_labels (bool): 라벨 표시
        show_conf (bool): Confidence 표시
        device (int/str): Device
    """
    print("=" * 80)
    print("YOLOv11 Segmentation Inference")
    print("=" * 80)
    print(f"  Model: {model_path}")
    print(f"  Source: {source}")
    print(f"  Confidence threshold: {conf}")
    print(f"  IoU threshold: {iou}")
    print("=" * 80 + "\n")

    # 모델 로드
    model = YOLO(model_path)

    # 추론 실행
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        show_labels=show_labels,
        show_conf=show_conf,
        project=output_dir,
        device=device,
        verbose=True
    )

    print(f"\n추론 완료! 결과 저장 위치: {output_dir}")

    return results


def visualize_results(results, save_dir=None, show=True):
    """
    추론 결과 시각화

    Args:
        results: YOLO 추론 결과 객체
        save_dir (str): 저장 디렉토리
        show (bool): 결과 표시
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, r in enumerate(results):
        # 원본 이미지
        img = r.orig_img.copy()

        # Boxes와 Masks
        boxes = r.boxes
        masks = r.masks

        # 결과 플롯
        plotted = r.plot()

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 원본 이미지
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 예측 결과
        axes[1].imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Prediction (Objects: {len(boxes)})', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()

        if save_dir:
            save_path = save_dir / f"result_{idx:04d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def analyze_predictions(results, class_names=None):
    """
    예측 결과 분석

    Args:
        results: YOLO 추론 결과
        class_names (list): 클래스 이름 리스트
    """
    print("\n" + "=" * 80)
    print("Prediction Analysis")
    print("=" * 80)

    total_objects = 0
    total_images = len(results)
    conf_scores = []

    for idx, r in enumerate(results):
        boxes = r.boxes
        num_objects = len(boxes)
        total_objects += num_objects

        if num_objects > 0:
            confs = boxes.conf.cpu().numpy()
            conf_scores.extend(confs)

            print(f"\n[Image {idx + 1}/{total_images}]")
            print(f"  Objects detected: {num_objects}")
            print(f"  Confidence: min={confs.min():.3f}, max={confs.max():.3f}, avg={confs.mean():.3f}")

            # 클래스별 통계
            classes = boxes.cls.cpu().numpy()
            unique_classes, counts = np.unique(classes, return_counts=True)

            for cls, count in zip(unique_classes, counts):
                class_name = class_names[int(cls)] if class_names else f"Class {int(cls)}"
                print(f"    {class_name}: {count}")

    # 전체 통계
    print(f"\n[Overall Statistics]")
    print(f"  Total images: {total_images}")
    print(f"  Total objects: {total_objects}")
    print(f"  Avg objects per image: {total_objects / total_images:.2f}")

    if conf_scores:
        conf_scores = np.array(conf_scores)
        print(f"  Confidence: min={conf_scores.min():.3f}, max={conf_scores.max():.3f}, avg={conf_scores.mean():.3f}")

    # Confidence 분포 시각화
    if conf_scores:
        plt.figure(figsize=(10, 5))
        plt.hist(conf_scores, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('confidence_distribution.png', dpi=150)
        print(f"\n  Confidence 분포 그래프 저장: confidence_distribution.png")
        plt.show()


def batch_inference_with_stats(model_path, image_dir, output_dir, **kwargs):
    """
    배치 추론 및 통계 생성

    Args:
        model_path (str): 모델 경로
        image_dir (str): 이미지 디렉토리
        output_dir (str): 출력 디렉토리
        **kwargs: run_inference에 전달할 추가 인자
    """
    # 추론 실행
    results = run_inference(
        model_path=model_path,
        source=image_dir,
        output_dir=output_dir,
        **kwargs
    )

    # 클래스 이름 가져오기
    model = YOLO(model_path)
    class_names = model.names

    # 결과 분석
    analyze_predictions(results, class_names)

    # 시각화 (처음 5개만)
    print(f"\n처음 5개 결과 시각화 중...")
    visualize_results(
        results[:5],
        save_dir=Path(output_dir) / "visualizations",
        show=False
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv11 Segmentation Inference')

    # 필수 인자
    parser.add_argument('--model', type=str, required=True,
                       help='모델 가중치 경로 (예: runs/segment/train/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='입력 소스 (이미지 파일/폴더/비디오)')

    # 추론 설정
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold (NMS)')
    parser.add_argument('--device', default='0', help='Device (0, 1, ...) or cpu')

    # 출력 설정
    parser.add_argument('--output', type=str, default='runs/predict', help='결과 저장 디렉토리')
    parser.add_argument('--save', action='store_true', default=True, help='결과 이미지 저장')
    parser.add_argument('--save-txt', action='store_true', default=True, help='라벨 저장')
    parser.add_argument('--save-conf', action='store_true', default=True, help='Confidence 저장')

    # 시각화 설정
    parser.add_argument('--show-labels', action='store_true', default=True, help='라벨 표시')
    parser.add_argument('--show-conf', action='store_true', default=True, help='Confidence 표시')
    parser.add_argument('--analyze', action='store_true', help='결과 분석 수행')

    args = parser.parse_args()

    if args.analyze or Path(args.source).is_dir():
        # 배치 추론 + 분석
        batch_inference_with_stats(
            model_path=args.model,
            image_dir=args.source,
            output_dir=args.output,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            show_labels=args.show_labels,
            show_conf=args.show_conf
        )
    else:
        # 단일 이미지 추론
        results = run_inference(
            model_path=args.model,
            source=args.source,
            output_dir=args.output,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            show_labels=args.show_labels,
            show_conf=args.show_conf
        )

        # 결과 표시
        visualize_results(results, show=True)
