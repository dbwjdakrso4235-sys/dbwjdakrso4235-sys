"""
YOLOv11 Segmentation Training Script
곤포사일리지 데이터셋으로 YOLOv11-seg 모델 학습
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import torch
from datetime import datetime


def train_yolov11_seg(
    data_yaml="E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml",
    model_size='n',  # n, s, m, l, x
    epochs=100,
    imgsz=640,
    batch=16,
    project='runs/segment',
    name=None,
    device=0,
    resume=False,
    pretrained=True,
    **kwargs
):
    """
    YOLOv11 Segmentation 모델 학습

    Args:
        data_yaml (str): 데이터셋 YAML 파일 경로
        model_size (str): 모델 크기 (n/s/m/l/x)
        epochs (int): 학습 epoch 수
        imgsz (int): 입력 이미지 크기
        batch (int): 배치 크기
        project (str): 프로젝트 디렉토리
        name (str): 실험 이름
        device (int/str): GPU device (0, 1, ... 또는 'cpu')
        resume (bool): 학습 재개 여부
        pretrained (bool): 사전학습 모델 사용 여부
        **kwargs: 추가 학습 파라미터
    """

    print("=" * 80)
    print("YOLOv11 Segmentation Training")
    print("=" * 80)

    # 시스템 정보
    print(f"\n[System Info]")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    # 학습 설정
    print(f"\n[Training Config]")
    print(f"  Model: yolo11{model_size}-seg")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")

    # 실험 이름 자동 생성
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"silage_bale_yolo11{model_size}_{timestamp}"

    print(f"  Experiment name: {name}")

    # 모델 로드
    if resume:
        # 학습 재개
        last_checkpoint = Path(project) / name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            print(f"\n[Resume] 이전 학습 재개: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            print(f"\n[Warning] Checkpoint not found: {last_checkpoint}")
            print("새로운 학습을 시작합니다.")
            resume = False
            model = YOLO(f'yolo11{model_size}-seg.pt' if pretrained else f'yolo11{model_size}-seg.yaml')
    else:
        # 새로운 학습
        if pretrained:
            model_path = f'yolo11{model_size}-seg.pt'
            print(f"\n[Model] 사전학습 모델 로드: {model_path}")
        else:
            model_path = f'yolo11{model_size}-seg.yaml'
            print(f"\n[Model] 아키텍처에서 새로 시작: {model_path}")

        model = YOLO(model_path)

    # 기본 학습 파라미터
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'name': name,
        'project': project,
        'device': device,
        'resume': resume,

        # Optimizer
        'optimizer': kwargs.get('optimizer', 'AdamW'),
        'lr0': kwargs.get('lr0', 0.01),
        'lrf': kwargs.get('lrf', 0.01),
        'momentum': kwargs.get('momentum', 0.937),
        'weight_decay': kwargs.get('weight_decay', 0.0005),

        # Training settings
        'patience': kwargs.get('patience', 50),
        'save': kwargs.get('save', True),
        'save_period': kwargs.get('save_period', 10),
        'workers': kwargs.get('workers', 8),
        'cos_lr': kwargs.get('cos_lr', False),
        'close_mosaic': kwargs.get('close_mosaic', 10),

        # Augmentation
        'hsv_h': kwargs.get('hsv_h', 0.015),
        'hsv_s': kwargs.get('hsv_s', 0.7),
        'hsv_v': kwargs.get('hsv_v', 0.4),
        'degrees': kwargs.get('degrees', 0.0),
        'translate': kwargs.get('translate', 0.1),
        'scale': kwargs.get('scale', 0.5),
        'shear': kwargs.get('shear', 0.0),
        'perspective': kwargs.get('perspective', 0.0),
        'flipud': kwargs.get('flipud', 0.0),
        'fliplr': kwargs.get('fliplr', 0.5),
        'mosaic': kwargs.get('mosaic', 1.0),
        'mixup': kwargs.get('mixup', 0.0),
        'copy_paste': kwargs.get('copy_paste', 0.0),

        # Validation
        'val': kwargs.get('val', True),
        'plots': kwargs.get('plots', True),
        'verbose': kwargs.get('verbose', True),
    }

    print(f"\n[Hyperparameters]")
    print(f"  Optimizer: {train_args['optimizer']}")
    print(f"  Learning rate: {train_args['lr0']} -> {train_args['lrf']}")
    print(f"  Momentum: {train_args['momentum']}")
    print(f"  Weight decay: {train_args['weight_decay']}")
    print(f"  Patience: {train_args['patience']}")

    print(f"\n[Data Augmentation]")
    print(f"  Flip LR: {train_args['fliplr']}")
    print(f"  Translate: {train_args['translate']}")
    print(f"  Scale: {train_args['scale']}")
    print(f"  Mosaic: {train_args['mosaic']}")

    print("\n" + "=" * 80)
    print("학습 시작...")
    print("=" * 80 + "\n")

    # 학습 실행
    try:
        results = model.train(**train_args)

        print("\n" + "=" * 80)
        print("학습 완료!")
        print("=" * 80)

        # 결과 저장 경로
        save_dir = Path(project) / name
        print(f"\n[Results]")
        print(f"  Weights: {save_dir / 'weights'}")
        print(f"  Best model: {save_dir / 'weights' / 'best.pt'}")
        print(f"  Last model: {save_dir / 'weights' / 'last.pt'}")
        print(f"  Plots: {save_dir}")

        return results, model

    except KeyboardInterrupt:
        print("\n\n학습이 사용자에 의해 중단되었습니다.")
        print(f"체크포인트: {Path(project) / name / 'weights' / 'last.pt'}")
        print("다음 명령으로 재개할 수 있습니다:")
        print(f"  python scripts/train.py --resume --name {name}")
        return None, model

    except Exception as e:
        print(f"\n\n에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, model


def validate_model(model_path, data_yaml, imgsz=640, batch=16, device=0):
    """
    모델 검증

    Args:
        model_path (str): 모델 파일 경로
        data_yaml (str): 데이터셋 YAML 경로
        imgsz (int): 이미지 크기
        batch (int): 배치 크기
        device (int/str): Device
    """
    print("\n" + "=" * 80)
    print("Model Validation")
    print("=" * 80)

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        split='val',
        save_json=True,
        plots=True
    )

    print(f"\n[Validation Results]")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")
    print(f"  mAP50: {metrics.seg.map50:.4f}")
    print(f"  mAP75: {metrics.seg.map75:.4f}")
    print(f"  Precision: {metrics.seg.p[0]:.4f}")
    print(f"  Recall: {metrics.seg.r[0]:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLOv11 Segmentation Training')

    # 필수 인자
    parser.add_argument('--data', type=str, default='E:/namwon_ai/dataset_silage_bale_rgb/dataset.yaml',
                       help='dataset.yaml 경로')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='모델 크기')

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100, help='학습 epoch 수')
    parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    parser.add_argument('--device', default='0', help='GPU device (0, 1, ...) or cpu')
    parser.add_argument('--workers', type=int, default=8, help='데이터 로더 워커 수')

    # 실험 관리
    parser.add_argument('--project', type=str, default='runs/segment', help='프로젝트 디렉토리')
    parser.add_argument('--name', type=str, default=None, help='실험 이름')
    parser.add_argument('--resume', action='store_true', help='학습 재개')
    parser.add_argument('--pretrained', action='store_true', default=True, help='사전학습 모델 사용')

    # 하이퍼파라미터
    parser.add_argument('--lr0', type=float, default=0.01, help='초기 learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='최종 learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')

    # 검증만 수행
    parser.add_argument('--val-only', action='store_true', help='검증만 수행')
    parser.add_argument('--weights', type=str, help='검증할 모델 가중치 경로')

    args = parser.parse_args()

    if args.val_only:
        # 검증만 수행
        if not args.weights:
            print("Error: --weights 인자가 필요합니다")
        else:
            validate_model(
                model_path=args.weights,
                data_yaml=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device
            )
    else:
        # 학습 수행
        results, model = train_yolov11_seg(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
            resume=args.resume,
            pretrained=args.pretrained,
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            patience=args.patience,
            workers=args.workers,
        )
