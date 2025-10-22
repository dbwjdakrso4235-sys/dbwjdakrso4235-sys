"""
데이터셋 전처리 스크립트
4-band TIF 이미지를 RGB로 변환하고 새로운 데이터셋 생성
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.preprocess import batch_convert_dataset, verify_conversion, check_image_stats
import shutil
import yaml


def preprocess_silage_bale_dataset(
    input_dataset_path="E:/namwon_ai/dataset_silage_bale",
    output_dataset_path="E:/namwon_ai/dataset_silage_bale_rgb",
    save_format='png',
    copy_labels=True,
    create_yaml=True
):
    """
    곤포사일리지 데이터셋 전처리

    Args:
        input_dataset_path (str): 원본 데이터셋 경로
        output_dataset_path (str): 출력 데이터셋 경로
        save_format (str): 이미지 저장 포맷 ('png', 'jpg', 'tif')
        copy_labels (bool): 라벨 파일 복사 여부
        create_yaml (bool): dataset.yaml 생성 여부
    """
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)

    print("=" * 60)
    print("곤포사일리지 데이터셋 전처리 시작")
    print("=" * 60)
    print(f"입력 경로: {input_path}")
    print(f"출력 경로: {output_path}")
    print(f"저장 포맷: {save_format}")
    print("=" * 60)

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 이미지 변환
    print("\n[1/3] 이미지 변환 중...")
    input_images = input_path / "images"
    output_images = output_path / "images"

    stats = batch_convert_dataset(
        input_dir=input_images,
        output_dir=output_images,
        splits=['train', 'val', 'test'],
        save_format=save_format,
        verbose=True
    )

    print("\n변환 완료!")
    print(f"  총 {stats['total']}개 파일")
    print(f"  성공: {stats['success']}개")
    print(f"  실패: {stats['failed']}개")

    for split, split_stats in stats['splits'].items():
        print(f"  - {split}: {split_stats['success']}/{split_stats['total']} 성공")

    # 2. 라벨 복사
    if copy_labels:
        print("\n[2/3] 라벨 파일 복사 중...")
        input_labels = input_path / "labels"
        output_labels = output_path / "labels"

        for split in ['train', 'val', 'test']:
            src_dir = input_labels / split
            dst_dir = output_labels / split
            dst_dir.mkdir(parents=True, exist_ok=True)

            if src_dir.exists():
                label_files = list(src_dir.glob('*.txt'))
                print(f"  {split}: {len(label_files)}개 라벨 파일 복사")

                for label_file in label_files:
                    shutil.copy2(label_file, dst_dir / label_file.name)

    # 3. dataset.yaml 생성
    if create_yaml:
        print("\n[3/3] dataset.yaml 생성 중...")

        # 원본 yaml 읽기
        original_yaml_path = input_path / "dataset.yaml"
        if original_yaml_path.exists():
            with open(original_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 경로 업데이트
            config['path'] = str(output_path).replace('\\', '/')

            # 이미지 포맷 업데이트
            if save_format != 'tif':
                config['image_format'] = save_format

            # 저장
            output_yaml_path = output_path / "dataset.yaml"
            with open(output_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            print(f"  dataset.yaml 생성 완료: {output_yaml_path}")
        else:
            print(f"  경고: 원본 dataset.yaml을 찾을 수 없습니다: {original_yaml_path}")

    # 4. 변환 검증 (샘플)
    print("\n[검증] 변환 결과 샘플 확인 중...")
    train_original = input_images / "train"
    train_converted = output_images / "train"

    if train_original.exists() and train_converted.exists():
        original_files = list(train_original.glob('*.tif'))[:3]  # 처음 3개만

        for orig_file in original_files:
            if save_format.lower() == 'tif':
                conv_file = train_converted / orig_file.name
            else:
                conv_file = train_converted / f"{orig_file.stem}.{save_format}"

            if conv_file.exists():
                result = verify_conversion(orig_file, conv_file)
                print(f"  {orig_file.name}: ", end='')
                if result['match']:
                    print(f"[OK] {result['original_shape']}")
                else:
                    print(f"[MISMATCH] {result['original_shape']} -> {result['converted_shape']}")

    print("\n" + "=" * 60)
    print("전처리 완료!")
    print("=" * 60)
    print(f"\n새로운 데이터셋 경로: {output_path}")
    print(f"학습에 사용할 dataset.yaml: {output_path / 'dataset.yaml'}")
    print("\n다음 단계: 학습 스크립트 실행")
    print("=" * 60)


def test_single_image(image_path):
    """
    단일 이미지 테스트

    Args:
        image_path (str): 테스트할 이미지 경로
    """
    print("\n=== 단일 이미지 테스트 ===")
    print(f"이미지: {image_path}")

    # 통계 확인
    stats = check_image_stats(image_path)
    if stats:
        print(f"\n원본 이미지 정보:")
        print(f"  Bands: {stats['bands']}")
        print(f"  Dtype: {stats['dtype']}")
        print(f"  Shape: {stats['shape']}")

        print(f"\n밴드별 통계:")
        for bs in stats['band_stats']:
            print(f"  Band {bs['band']}: min={bs['min']:.1f}, max={bs['max']:.1f}, mean={bs['mean']:.1f}")

    # 변환 테스트
    from utils.preprocess import convert_4band_to_rgb
    output_file = Path(image_path).parent / f"{Path(image_path).stem}_test_rgb.png"

    rgb = convert_4band_to_rgb(image_path, output_file, save_format='png')

    if rgb is not None:
        print(f"\n변환 성공!")
        print(f"  저장 위치: {output_file}")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  RGB range: [{rgb.min()}, {rgb.max()}]")
    else:
        print(f"\n변환 실패!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='곤포사일리지 데이터셋 전처리')
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'test'],
        help='실행 모드: full (전체 변환) 또는 test (단일 이미지 테스트)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='E:/namwon_ai/dataset_silage_bale',
        help='입력 데이터셋 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='E:/namwon_ai/dataset_silage_bale_rgb',
        help='출력 데이터셋 경로'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'jpg', 'tif'],
        help='저장 포맷'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        help='테스트할 이미지 경로 (mode=test일 때 사용)'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        # 전체 데이터셋 변환
        preprocess_silage_bale_dataset(
            input_dataset_path=args.input,
            output_dataset_path=args.output,
            save_format=args.format,
            copy_labels=True,
            create_yaml=True
        )
    elif args.mode == 'test':
        # 단일 이미지 테스트
        if args.test_image:
            test_single_image(args.test_image)
        else:
            # 기본 테스트 이미지
            default_test = "E:/namwon_ai/dataset_silage_bale/images/train/1F001D60362.tif"
            print(f"테스트 이미지가 지정되지 않아 기본 이미지 사용: {default_test}")
            test_single_image(default_test)
