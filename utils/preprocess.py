"""
4-Band to RGB Conversion Utilities
곤포사일리지 데이터셋의 4밴드 TIF 이미지를 RGB 3밴드로 변환
"""

import rasterio
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm


def convert_4band_to_rgb(tif_path, output_path=None, save_format='png'):
    """
    4밴드 TIF 이미지를 RGB 3밴드로 변환

    Args:
        tif_path (str): 입력 4밴드 TIF 파일 경로
        output_path (str, optional): 출력 파일 경로. None이면 자동 생성
        save_format (str): 저장 포맷 ('png', 'jpg', 'tif')

    Returns:
        numpy.ndarray: RGB 이미지 (H, W, 3)
    """
    try:
        with rasterio.open(tif_path) as src:
            # 메타데이터 확인
            num_bands = src.count

            if num_bands >= 3:
                # 첫 3개 밴드 읽기 (R, G, B)
                # rasterio는 1-indexed이므로 [1, 2, 3]
                rgb = src.read([1, 2, 3])

                # (C, H, W) -> (H, W, C) 변환
                rgb = np.transpose(rgb, (1, 2, 0))

                # 데이터 타입 확인 및 정규화
                if rgb.dtype == np.uint8:
                    # 이미 uint8이면 그대로 사용
                    rgb_normalized = rgb
                elif rgb.dtype == np.uint16:
                    # uint16 -> uint8 변환
                    rgb_normalized = (rgb / 256).astype(np.uint8)
                else:
                    # float 등 다른 타입은 0-255 범위로 정규화
                    rgb_min = rgb.min()
                    rgb_max = rgb.max()
                    if rgb_max > rgb_min:
                        rgb_normalized = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                    else:
                        rgb_normalized = rgb.astype(np.uint8)

                # 저장
                if output_path is not None:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    if save_format.lower() == 'png':
                        Image.fromarray(rgb_normalized).save(output_path)
                    elif save_format.lower() == 'jpg':
                        Image.fromarray(rgb_normalized).save(output_path, quality=95)
                    elif save_format.lower() == 'tif':
                        # TIF로 저장 (3밴드)
                        profile = src.profile.copy()
                        profile.update(
                            count=3,
                            dtype=rasterio.uint8
                        )
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            for i in range(3):
                                dst.write(rgb_normalized[:, :, i], i + 1)

                return rgb_normalized

            else:
                raise ValueError(f"이미지에 충분한 밴드가 없습니다: {num_bands}개")

    except Exception as e:
        print(f"Error processing {tif_path}: {e}")
        return None


def batch_convert_dataset(input_dir, output_dir, splits=['train', 'val', 'test'],
                          save_format='png', verbose=True):
    """
    데이터셋 전체를 배치로 변환

    Args:
        input_dir (str): 입력 디렉토리 (images 폴더 포함)
        output_dir (str): 출력 디렉토리
        splits (list): 변환할 split 리스트
        save_format (str): 저장 포맷
        verbose (bool): 진행 상황 출력 여부

    Returns:
        dict: 변환 통계 정보
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'splits': {}
    }

    for split in splits:
        split_input = input_dir / split
        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)

        # TIF 파일 찾기
        tif_files = list(split_input.glob('*.tif')) + list(split_input.glob('*.TIF'))

        stats['splits'][split] = {
            'total': len(tif_files),
            'success': 0,
            'failed': 0
        }

        if verbose:
            print(f"\n{split} split 변환 중: {len(tif_files)}개 파일")
            iterator = tqdm(tif_files, desc=f"Converting {split}")
        else:
            iterator = tif_files

        for tif_file in iterator:
            # 출력 파일명 생성
            if save_format.lower() == 'tif':
                output_file = split_output / tif_file.name
            else:
                output_file = split_output / f"{tif_file.stem}.{save_format}"

            # 변환
            result = convert_4band_to_rgb(tif_file, output_file, save_format)

            stats['total'] += 1
            stats['splits'][split]['total'] += 1

            if result is not None:
                stats['success'] += 1
                stats['splits'][split]['success'] += 1
            else:
                stats['failed'] += 1
                stats['splits'][split]['failed'] += 1

    return stats


def verify_conversion(original_path, converted_path):
    """
    변환 결과 검증

    Args:
        original_path (str): 원본 TIF 경로
        converted_path (str): 변환된 이미지 경로

    Returns:
        dict: 검증 결과
    """
    results = {
        'original_bands': None,
        'original_shape': None,
        'converted_shape': None,
        'match': False
    }

    try:
        # 원본 확인
        with rasterio.open(original_path) as src:
            results['original_bands'] = src.count
            results['original_shape'] = (src.height, src.width)

        # 변환된 이미지 확인
        converted = cv2.imread(str(converted_path))
        if converted is not None:
            results['converted_shape'] = converted.shape[:2]  # (H, W)
            results['match'] = (results['original_shape'] == results['converted_shape'])

        return results

    except Exception as e:
        print(f"Verification error: {e}")
        return results


def check_image_stats(image_path):
    """
    이미지 통계 확인 (디버깅용)

    Args:
        image_path (str): 이미지 경로

    Returns:
        dict: 통계 정보
    """
    try:
        with rasterio.open(image_path) as src:
            stats = {
                'bands': src.count,
                'dtype': src.dtypes[0],
                'shape': (src.height, src.width),
                'crs': src.crs,
                'transform': src.transform,
            }

            # 각 밴드의 min/max 확인
            band_stats = []
            for i in range(1, src.count + 1):
                band = src.read(i)
                band_stats.append({
                    'band': i,
                    'min': float(band.min()),
                    'max': float(band.max()),
                    'mean': float(band.mean()),
                    'std': float(band.std())
                })

            stats['band_stats'] = band_stats
            return stats

    except Exception as e:
        print(f"Error reading stats: {e}")
        return None


if __name__ == "__main__":
    # 테스트 코드
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing conversion on: {test_file}")

        # 통계 확인
        stats = check_image_stats(test_file)
        if stats:
            print("\n=== 원본 이미지 통계 ===")
            print(f"Bands: {stats['bands']}")
            print(f"Dtype: {stats['dtype']}")
            print(f"Shape: {stats['shape']}")
            print("\nBand Statistics:")
            for bs in stats['band_stats']:
                print(f"  Band {bs['band']}: min={bs['min']:.2f}, max={bs['max']:.2f}, mean={bs['mean']:.2f}")

        # 변환 테스트
        output_file = Path(test_file).parent / f"{Path(test_file).stem}_rgb.png"
        rgb = convert_4band_to_rgb(test_file, output_file, save_format='png')

        if rgb is not None:
            print(f"\n변환 성공! 저장 위치: {output_file}")
            print(f"RGB shape: {rgb.shape}")
            print(f"RGB dtype: {rgb.dtype}")
            print(f"RGB range: [{rgb.min()}, {rgb.max()}]")
    else:
        print("Usage: python preprocess.py <path_to_tif_file>")
        print("\nExample:")
        print("  python utils/preprocess.py E:/namwon_ai/dataset_silage_bale/images/train/1F001D60362.tif")
