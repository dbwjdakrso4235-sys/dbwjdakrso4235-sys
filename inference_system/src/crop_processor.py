#!/usr/bin/env python3
"""
SHP 기반 TIF 이미지 크롭 처리 모듈

기능:
- Shapefile로 정의된 영역만 대용량 TIF에서 추출
- 좌표계 자동 감지 및 변환
- 4-band TIF → RGB 자동 변환
- 메모리 효율적 처리 (window 읽기)

작성일: 2025-10-23
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.mask import mask as rasterio_mask
import geopandas as gpd
from shapely.geometry import mapping, Polygon, box
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CroppedRegion:
    """크롭된 영역 데이터"""
    polygon_id: str
    image: np.ndarray  # RGB, shape=(H, W, 3), dtype=uint8
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    transform: Any  # rasterio.Affine transform
    crs: Any  # rasterio CRS
    polygon: Polygon  # 원본 폴리곤
    metadata: Dict[str, Any]  # 추가 메타데이터


class CropProcessor:
    """Shapefile 기반 TIF 이미지 크롭 처리"""

    def __init__(self, tif_path: str, shp_path: str):
        """
        Args:
            tif_path: 대용량 TIF 파일 경로
            shp_path: Shapefile 경로
        """
        self.tif_path = Path(tif_path)
        self.shp_path = Path(shp_path)
        self.gdf = None
        self.tif_src = None
        self.tif_crs = None

        self._validate_paths()
        self._load_shapefile()

    def _validate_paths(self):
        """파일 경로 검증"""
        if not self.tif_path.exists():
            raise FileNotFoundError(f"TIF 파일을 찾을 수 없습니다: {self.tif_path}")

        if not self.shp_path.exists():
            raise FileNotFoundError(f"Shapefile을 찾을 수 없습니다: {self.shp_path}")

        logger.info(f"TIF 파일: {self.tif_path} ({self.tif_path.stat().st_size / 1e9:.2f} GB)")
        logger.info(f"Shapefile: {self.shp_path}")

    def _load_shapefile(self):
        """Shapefile 로드 및 검증"""
        try:
            self.gdf = gpd.read_file(self.shp_path)
            logger.info(f"Shapefile 로드 완료: {len(self.gdf)}개 폴리곤")
            logger.info(f"Shapefile CRS: {self.gdf.crs}")

            # 컬럼 정보 출력
            logger.info(f"Shapefile 컬럼: {list(self.gdf.columns)}")

        except Exception as e:
            logger.error(f"Shapefile 로드 실패: {e}")
            raise

    def _convert_4band_to_rgb(self, data: np.ndarray) -> np.ndarray:
        """
        4-band TIF → RGB 변환

        Args:
            data: (C, H, W) 형태의 배열, C>=3

        Returns:
            (H, W, 3) RGB 배열, dtype=uint8
        """
        # RGB 밴드만 추출 (밴드 1,2,3)
        if data.shape[0] > 3:
            rgb = data[:3]  # (3, H, W)
        else:
            rgb = data

        # (C, H, W) → (H, W, C)
        rgb = np.transpose(rgb, (1, 2, 0))

        # dtype에 따라 정규화
        if rgb.dtype == np.uint8:
            rgb_normalized = rgb
        elif rgb.dtype == np.uint16:
            # 16bit → 8bit (0-65535 → 0-255)
            rgb_normalized = (rgb / 256).astype(np.uint8)
        else:
            # float 등 기타 타입
            rgb_min, rgb_max = rgb.min(), rgb.max()
            if rgb_max > rgb_min:
                rgb_normalized = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
            else:
                rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)

        return rgb_normalized

    def _ensure_crs_match(self):
        """TIF와 Shapefile의 좌표계 일치 확인 및 변환"""
        with rasterio.open(self.tif_path) as src:
            self.tif_crs = src.crs
            logger.info(f"TIF CRS: {self.tif_crs}")

        if self.tif_crs != self.gdf.crs:
            logger.warning(f"좌표계 불일치 감지! TIF: {self.tif_crs}, SHP: {self.gdf.crs}")
            logger.info("Shapefile 좌표계를 TIF에 맞춰 변환 중...")
            self.gdf = self.gdf.to_crs(self.tif_crs)
            logger.info("좌표계 변환 완료")

    def crop_by_polygon(self, polygon_id: int, use_bounds: bool = True) -> Optional[CroppedRegion]:
        """
        특정 폴리곤 영역 크롭

        Args:
            polygon_id: 폴리곤 인덱스 (0부터 시작)
            use_bounds: True면 bounds로 크롭, False면 mask로 크롭

        Returns:
            CroppedRegion 객체 또는 None (실패 시)
        """
        if polygon_id >= len(self.gdf):
            logger.error(f"유효하지 않은 polygon_id: {polygon_id} (최대: {len(self.gdf)-1})")
            return None

        try:
            # 폴리곤 가져오기
            row = self.gdf.iloc[polygon_id]
            polygon = row.geometry
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)

            # 좌표계 일치 확인
            self._ensure_crs_match()

            with rasterio.open(self.tif_path) as src:
                if use_bounds:
                    # 방법 1: Bounds 기반 크롭 (빠름, 직사각형)
                    window = from_bounds(*bounds, src.transform)

                    # Window 크기 계산
                    height = int(window.height)
                    width = int(window.width)

                    if height <= 0 or width <= 0:
                        logger.warning(f"Polygon {polygon_id}: 유효하지 않은 크기 ({width}x{height})")
                        return None

                    # 데이터 읽기
                    data = src.read(window=window)

                    # Transform 계산
                    transform = src.window_transform(window)

                else:
                    # 방법 2: Mask 기반 크롭 (정확함, 폴리곤 모양)
                    out_image, out_transform = rasterio_mask(
                        src,
                        [mapping(polygon)],
                        crop=True,
                        all_touched=True
                    )
                    data = out_image
                    transform = out_transform

                # 4-band → RGB 변환
                rgb_image = self._convert_4band_to_rgb(data)

                # 메타데이터 수집
                metadata = {col: row[col] for col in self.gdf.columns if col != 'geometry'}

                # CroppedRegion 생성
                result = CroppedRegion(
                    polygon_id=f"polygon_{polygon_id}",
                    image=rgb_image,
                    bounds=bounds,
                    transform=transform,
                    crs=src.crs,
                    polygon=polygon,
                    metadata=metadata
                )

                logger.debug(f"Polygon {polygon_id}: 크롭 완료 ({rgb_image.shape[1]}x{rgb_image.shape[0]})")

                return result

        except Exception as e:
            logger.error(f"Polygon {polygon_id} 크롭 실패: {e}")
            return None

    def batch_crop(
        self,
        polygon_ids: Optional[List[int]] = None,
        use_bounds: bool = True,
        save_images: bool = False,
        output_dir: Optional[Path] = None
    ) -> List[CroppedRegion]:
        """
        여러 폴리곤 배치 크롭

        Args:
            polygon_ids: 크롭할 폴리곤 ID 리스트 (None이면 전체)
            use_bounds: bounds 기반 크롭 여부
            save_images: 크롭 이미지 저장 여부
            output_dir: 이미지 저장 디렉토리

        Returns:
            CroppedRegion 리스트
        """
        if polygon_ids is None:
            polygon_ids = list(range(len(self.gdf)))

        logger.info(f"배치 크롭 시작: {len(polygon_ids)}개 폴리곤")

        if save_images:
            if output_dir is None:
                output_dir = Path("inference_system/output/cropped_images")
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"크롭 이미지 저장 위치: {output_dir}")

        results = []

        for pid in tqdm(polygon_ids, desc="크롭 진행"):
            cropped = self.crop_by_polygon(pid, use_bounds=use_bounds)

            if cropped is not None:
                results.append(cropped)

                if save_images:
                    # 이미지가 비어있지 않은지 확인
                    if cropped.image.size > 0:
                        # BGR로 변환하여 저장 (OpenCV)
                        img_bgr = cv2.cvtColor(cropped.image, cv2.COLOR_RGB2BGR)
                        output_path = output_dir / f"{cropped.polygon_id}.png"
                        cv2.imwrite(str(output_path), img_bgr)
                    else:
                        logger.warning(f"{cropped.polygon_id}: 빈 이미지, 저장 건너뜀")

        logger.info(f"배치 크롭 완료: {len(results)}/{len(polygon_ids)} 성공")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Shapefile 통계 정보"""
        if self.gdf is None:
            return {}

        # 면적 계산 (평방미터)
        areas = self.gdf.geometry.area

        stats = {
            'total_polygons': len(self.gdf),
            'crs': str(self.gdf.crs),
            'bounds': self.gdf.total_bounds.tolist(),  # [minx, miny, maxx, maxy]
            'area_stats': {
                'min': float(areas.min()),
                'max': float(areas.max()),
                'mean': float(areas.mean()),
                'median': float(areas.median()),
                'total': float(areas.sum())
            },
            'columns': list(self.gdf.columns)
        }

        return stats

    def filter_by_area(self, min_area: float = 0, max_area: float = float('inf')) -> List[int]:
        """
        면적 기준으로 폴리곤 필터링

        Args:
            min_area: 최소 면적 (m²)
            max_area: 최대 면적 (m²)

        Returns:
            필터링된 폴리곤 ID 리스트
        """
        areas = self.gdf.geometry.area
        mask = (areas >= min_area) & (areas <= max_area)
        filtered_ids = self.gdf[mask].index.tolist()

        logger.info(f"면적 필터링: {len(filtered_ids)}/{len(self.gdf)} 폴리곤 선택")
        logger.info(f"조건: {min_area:.0f}m² ≤ area ≤ {max_area:.0f}m²")

        return filtered_ids

    def visualize_coverage(self, polygon_ids: Optional[List[int]] = None, output_path: Optional[Path] = None):
        """
        폴리곤 커버리지 시각화

        Args:
            polygon_ids: 시각화할 폴리곤 ID (None이면 전체)
            output_path: 저장 경로
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPLPolygon

        if polygon_ids is None:
            polygon_ids = list(range(len(self.gdf)))

        fig, ax = plt.subplots(figsize=(15, 15))

        # 전체 폴리곤 (회색)
        self.gdf.plot(ax=ax, facecolor='lightgray', edgecolor='black', alpha=0.3)

        # 선택된 폴리곤 (빨강)
        selected_gdf = self.gdf.iloc[polygon_ids]
        selected_gdf.plot(ax=ax, facecolor='red', edgecolor='darkred', alpha=0.5)

        ax.set_title(f'Polygon Coverage: {len(polygon_ids)}/{len(self.gdf)} selected', fontsize=16)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)

        if output_path is None:
            output_path = Path("inference_system/output/coverage.png")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"커버리지 맵 저장: {output_path}")
        plt.close()


def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='SHP 기반 TIF 크롭 처리')
    parser.add_argument('--tif', type=str, required=True, help='TIF 파일 경로')
    parser.add_argument('--shp', type=str, required=True, help='Shapefile 경로')
    parser.add_argument('--output', type=str, default='inference_system/output/cropped_images', help='출력 디렉토리')
    parser.add_argument('--limit', type=int, default=None, help='처리할 폴리곤 개수 제한')
    parser.add_argument('--min-area', type=float, default=0, help='최소 면적 (m²)')
    parser.add_argument('--max-area', type=float, default=float('inf'), help='최대 면적 (m²)')
    parser.add_argument('--visualize', action='store_true', help='커버리지 시각화')

    args = parser.parse_args()

    # CropProcessor 초기화
    processor = CropProcessor(args.tif, args.shp)

    # 통계 출력
    stats = processor.get_statistics()
    logger.info("=" * 80)
    logger.info("Shapefile 통계")
    logger.info("=" * 80)
    logger.info(f"총 폴리곤: {stats['total_polygons']}개")
    logger.info(f"좌표계: {stats['crs']}")
    logger.info(f"면적 범위: {stats['area_stats']['min']:.0f} ~ {stats['area_stats']['max']:.0f} m²")
    logger.info(f"평균 면적: {stats['area_stats']['mean']:.0f} m²")
    logger.info(f"총 면적: {stats['area_stats']['total']:.0f} m²")

    # 면적 기준 필터링
    polygon_ids = processor.filter_by_area(min_area=args.min_area, max_area=args.max_area)

    # 처리 개수 제한
    if args.limit is not None:
        polygon_ids = polygon_ids[:args.limit]
        logger.info(f"처리 제한: {len(polygon_ids)}개 폴리곤")

    # 커버리지 시각화
    if args.visualize:
        processor.visualize_coverage(polygon_ids)

    # 배치 크롭
    results = processor.batch_crop(
        polygon_ids=polygon_ids,
        save_images=True,
        output_dir=Path(args.output)
    )

    logger.info("=" * 80)
    logger.info(f"크롭 완료: {len(results)}개 이미지 생성")
    logger.info(f"저장 위치: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
