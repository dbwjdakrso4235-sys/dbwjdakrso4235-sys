"""
SHP 파일 CRS 수정
"""
import geopandas as gpd
import os

print("=" * 80)
print("SHP 파일 CRS 수정")
print("=" * 80)
print()

# 파일 경로
input_shp = r"E:\namwon_ai\gonpo\gonpo_251028.shp"
output_shp = r"E:\namwon_ai\gonpo\gonpo_251028_fixed.shp"

print(f"입력: {input_shp}")
print(f"출력: {output_shp}")
print()

# 원본 로드
print("[1] 원본 SHP 로드 중...")
shp = gpd.read_file(input_shp)
print(f"    원본 CRS: {shp.crs}")
print(f"    폴리곤 수: {len(shp)}")
print()

# CRS 수정
print("[2] CRS를 EPSG:5186으로 수정 중...")
shp_fixed = shp.set_crs("EPSG:5186", allow_override=True)
print(f"    수정된 CRS: {shp_fixed.crs}")
print()

# 저장
print("[3] 수정된 파일 저장 중...")
shp_fixed.to_file(output_shp)
print(f"    저장 완료: {output_shp}")
print()

# 검증
print("[4] 저장된 파일 검증 중...")
shp_verify = gpd.read_file(output_shp)
print(f"    CRS: {shp_verify.crs}")
print(f"    폴리곤 수: {len(shp_verify)}")
print()

# 파일 크기 확인
file_size = os.path.getsize(output_shp)
print(f"    파일 크기: {file_size:,} bytes")
print()

print("=" * 80)
print("[완료] SHP 파일 CRS 수정 완료!")
print("=" * 80)
print()

print("다음 단계:")
print(f"  1. 수정된 SHP 파일 사용: {output_shp}")
print(f"  2. 추론 스크립트에서 경로 변경")
print(f"  3. 추론 재실행")
