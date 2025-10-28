"""
TIF와 SHP 공간적 중첩 확인
"""
import rasterio
import geopandas as gpd
from shapely.geometry import box

print("=" * 80)
print("TIF와 SHP 공간적 중첩 확인")
print("=" * 80)
print()

# 파일 경로
tif_path = r"E:\namwon_ai\input_tif\금지면_1차.tif"
shp_path = r"E:\namwon_ai\gonpo\gonpo_251028.shp"

print("[파일 정보]")
print(f"  TIF: {tif_path}")
print(f"  SHP: {shp_path}")
print()

# TIF 정보 읽기
print("[TIF 정보]")
with rasterio.open(tif_path) as src:
    tif_bounds = src.bounds
    tif_crs = src.crs
    print(f"  CRS: {tif_crs}")
    print(f"  Bounds (minx, miny, maxx, maxy):")
    print(f"    {tif_bounds.left:.2f}, {tif_bounds.bottom:.2f}, {tif_bounds.right:.2f}, {tif_bounds.top:.2f}")
    print(f"  Width x Height: {src.width} x {src.height}")
    print(f"  Resolution: {src.res}")

    # TIF 범위를 폴리곤으로
    tif_box = box(tif_bounds.left, tif_bounds.bottom, tif_bounds.right, tif_bounds.top)
    tif_area = tif_box.area
    print(f"  면적: {tif_area:,.0f} (CRS 단위)")
print()

# SHP 정보 읽기
print("[SHP 정보]")
shp = gpd.read_file(shp_path)
print(f"  CRS: {shp.crs}")
print(f"  폴리곤 수: {len(shp)}")
print(f"  컬럼: {list(shp.columns)}")
print()

# SHP의 전체 범위
shp_bounds = shp.total_bounds
print(f"  Bounds (minx, miny, maxx, maxy):")
print(f"    {shp_bounds[0]:.6f}, {shp_bounds[1]:.6f}, {shp_bounds[2]:.6f}, {shp_bounds[3]:.6f}")
print()

# 각 폴리곤 정보
print("[폴리곤별 정보]")
for idx, row in shp.iterrows():
    geom = row.geometry
    bounds = geom.bounds
    area = geom.area
    print(f"\n  Polygon {idx}:")
    print(f"    ID: {row.get('id', 'N/A')}")
    print(f"    Bounds: ({bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f})")
    print(f"    면적: {area:.6f} (CRS 단위)")
    print(f"    중심점: ({geom.centroid.x:.6f}, {geom.centroid.y:.6f})")
print()

# 좌표계 변환하여 중첩 확인
print("[좌표계 변환 및 중첩 확인]")
print(f"  SHP CRS: {shp.crs}")
print(f"  TIF CRS: {tif_crs}")

if shp.crs != tif_crs:
    print(f"  -> SHP를 {tif_crs}로 변환 중...")
    shp_transformed = shp.to_crs(tif_crs)
    print(f"  변환 완료")
else:
    shp_transformed = shp
    print(f"  좌표계 일치 (변환 불필요)")
print()

# 변환 후 범위 확인
print("[변환 후 SHP 범위]")
shp_transformed_bounds = shp_transformed.total_bounds
print(f"  Bounds (minx, miny, maxx, maxy):")
print(f"    {shp_transformed_bounds[0]:.2f}, {shp_transformed_bounds[1]:.2f}, {shp_transformed_bounds[2]:.2f}, {shp_transformed_bounds[3]:.2f}")
print()

# TIF 범위와 비교
print("[중첩 분석]")
print("\nTIF 범위:")
print(f"  X: {tif_bounds.left:.2f} ~ {tif_bounds.right:.2f}")
print(f"  Y: {tif_bounds.bottom:.2f} ~ {tif_bounds.top:.2f}")

print("\nSHP 범위 (변환 후):")
print(f"  X: {shp_transformed_bounds[0]:.2f} ~ {shp_transformed_bounds[2]:.2f}")
print(f"  Y: {shp_transformed_bounds[1]:.2f} ~ {shp_transformed_bounds[3]:.2f}")
print()

# 중첩 여부 판단
overlap_x = not (shp_transformed_bounds[2] < tif_bounds.left or shp_transformed_bounds[0] > tif_bounds.right)
overlap_y = not (shp_transformed_bounds[3] < tif_bounds.bottom or shp_transformed_bounds[1] > tif_bounds.top)
overlap = overlap_x and overlap_y

print("=" * 80)
if overlap:
    print("[결과] TIF와 SHP가 중첩됩니다!")
else:
    print("[결과] TIF와 SHP가 중첩되지 않습니다!")
    if not overlap_x:
        print("  -> X축(경도) 범위가 겹치지 않음")
    if not overlap_y:
        print("  -> Y축(위도) 범위가 겹치지 않음")
print("=" * 80)
print()

# 각 폴리곤별 중첩 확인
print("[폴리곤별 중첩 상세]")

tif_polygon = box(tif_bounds.left, tif_bounds.bottom, tif_bounds.right, tif_bounds.top)

for idx, row in shp_transformed.iterrows():
    poly = row.geometry
    intersects = poly.intersects(tif_polygon)

    print(f"\n  Polygon {idx}:")
    print(f"    중첩 여부: {'[O] 중첩' if intersects else '[X] 중첩 없음'}")

    if intersects:
        intersection = poly.intersection(tif_polygon)
        intersection_area = intersection.area
        poly_area = poly.area
        overlap_ratio = (intersection_area / poly_area * 100) if poly_area > 0 else 0

        print(f"    중첩 면적: {intersection_area:,.2f} m^2")
        print(f"    폴리곤 면적: {poly_area:,.2f} m^2")
        print(f"    중첩 비율: {overlap_ratio:.1f}%")
    else:
        # TIF 중심과의 거리 계산
        tif_center_x = (tif_bounds.left + tif_bounds.right) / 2
        tif_center_y = (tif_bounds.bottom + tif_bounds.top) / 2
        poly_center = poly.centroid

        distance = ((poly_center.x - tif_center_x)**2 + (poly_center.y - tif_center_y)**2)**0.5
        print(f"    TIF 중심과의 거리: {distance:,.2f} m")

        # 어느 방향에 있는지
        dx = poly_center.x - tif_center_x
        dy = poly_center.y - tif_center_y
        direction = []
        if dy > 0:
            direction.append("북쪽")
        else:
            direction.append("남쪽")
        if dx > 0:
            direction.append("동쪽")
        else:
            direction.append("서쪽")

        print(f"    위치: TIF 중심의 {', '.join(direction)}")

print()
print("=" * 80)
print("분석 완료")
print("=" * 80)
