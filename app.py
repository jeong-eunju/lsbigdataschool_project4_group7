from shiny import App, ui, render, reactive
from pathlib import Path
import platform
import matplotlib
import json
import folium
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
import gc  # 가비지 컬렉터 추가
import psutil  # 메모리 모니터링
from htmltools import TagList, tags
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
WWW_DIR  = BASE_DIR / "www"

# ====== [1] 한글 폰트 설정 ======
if platform.system() == 'Windows':
    matplotlib.rc('font', family='맑은 고딕')
elif platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# ====== [2] 메모리 최적화 함수들 ======
def optimize_dataframe_memory(df):
    """DataFrame의 메모리 사용량을 최적화하는 함수"""
    if df.empty:
        return df
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # 숫자형 컬럼만 최적화 (category 변환은 제외)
        if col_type != 'object' and 'category' not in str(col_type):
            try:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            except Exception:
                # 오류 발생 시 원본 유지
                continue
    
    return df

def get_memory_usage():
    """현재 메모리 사용량 반환"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024   # MB
        }
    except:
        return {'rss': 0, 'vms': 0}

def log_memory_usage(stage):
    """메모리 사용량 로깅"""
    usage = get_memory_usage()
    print(f"[{stage}] 메모리 사용량 - RSS: {usage['rss']:.1f}MB")

# ====== [3] 경상북도용 유틸리티 함수 ======
def unify_and_filter_region(df: pd.DataFrame, col: str, second_col: str = None) -> pd.DataFrame:
    """지역명을 통일하고 필터링하는 함수"""
    try:
        df = df.copy()

        # 시군구 단위 기준 정리
        region_keywords = [
            "포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", "문경시", "경산시",
            "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", "성주군", "칠곡군", "예천군", "봉화군",
            "울진군", "울릉군"
        ]

        pattern = "(" + "|".join(region_keywords) + ")"

        if second_col and second_col in df.columns:
            df['region_raw'] = df[col].astype(str).str.strip() + " " + df[second_col].astype(str).str.strip()
            df['region'] = df['region_raw'].str.extract(pattern)[0]
        else:
            df['region'] = df[col].astype(str).str.strip().str.extract(pattern)[0]

        # 군위군 제거
        result = df[df['region'] != '군위군'].copy()
        return optimize_dataframe_memory(result)
    
    except Exception as e:
        print(f"지역 필터링 오류: {e}")
        # 오류 발생 시 원본 DataFrame에 기본 region 컬럼 추가
        df_copy = df.copy()
        df_copy['region'] = '기타'
        return df_copy

# ====== [4] 영천시용 거리 계산 함수 ======
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

@np.vectorize
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """벡터화된 거리 계산 함수"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ====== [5] 경상북도 고정 데이터 ======
REGIONS = ["포항시","경주시","김천시","안동시","구미시","영주시","영천시","상주시","문경시","경산시",
           "의성군","청송군","영양군","영덕군","청도군","고령군","성주군","칠곡군","예천군","봉화군",
           "울진군","울릉군"]

POPULATION = np.array([498296,257668,138999,154788,410306,99894,101185,93081,67674,285618,
                      49336,23867,15494,34338,41641,32350,43543,111928,54868,28988,
                      47872,9199], dtype=np.int32)

pop_df = pd.DataFrame({
    '시군': REGIONS,  # 일반 리스트로 유지
    '인구수': POPULATION
})

# 22개 지역별 고유 색상 배정
REGION_COLORS = {
    "포항시": "#d14747",
    "경주시": "#d16c47",
    "김천시": "#d19247",
    "안동시": "#d1b847",
    "구미시": "#c4d147",
    "영주시": "#9fd147",
    "영천시": "#79d147",
    "상주시": "#53d147",
    "문경시": "#47d160",
    "경산시": "#47d185",
    "의성군": "#47d1ab",
    "청송군": "#47d1d1",
    "영양군": "#47abd1",
    "영덕군": "#4785d1",
    "청도군": "#4760d1",
    "고령군": "#5347d1",
    "성주군": "#7947d1",
    "칠곡군": "#9f47d1",
    "예천군": "#c447d1",
    "봉화군": "#d147b8",
    "울진군": "#d14792",
    "울릉군": "#d1476c",
}



# ====== [6] 영천시 데이터 로딩 (수정된 버전) ======
try:
    # 영천시 저수지 데이터
    df_yeongcheon = pd.read_excel(DATA_DIR / '경상북도_영천시_저수지및댐.xlsx').dropna()
    
    # 기본적인 데이터 타입 확인 및 변환
    required_cols = ['시설명', '소재지지번주소', '위도', '경도', '면적', '둘레']
    for col in required_cols:
        if col not in df_yeongcheon.columns:
            print(f"필수 컬럼 {col}이 없습니다.")
            raise Exception(f"필수 컬럼 {col}이 없습니다.")
    
    # 숫자형 컬럼 변환
    for col in ['위도', '경도', '면적', '둘레']:
        df_yeongcheon[col] = pd.to_numeric(df_yeongcheon[col], errors='coerce')
    
    # 결측값 제거
    df_yeongcheon = df_yeongcheon.dropna(subset=['위도', '경도', '면적', '둘레'])
    
    if len(df_yeongcheon) == 0:
        raise Exception("유효한 데이터가 없습니다.")
    
    df_yeongcheon = optimize_dataframe_memory(df_yeongcheon)
    print(f"영천시 저수지 데이터 로딩 완료: {len(df_yeongcheon)}개")
    
    # 반려동물 동반 가능 시설 위치 리스트 (numpy array로 최적화)
    locations = np.array([
        [36.01841762, 128.929917], [35.97173253, 128.939907], [35.95973738, 128.93954],
        [35.96594607, 128.918217], [35.93153836, 128.87455], [35.91826818, 129.011153],
        [35.96426248, 128.924962], [35.93361167, 128.876258], [35.9719872, 128.941242],
        [35.96837891, 128.933538], [35.97188757, 128.939926], [35.99097629, 128.823406],
        [35.96468716, 128.938253], [35.96440713, 128.926174], [35.96371503, 128.939265],
        [35.93358914, 128.871295], [35.9721287, 128.93577], [35.93067846, 128.870576],
        [36.04169724, 128.787972], [35.96482654, 128.93923], [35.96359599, 128.936706],
        [35.97502046, 128.947498], [35.95795857, 128.913141], [35.98890075, 128.95512],
        [35.9581778, 128.913096], [35.96122769, 128.92841], [35.97523599, 128.94772],
        [36.04494524, 128.799646], [35.97216113, 128.937574], [36.03276243, 128.889247],
        [35.94410222, 128.897127], [36.01859865, 128.929978], [35.99026252, 128.794311],
        [36.12326392, 128.901235], [35.97438618, 128.945949], [35.98250081, 128.95221],
        [35.9722037, 128.935942], [35.95180542, 128.930731], [35.90275615, 128.856568],
        [35.9584334, 128.909988], [36.05347739, 128.89304], [35.97592922, 128.953099],
        [35.90275615, 128.856568],
    ], dtype=np.float32)
    
    # 반경 2km 시설 수 계산 (벡터화)
    def count_nearby_facilities_vectorized(df_lats, df_lons, locations, radius_km=2):
        facility_counts = np.zeros(len(df_lats), dtype=np.int8)
        
        try:
            for i, (lat, lon) in enumerate(zip(df_lats, df_lons)):
                if pd.isna(lat) or pd.isna(lon):
                    facility_counts[i] = 0
                    continue
                distances = haversine_vectorized(lat, lon, locations[:, 0], locations[:, 1])
                facility_counts[i] = np.sum(distances <= radius_km)
        except Exception as e:
            print(f"시설 수 계산 오류: {e}")
            facility_counts = np.zeros(len(df_lats), dtype=np.int8)
        
        return facility_counts

    df_yeongcheon['반경2km_시설수'] = count_nearby_facilities_vectorized(
        df_yeongcheon['위도'].values, 
        df_yeongcheon['경도'].values, 
        locations
    )

    # 정규화 함수
    def normalize(series): 
        try:
            # 숫자형 데이터가 아닌 경우 처리
            if not pd.api.types.is_numeric_dtype(series):
                return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)
            
            min_val, max_val = series.min(), series.max()
            if max_val == min_val or pd.isna(min_val) or pd.isna(max_val):
                return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)
            return ((series - min_val) / (max_val - min_val)).astype(np.float32)
        except Exception:
            # 오류 발생 시 0으로 채운 시리즈 반환
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=np.float32)

    df_yeongcheon['면적_정규화'] = normalize(df_yeongcheon['면적'])
    df_yeongcheon['둘레_정규화'] = normalize(df_yeongcheon['둘레'])
    df_yeongcheon['시설수_정규화'] = normalize(df_yeongcheon['반경2km_시설수'])

    # 중심지와의 거리 계산
    centers = np.array([
        [35.92646737, 128.8823282], [36.01411727, 129.02104376],
        [35.96487293, 128.94139421], [36.05822495, 128.89287688],
        [36.03249647, 128.79147335]
    ], dtype=np.float32)
    
    def get_closest_center_distance_vectorized(lats, lons, centers):
        min_distances = np.full(len(lats), np.inf, dtype=np.float32)
        
        try:
            for center_lat, center_lon in centers:
                distances = haversine_vectorized(lats, lons, center_lat, center_lon)
                min_distances = np.minimum(min_distances, distances)
        except Exception as e:
            print(f"거리 계산 오류: {e}")
            min_distances = np.ones(len(lats), dtype=np.float32)
        
        return min_distances

    df_yeongcheon['중심거리_km'] = get_closest_center_distance_vectorized(
        df_yeongcheon['위도'].values, 
        df_yeongcheon['경도'].values, 
        centers
    )
    df_yeongcheon['거리_정규화'] = 1 - normalize(df_yeongcheon['중심거리_km'])

    # 적합도 점수 계산
    df_yeongcheon['적합도점수'] = (
        0.3 * df_yeongcheon['면적_정규화'] +
        0.3 * df_yeongcheon['둘레_정규화'] +
        0.2 * df_yeongcheon['거리_정규화'] +
        0.2 * df_yeongcheon['시설수_정규화']
    ).astype(np.float32)

    # ====== 영천시 새로운 shapefile 사용 ======
    try:
        # 새로운 영천시 읍면동 shapefile 로딩
        gdf_yeongcheon = gpd.read_file(DATA_DIR / "DA_EMD_202307.shp", encoding='cp949')
        gdf_yeongcheon = gdf_yeongcheon.to_crs(epsg=4326)
        
        # 실제 컬럼명 확인 및 출력
        print(f"영천시 Shapefile 컬럼명: {list(gdf_yeongcheon.columns)}")
        print(f"영천시 데이터 샘플:")
        if len(gdf_yeongcheon) > 0:
            print(gdf_yeongcheon.head(3))
        
        # 가능한 컬럼명 패턴들을 시도
        code_col = None
        name_col = None
        
        # 코드 컬럼 찾기
        for col in gdf_yeongcheon.columns:
            col_upper = col.upper()
            if any(pattern in col_upper for pattern in ['CD', 'CODE', '코드']):
                code_col = col
                break
        
        # 이름 컬럼 찾기  
        for col in gdf_yeongcheon.columns:
            col_upper = col.upper()
            if any(pattern in col_upper for pattern in ['NM', 'NAME', '명', '이름']):
                name_col = col
                break
        
        # 컬럼을 찾지 못한 경우 기본값 시도
        if code_col is None:
            possible_code_cols = ['ADM_DR_CD', 'EMD_CD', 'DONG_CD', 'CD']
            for col in possible_code_cols:
                if col in gdf_yeongcheon.columns:
                    code_col = col
                    break
        
        if name_col is None:
            possible_name_cols = ['ADM_DR_NM', 'EMD_NM', 'DONG_NM', 'NM']
            for col in possible_name_cols:
                if col in gdf_yeongcheon.columns:
                    name_col = col
                    break
        
        print(f"영천시 사용할 코드 컬럼: {code_col}")
        print(f"영천시 사용할 이름 컬럼: {name_col}")
        
        # 컬럼을 찾은 경우에만 진행
        if code_col and name_col:
            # 필요한 컬럼만 유지
            gdf_yeongcheon = gdf_yeongcheon[['geometry', code_col, name_col]].copy()
            
            # 영천시 행정동 코드 리스트 (기존과 동일)
            yc_codes = [
                "37070330", "37070340", "37070350", "37070360", "37070370", "37070380",
                "37070520", "37070540", "37070550", "37070510", "37070310", "37070320",
                "37070110", "37070390", "37070400", "37070530"
            ]
            
            # 영천시 해당 행정동만 필터링
            print(f"영천시 필터링 전 데이터 수: {len(gdf_yeongcheon)}")
            gdf_yeongcheon = gdf_yeongcheon[gdf_yeongcheon[code_col].astype(str).isin(yc_codes)]
            print(f"영천시 필터링 후 데이터 수: {len(gdf_yeongcheon)}")
            
            if len(gdf_yeongcheon) == 0:
                print("경고: 영천시 해당 행정동 데이터가 없습니다.")
                print(f"찾고 있는 코드: {yc_codes}")
                sample_codes = gdf_yeongcheon[code_col].astype(str).unique()[:10]  # 처음 10개만
                print(f"파일에 있는 코드 샘플: {sample_codes}")
                raise Exception("영천시 데이터 없음")
            
            # 컬럼명 표준화
            gdf_yeongcheon = gdf_yeongcheon.rename(columns={code_col: 'ADM_CD', name_col: 'ADM_NM'})

            # 지오메트리 단순화 (성능 향상)
            gdf_yeongcheon['geometry'] = gdf_yeongcheon['geometry'].simplify(0.001)

            # 저수지 포인트 데이터 생성
            df_yeongcheon['geometry'] = [Point(xy) for xy in zip(df_yeongcheon['경도'], df_yeongcheon['위도'])]
            df_yeongcheon_gdf = gpd.GeoDataFrame(df_yeongcheon, geometry='geometry', crs='EPSG:4326')
            
            # Spatial join으로 행정동 정보 결합
            joined = gpd.sjoin(df_yeongcheon_gdf, gdf_yeongcheon[['geometry', 'ADM_CD', 'ADM_NM']], 
                             how='left', predicate='within')
            
            # 행정동명 컬럼 추가
            df_yeongcheon['행정동명'] = joined['ADM_NM'].values
            
            # 메모리 정리
            del df_yeongcheon_gdf, joined
            gc.collect()
            
            # 고유 지역 리스트 생성
            unique_areas = ["전체"] + sorted(df_yeongcheon['행정동명'].dropna().unique().tolist())
            print(f"영천시 행정동 데이터 결합 완료: {len(unique_areas)-1}개 행정동")
            print(f"영천시 포함된 행정동: {unique_areas[1:]}")
            
        else:
            print("영천시 적절한 컬럼을 찾을 수 없습니다.")
            print(f"영천시 전체 컬럼 목록: {list(gdf_yeongcheon.columns)}")
            print("영천시 기본값으로 진행합니다.")
            raise Exception("영천시 컬럼 매핑 실패")
        
    except Exception as e:
        print(f"영천시 Shapefile 로딩 오류: {e}")
        print("영천시 기존 방식으로 대체합니다.")
        gdf_yeongcheon = gpd.GeoDataFrame()
        df_yeongcheon['행정동명'] = '알 수 없음'
        unique_areas = ["전체", "알 수 없음"]
        
    # 최종 최적화
    df_yeongcheon = optimize_dataframe_memory(df_yeongcheon)
    print(f"영천시 데이터 로딩 완료: {len(df_yeongcheon)}개 저수지")
    
except Exception as e:
    print(f"영천시 데이터 로딩 오류: {e}")
    df_yeongcheon = pd.DataFrame()
    gdf_yeongcheon = gpd.GeoDataFrame()
    unique_areas = ["전체"]

# ====== [7] 경상북도 Shapefile 로딩 (수정된 버전) ======
def load_and_optimize_shapefile(shp_path: str):
    """새로운 경상북도 시군구 전용 Shapefile을 로드하고 최적화"""
    try:
        # 새로운 경상북도 시군구 shapefile 로딩
        gdf = gpd.read_file(shp_path, encoding='cp949')
        
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        elif gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')

        # 실제 컬럼명 확인 및 출력
        print(f"경상북도 Shapefile 컬럼명: {list(gdf.columns)}")
        print(f"경상북도 데이터 샘플:")
        if len(gdf) > 0:
            print(gdf.head(3))
        
        # 가능한 컬럼명 패턴들을 시도
        code_col = None
        name_col = None
        
        # 코드 컬럼 찾기
        for col in gdf.columns:
            col_upper = col.upper()
            if any(pattern in col_upper for pattern in ['CD', 'CODE', '코드']):
                code_col = col
                break
        
        # 이름 컬럼 찾기  
        for col in gdf.columns:
            col_upper = col.upper()
            if any(pattern in col_upper for pattern in ['NM', 'NAME', '명', '이름']):
                name_col = col
                break
        
        # 컬럼을 찾지 못한 경우 기본값 시도
        if code_col is None:
            possible_code_cols = ['ADM_CD', 'SIG_CD', 'SIGUN_CD', 'CD']
            for col in possible_code_cols:
                if col in gdf.columns:
                    code_col = col
                    break
        
        if name_col is None:
            possible_name_cols = ['ADM_NM', 'SIG_NM', 'SIGUN_NM', 'NM']
            for col in possible_name_cols:
                if col in gdf.columns:
                    name_col = col
                    break
        
        print(f"경상북도 사용할 코드 컬럼: {code_col}")
        print(f"경상북도 사용할 이름 컬럼: {name_col}")
        
        # 컬럼을 찾은 경우에만 진행
        if code_col and name_col:
            # 필요한 컬럼만 유지
            gdf = gdf[['geometry', code_col, name_col]].copy()
            
            # 컬럼명 표준화
            gdf = gdf.rename(columns={code_col: 'SGG_CD', name_col: 'SGG_NM'})
            
            # 행정구역명 정리
            gdf['SGG_NM'] = gdf['SGG_NM'].astype(str).str.strip()
            gdf['SGG_NM'] = gdf['SGG_NM'].str.replace(r'^경상북도\s*', '', regex=True)
            
            print(f"경상북도 정리 전 지역명 샘플: {gdf['SGG_NM'].head().tolist()}")
            
            # 경상북도 시군 필터링
            pattern = '|'.join(REGIONS)
            gdf = gdf[gdf['SGG_NM'].str.contains(pattern, na=False, regex=True)]
            gdf = gdf[~gdf.geometry.is_empty]
            
            print(f"경상북도 필터링 후 지역 수: {len(gdf)}")
            print(f"경상북도 포함된 지역: {sorted(gdf['SGG_NM'].unique())}")

            # 지오메트리 단순화로 메모리 절약
            gdf['geometry'] = gdf['geometry'].simplify(0.01)

            # 포항시 통합 (필요한 경우)
            gdf.loc[gdf['SGG_NM'].str.contains('포항시', na=False), 'SGG_NM'] = '포항시'

            # 중복 통합
            if gdf['SGG_NM'].duplicated().any():
                print("경상북도 중복 지역 통합 중...")
                gdf_list = []
                for region in gdf['SGG_NM'].unique():
                    sub = gdf[gdf['SGG_NM'] == region]
                    geometry = sub.geometry.unary_union if len(sub) > 1 else sub.geometry.iloc[0]
                    gdf_list.append({'SGG_NM': region, 'geometry': geometry})
                gdf = gpd.GeoDataFrame(gdf_list, crs='EPSG:4326')

            # 컬럼명 변경 (기존 코드와 호환성 유지)
            gdf = gdf.rename(columns={'SGG_NM': '행정구역'})
            
            # category 타입으로 최적화
            gdf['행정구역'] = gdf['행정구역'].astype('category')
            
            # 메모리 정리
            gc.collect()
            
            print(f"경상북도 Shapefile 로딩 완료: {len(gdf)}개 시군구")
            return gdf
            
        else:
            print("경상북도 적절한 컬럼을 찾을 수 없습니다.")
            print(f"경상북도 전체 컬럼 목록: {list(gdf.columns)}")
            raise Exception("경상북도 컬럼 매핑 실패")
            
    except Exception as e:
        print(f"경상북도 shapefile 로딩 오류: {e}")
        raise e

try:
    gdf_gyeongbuk = load_and_optimize_shapefile(DATA_DIR / "DA_SIG_202307.shp")
    unique_gyeongbuk_areas = sorted(gdf_gyeongbuk['행정구역'].unique())
    print(f"경상북도 최종 지역 목록: {unique_gyeongbuk_areas}")
except Exception as e:
    print(f"경상북도 shapefile 로딩 오류: {e}")
    print("경상북도 기존 방식으로 대체합니다.")
    gdf_gyeongbuk = gpd.GeoDataFrame()
    unique_gyeongbuk_areas = []

# ====== 영천시 지도 생성 함수 (수정된 버전) ======
def create_yeongcheon_map(selected_marker=None, map_type="normal", locations=None, selected_area=None):
    """영천시 지도 생성 (수정된 버전)"""
    if df_yeongcheon.empty:
        return "<div>지도 데이터를 불러올 수 없습니다.</div>"
    
    # 기본 중심점과 줌 레벨
    center_lat, center_lng = 35.961380, 128.927778
    zoom = 11

    if locations is None:
        locations = pd.DataFrame()

    # 1. 읍면동 선택에 따른 지도 중심/줌 결정 (우선순위 높음)
    if selected_area and selected_area != "전체" and not gdf_yeongcheon.empty:
        # 표준화된 컬럼명 사용
        area_gdf = gdf_yeongcheon[gdf_yeongcheon['ADM_NM'] == selected_area]
        if not area_gdf.empty:
            bounds = area_gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lng = (bounds[0] + bounds[2]) / 2
            zoom = 13
            
    # 2. 저수지가 선택되었고 locations에 해당 저수지가 있을 때만 위치 변경
    elif selected_marker and not locations.empty and selected_marker in locations['시설명'].values:
        selected = locations[locations['시설명'] == selected_marker]
        if not selected.empty:
            center_lat, center_lng = selected.iloc[0]['위도'], selected.iloc[0]['경도']
            zoom = 15

    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=zoom, 
        width="100%", 
        height="100%",
        max_zoom=18,
        zoom_control=False,
        prefer_canvas=True  # 캔버스 렌더링으로 메모리 절약
    )
    
    if map_type == "satellite":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI", 
            name="위성 지도", 
            control=False,
            max_zoom=18
        ).add_to(m)
    else:
        folium.TileLayer(
            "OpenStreetMap", 
            name="일반 지도", 
            control=False,
            max_zoom=19
        ).add_to(m)

    # 현재 지도 상태 유지를 위한 JavaScript 추가
    preserve_state_script = f"""
    <script>
    var selectedReservoir = "{selected_marker if selected_marker else ""}";
    var isReservoirSelected = selectedReservoir !== "" && selectedReservoir !== "None";
    
    setTimeout(function() {{
        var mapElement = document.querySelector('.folium-map');
        if (mapElement) {{
            var leafletMap = window[mapElement.id];
            if (leafletMap) {{
                if (!isReservoirSelected && sessionStorage.getItem('mapZoom') && sessionStorage.getItem('mapCenter')) {{
                    var savedZoom = parseInt(sessionStorage.getItem('mapZoom'));
                    var savedCenter = JSON.parse(sessionStorage.getItem('mapCenter'));
                    leafletMap.setView([savedCenter.lat, savedCenter.lng], savedZoom);
                }} else if (isReservoirSelected) {{
                    setTimeout(function() {{
                        var currentZoom = leafletMap.getZoom();
                        var currentCenter = leafletMap.getCenter();
                        sessionStorage.setItem('mapZoom', currentZoom);
                        sessionStorage.setItem('mapCenter', JSON.stringify({{
                            lat: currentCenter.lat,
                            lng: currentCenter.lng
                        }}));
                    }}, 1000);
                }}
                
                leafletMap.on('zoomend moveend', function() {{
                    setTimeout(function() {{
                        var currentZoom = leafletMap.getZoom();
                        var currentCenter = leafletMap.getCenter();
                        sessionStorage.setItem('mapZoom', currentZoom);
                        sessionStorage.setItem('mapCenter', JSON.stringify({{
                            lat: currentCenter.lat,
                            lng: currentCenter.lng
                        }}));
                    }}, 100);
                }});
            }}
        }}
    }}, 500);
    </script>
    """

    # 행정구역 경계 (표준화된 컬럼명 사용)
    if not gdf_yeongcheon.empty:
        folium.GeoJson(
            gdf_yeongcheon,
            name="영천시 읍면동 경계",
            style_function=lambda x: {
                'fillColor': 'transparent', 
                'color': 'DarkGreen', 
                'weight': 2,
                'fillOpacity': 0,
                'opacity': 0.7
            },
            # 표준화된 컬럼명으로 툴팁 수정
            tooltip=folium.GeoJsonTooltip(
                fields=["ADM_NM"],
                aliases=["행정동:"],
                sticky=True
            )
        ).add_to(m)

    # 선택된 저수지 마커 표시
    if not locations.empty:
        for _, row in locations.iterrows():
            color = 'red' if selected_marker == row['시설명'] else 'blue'
            folium.Marker(
                location=[row['위도'], row['경도']],
                tooltip=row['시설명'],
                icon=folium.Icon(
                    icon='paw',    # Font-Awesome paw 아이콘
                    prefix='fa',   # Font-Awesome 사용을 알리는 접두사
                    color=color    # 기존 색상(red/blue) 유지
                )
            ).add_to(m)


    # 모든 저수지 점 표시
    for _, row in df_yeongcheon.iterrows():
        if locations.empty or row['시설명'] not in locations['시설명'].values:
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=3, 
                color='#1E90FF', 
                fill=True, 
                fill_color='#1E90FF', 
                fill_opacity=0.6,
                tooltip=f"{row['시설명']} ({row['행정동명']})"
            ).add_to(m)

    # JavaScript 추가
    m.get_root().html.add_child(folium.Element(preserve_state_script))

    return m._repr_html_()

# ====== [8] 경상북도 분석 함수들 ======
def analyze_air_pollution_data(file_path: str) -> pd.DataFrame:
    """대기오염 데이터 분석"""
    pollutants = {
        'PM2.5': '미세먼지_PM2.5__월별_도시별_대기오염도',
        'PM10': '미세먼지_PM10__월별_도시별_대기오염도',
        'O3': '오존_월별_도시별_대기오염도',
        'CO': '일산화탄소_월별_도시별_대기오염도',
        'NO2': '이산화질소_월별_도시별_대기오염도'
    }
    
    result_dfs = []
    for pollutant, sheet_name in pollutants.items():
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            gyeongbuk_df = df[df['구분(1)'] == '경상북도']
            month_cols = [col for col in df.columns if str(col).replace('.', '').isdigit()]
            avg_df = gyeongbuk_df.groupby('구분(2)')[month_cols].mean().mean(axis=1).reset_index()
            avg_df.columns = ['시군구', f'{pollutant}_평균']
            avg_df = optimize_dataframe_memory(avg_df)
            result_dfs.append(avg_df)
            
            # 메모리 정리
            del df, gyeongbuk_df
            gc.collect()
            
        except Exception as e:
            pass
    
    if result_dfs:
        result_df = result_dfs[0]
        for df in result_dfs[1:]:
            result_df = pd.merge(result_df, df, on='시군구', how='outer')
        return optimize_dataframe_memory(result_df)
    
    return pd.DataFrame()

def analyze_crime_rate(crime_file_path, population_file_path):
    """범죄율 데이터 분석"""
    crime_df = pd.read_excel(crime_file_path)
    region_columns = [col for col in crime_df.columns if col not in ['범죄대분류', '범죄중분류']]
    total_crimes = crime_df[region_columns].sum().reset_index()
    total_crimes.columns = ['시군구', '총범죄건수']

    pop_raw = pd.read_excel(population_file_path, sheet_name="1-2. 읍면동별 인구 및 세대현황", header=[3,4])
    pop_df = pop_raw[[("구분","Unnamed: 0_level_1"),("총계","총   계")]].copy()
    pop_df.columns = ["region", "population"]
    pop_df = unify_and_filter_region(pop_df, "region")

    crime_data = total_crimes.copy()
    crime_data['region'] = crime_data['시군구']

    merged = pd.merge(crime_data[['region', '총범죄건수']], pop_df, on="region", how="inner")
    merged["범죄율"] = merged["총범죄건수"] / merged["population"]
    merged = merged.sort_values("범죄율", ascending=False)
    
    # 메모리 정리
    del crime_df, pop_raw, pop_df, crime_data
    gc.collect()
    
    return optimize_dataframe_memory(merged)

def analyze_accident_data(excel_path: str) -> pd.DataFrame:
    """교통사고 데이터 분석"""
    df = pd.read_excel(excel_path)
    df = df.loc[df['구분'] == '사고']
    df = df.drop(columns=['연도', '구분']).mean()

    mapping_dict = {
        '포항북부': '포항시', '포항남부': '포항시', '경주': '경주시', '김천': '김천시',
        '안동': '안동시', '구미': '구미시', '영주': '영주시', '영천': '영천시', '상주': '상주시',
        '문경': '문경시', '경산': '경산시', '의성': '의성군', '청송': '청송군', '영양': '영양군',
        '영덕': '영덕군', '청도': '청도군', '고령': '고령군', '성주': '성주군', '칠곡': '칠곡군',
        '예천': '예천군', '봉화': '봉화군', '울진': '울진군', '울릉': '울릉군'
    }

    city_accident_avg = defaultdict(list)
    for region, value in df.items():
        std_city = mapping_dict.get(region)
        if std_city:
            city_accident_avg[std_city].append(value)

    city_accident_avg = {city: sum(values) / len(values) for city, values in city_accident_avg.items()}
    acc_df = pd.DataFrame({'시군': list(city_accident_avg.keys()), '평균사고건수': list(city_accident_avg.values())})
    merged_df = pd.merge(acc_df, pop_df, on='시군')
    merged_df['사고비율'] = merged_df['평균사고건수'] / merged_df['인구수']
    merged_df = merged_df.sort_values("사고비율", ascending=False)
    
    return optimize_dataframe_memory(merged_df)

def analyze_park_area(excel_path: str) -> pd.DataFrame:
    """공원면적 데이터 분석"""
    df = pd.read_excel(excel_path)
    df_subset = df.iloc[3:, [1, 3]]
    df_subset.columns = ['시군', '면적']
    merged_df = pd.merge(df_subset, pop_df, on='시군')
    merged_df['면적'] = pd.to_numeric(merged_df['면적'], errors='coerce')
    merged_df['인구수'] = pd.to_numeric(merged_df['인구수'], errors='coerce')
    merged_df['공원면적비율'] = merged_df['면적'] / merged_df['인구수']
    merged_df = merged_df.sort_values("공원면적비율", ascending=True)
    
    return optimize_dataframe_memory(merged_df)

def analyze_population_facility_ratio(facility_file_path: str, population_file_path: str) -> pd.DataFrame:
    """반려동물 시설 데이터 분석"""
    facility_df = pd.read_csv(facility_file_path, encoding="cp949")
    facility_df = unify_and_filter_region(facility_df, "시도 명칭", "시군구 명칭")

    pop_raw = pd.read_excel(population_file_path, sheet_name="1-2. 읍면동별 인구 및 세대현황", header=[3, 4])
    pop_df_local = pop_raw[[("구분", "Unnamed: 0_level_1"), ("총계", "총   계")]].copy()
    pop_df_local.columns = ["region", "population"]
    pop_df_local = unify_and_filter_region(pop_df_local, "region")
    pop_df_local = pop_df_local[pop_df_local["region"].isin(facility_df["region"].unique())]
    
    df_fac_cnt = facility_df.groupby("region").size().reset_index(name="facility_count")
    df_merge = pd.merge(df_fac_cnt, pop_df_local, on="region")
    df_merge["per_person"] = df_merge["facility_count"] / df_merge["population"]
    df_merge = df_merge.sort_values("per_person", ascending=True)
    
    # 메모리 정리
    del facility_df, pop_raw, pop_df_local, df_fac_cnt
    gc.collect()
    
    return optimize_dataframe_memory(df_merge)


# ====== [9] 레이더 차트 함수 (배경색과 격자 수정 + hovertext/hovertemplate) ======
def plot_radar_chart(park_fp, acc_fp, facility_fp, pop_fp, crime_fp, pollution_fp, selected_regions=None):
    """종합 레이더 차트 (배경색과 격자 수정 + hovertext/hovertemplate 적용)"""
    try:
        # ───────────────────────────────────────────────────────────────────────────
        # 1) 공원면적 데이터
        df_park = pd.read_excel(park_fp).iloc[3:, [1, 3]]
        df_park.columns = ['시군', '면적']
        df_park['면적'] = pd.to_numeric(df_park['면적'], errors='coerce')
        df_park = df_park.merge(pop_df, on='시군')
        df_park['per_person'] = df_park['면적'] / df_park['인구수']
        df_park['park_norm'] = (df_park['per_person'] / df_park['per_person'].max()).astype(np.float32)

        # ───────────────────────────────────────────────────────────────────────────
        # 2) 교통사고 분석
        df_acc = pd.read_excel(acc_fp)
        df_acc = df_acc[df_acc['구분'] == '사고'].drop(columns=['연도', '구분'])
        acc_mean = df_acc.mean()
        mapping_acc = {
            '포항북부':'포항시','포항남부':'포항시','경주':'경주시','김천':'김천시','안동':'안동시','구미':'구미시',
            '영주':'영주시','영천':'영천시','상주':'상주시','문경':'문경시','경산':'경산시','의성':'의성군',
            '청송':'청송군','영양':'영양군','영덕':'영덕군','청도':'청도군','고령':'고령군','성주':'성주군',
            '칠곡':'칠곡군','예천':'예천군','봉화':'봉화군','울진':'울진군','울릉':'울릉군'
        }
        acc_dict = {}
        for k, v in acc_mean.items():
            city = mapping_acc.get(k)
            if city:
                acc_dict.setdefault(city, []).append(v)
        acc_avg = {city: np.mean(vals) for city, vals in acc_dict.items()}
        df_acc2 = pd.DataFrame.from_dict(acc_avg, orient='index', columns=['acc']).reset_index().rename(columns={'index': '시군'})
        df_acc2 = df_acc2.merge(pop_df, on='시군')
        df_acc2['acc_inv'] = 1 / (df_acc2['acc'] / df_acc2['인구수'])
        df_acc2['acc_norm'] = (df_acc2['acc_inv'] / df_acc2['acc_inv'].max()).astype(np.float32)

        # ───────────────────────────────────────────────────────────────────────────
        # 3) 반려동물 시설 분석
        df_fac = pd.read_csv(facility_fp, encoding='cp949')
        df_fac = df_fac[df_fac['시도 명칭'] == '경상북도']
        df_fac['시군'] = df_fac['시군구 명칭'].str.extract(r'^(.*?[시군])')[0]
        df_fac = df_fac[df_fac['시군'] != '군위군']
        fac_counts = df_fac['시군'].value_counts().rename_axis('시군').reset_index(name='facility_count')
        fac_df = fac_counts.merge(pop_df, on='시군')
        fac_df['per_person'] = fac_df['facility_count'] / fac_df['인구수']
        fac_df['fac_norm'] = (fac_df['per_person'] / fac_df['per_person'].max()).astype(np.float32)

        # ───────────────────────────────────────────────────────────────────────────
        # 4) 범죄율 분석
        crime_df = pd.read_excel(crime_fp)
        cols = [c for c in crime_df.columns if c not in ['범죄대분류', '범죄중분류']]
        crime_tot = crime_df[cols].sum().reset_index()
        crime_tot.columns = ['raw', 'crime']
        crime_tot['시군'] = crime_tot['raw'].str.split().str[0]
        crime_tot = crime_tot[crime_tot['시군'] != '군위군'][['시군', 'crime']]
        crime_tot = crime_tot.merge(pop_df, on='시군')
        crime_tot['crime_inv'] = 1 / (crime_tot['crime'] / crime_tot['인구수'])
        crime_tot['crime_norm'] = (crime_tot['crime_inv'] / crime_tot['crime_inv'].max()).astype(np.float32)

        # ───────────────────────────────────────────────────────────────────────────
        # 5) 대기오염 분석
        pollutants = {
            'PM2.5': '미세먼지_PM2.5__월별_도시별_대기오염도',
            'PM10': '미세먼지_PM10__월별_도시별_대기오염도',
            'O3': '오존_월별_도시별_대기오염도',
            'CO': '일산화탄소_월별_도시별_대기오염도',
            'NO2': '이산화질소_월별_도시별_대기오염도'
        }
        polls = []
        for pol, sheet in pollutants.items():
            try:
                dfp = pd.read_excel(pollution_fp, sheet_name=sheet)
                dfp = dfp[dfp['구분(1)'] == '경상북도']
                mcols = [c for c in dfp.columns if c not in ['구분(1)', '구분(2)']]
                dfp[mcols] = dfp[mcols].apply(pd.to_numeric, errors='coerce')
                avg = dfp.groupby('구분(2)')[mcols].mean().mean(axis=1).rename(pol)
                polls.append(avg)
            except:
                pass
        if polls:
            poll_df = pd.concat(polls, axis=1).reset_index().rename(columns={'구분(2)': '시군'})
            poll_df['시군'] = poll_df['시군'].astype(str).apply(lambda x: x + '시' if not x.endswith(('시', '군')) else x)
            for pol in pollutants:
                if pol in poll_df.columns:
                    poll_df[f'{pol}_n'] = (poll_df[pol] / poll_df[pol].max()).astype(np.float32)
            poll_cols = [f'{p}_n' for p in pollutants if f'{p}_n' in poll_df.columns]
            poll_df['poll_comp'] = poll_df[poll_cols].sum(axis=1)
            poll_df['poll_inv'] = 1 / poll_df['poll_comp']
            poll_df['poll_norm'] = (poll_df['poll_inv'] / poll_df['poll_inv'].max()).astype(np.float32)
        else:
            poll_df = pd.DataFrame({'시군': REGIONS, 'poll_norm': [0.5] * len(REGIONS)})

        # ───────────────────────────────────────────────────────────────────────────
        # 6) 데이터 통합
        metrics = pd.DataFrame({'시군': REGIONS})
        metrics = metrics.merge(df_park[['시군', 'park_norm']], on='시군', how='left')
        metrics = metrics.merge(df_acc2[['시군', 'acc_norm']], on='시군', how='left')
        metrics = metrics.merge(fac_df[['시군', 'fac_norm']], on='시군', how='left')
        metrics = metrics.merge(crime_tot[['시군', 'crime_norm']], on='시군', how='left')
        metrics = metrics.merge(poll_df[['시군', 'poll_norm']], on='시군', how='left')
        metrics = metrics.fillna(0.5)
        metrics = optimize_dataframe_memory(metrics)

        # ───────────────────────────────────────────────────────────────────────────
        # 7) 레이더 차트 그리기
        categories = ['산책 환경', '반려동물 시설', '교통 안전', '치안', '대기 환경']
        theta = categories + [categories[0]]
        fig = go.Figure()

        for _, row in metrics.iterrows():
            values = [
                row['park_norm'], row['fac_norm'], row['acc_norm'],
                row['crime_norm'], row['poll_norm']
            ] + [row['park_norm']]
            
            # 시군명을 point 하나하나에 반복해서 담아 줍니다
            cd = [row['시군']] * len(values)

            is_sel = selected_regions and row['시군'] in selected_regions
            col = REGION_COLORS.get(row['시군'], '#808080')
            width = 3 if is_sel else 1
            opacity = 1.0 if is_sel else 0.2
            showleg = is_sel
            if is_sel:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=theta,
                    name=row['시군'],
                    customdata=cd,
                    hovertemplate=(
                        "지역: %{customdata}<br>"
                        "지표: %{theta}<br>"
                        "점수: %{r:.2f}<extra></extra>"
                    ),
                    line=dict(width=width, color=col),
                    opacity=opacity,
                    showlegend=True
                ))
            else:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=theta,
                    name=row['시군'],
                    hoverinfo='skip',      # ◀ 여기를 추가하면 툴팁이 뜨지 않습니다
                    line=dict(width=width, color='lightgray'),
                    opacity=opacity,
                    showlegend=False
                ))


        # ───────────────────────────────────────────────────────────────────────────
        # 8) 스타일 수정
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.2],
                    side="clockwise",
                    angle=90,
                    gridcolor='lightgray',
                    showline=True,
                    linecolor='lightgray',
                    tick0=0,
                    dtick=0.2,
                    showticklabels=True
                ),
                angularaxis=dict(
                    rotation=90,
                    direction="clockwise",
                    gridcolor='lightgray'
                ),
                bgcolor='white'
            ),
            showlegend=True,
            legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
            width=520,
            height=500,
            margin=dict(t=20, b=0, l=0, r=0),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # ───────────────────────────────────────────────────────────────────────────
        # 9) 메모리 정리 & 반환
        del df_park, df_acc, df_acc2, df_fac, fac_df, crime_df, crime_tot, poll_df, metrics
        gc.collect()
        return fig

    except Exception as e:
        print(f"레이더 차트 생성 오류: {e}")
        return go.Figure()


# ====== [10] 지도 생성 함수들 ======
def create_gyeongbuk_map(selected_regions=None):
    """경상북도 지도 생성"""
    if gdf_gyeongbuk.empty:
        return "<div>지도 데이터를 불러올 수 없습니다.</div>"
    
    bounds = gdf_gyeongbuk.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
        prefer_canvas=True,
        zoom_control=False,
        max_zoom=12  # 최대 줌 제한으로 메모리 절약
    )

    def style_function(feature):
        name = feature["properties"]["행정구역"]
        if selected_regions and name in selected_regions:
            col = REGION_COLORS.get(name, "#FF0000")
            return {
                "fillColor": col, "color": "#2e7d32", "weight": 3,
                "fillOpacity": 0.5, "opacity": 1.0
            }
        else:
            return {
                "fillColor": "#f5f5f5", "color": "#4caf50", "weight": 2,
                "fillOpacity": 0.5, "opacity": 0.8
            }

    folium.GeoJson(
        gdf_gyeongbuk,
        name="경북 행정구역",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["행정구역"],
            aliases=["지역:"],
            sticky=True
        )
    ).add_to(m)

    for idx, row in gdf_gyeongbuk.iterrows():
        centroid = row.geometry.centroid
        region_name = row["행정구역"]
        is_selected = selected_regions and region_name in selected_regions
        text_color = "#ffffff" if is_selected else "#333333"
        font_weight = "bold" if is_selected else "normal"
        font_size = "14px" if is_selected else "12px"
        
        folium.Marker(
            location=[centroid.y, centroid.x],
            icon=folium.DivIcon(
                html=f"""<div style="
                    font-family: '맑은 고딕', sans-serif;
                    font-size: {font_size};
                    font-weight: {font_weight};
                    color: {text_color};
                    text-align: center;
                    white-space: nowrap;
                    pointer-events: none;
                ">{region_name}</div>""",
                icon_size=(100, 20),
                icon_anchor=(50, 10)
            )
        ).add_to(m)
    
    return m._repr_html_()



def create_barplot(data):
    """영천시 적합도 바 차트"""
    # 데이터 정렬
    df_sorted = data.sort_values(by='적합도점수', ascending=True)
    
    # Bar 차트 생성
    fig = px.bar(
        df_sorted,
        x='적합도점수',
        y='시설명',
        orientation='h',
        title=f'상위 {len(data)}개 저수지 개발 적합도 점수',
        height=300,
        width=320,
        color_discrete_sequence=['#1e3a8a']
    )
    
    # 툴팁 포맷팅: 시설명(%{y}), 점수 소수점 둘째자리까지
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>점수: %{x:.2f}<extra></extra>'
    )
    
    # 레이아웃 및 hoverlabel 스타일
    fig.update_layout(
        xaxis=dict(
            range=[0, 1.2],
            showgrid=False,
            tickvals=[0, 0.5, 1.0],
            ticktext=['0', '0.5', '1']
        ),
        yaxis_title=None,
        margin=dict(l=0, r=10, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10),
        hoverlabel=dict(
            bgcolor='#EEEEEE',    # 툴팁 배경
            font_color='#333333', # 글자색
            bordercolor='#AAAAAA' # 테두리색
        )
    )
    
    # Shiny UI로 반환
    return ui.HTML(
        fig.to_html(
            full_html=False,
            include_plotlyjs='embed',
            config={'displayModeBar': False}
        )
    )



# ====== [11] CSS 스타일 (헤더와 레이더 차트 간격 수정) ======
custom_css = """
/* ─ 배경 이미지 + 어두운 오버레이 ─ */
.welcome-background {
  /* 1) 배경 이미지는 그대로 */
  background-image: url('bg.png');
  background-size: cover;
  background-position: center;

  /* 2) 반투명 흰색 오버레이를 위에 쌓기 */
  background-repeat: no-repeat;
  background-blend-mode: overlay;
  background-color: rgba(255,255,255,0.8);

  /* 3) 기존 위치/크기 설정 유지 */
  position: fixed;
  top: 60px;             /* 헤더 높이만큼 아래로 */
  left: 0;
  width: 100%;
  height: calc(100vh - 60px);
  z-index: -1;
}

/* ─ 중앙 콘텐츠 박스 ─ */
.welcome-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: calc(100vh - 60px);
  text-align: center;
  padding: 0 20px;
}

/* ─ 제목/부제목 ─ */
.welcome-title {
  font-size: 2.5rem;
    font-weight: bold;
  color: #000;                
  text-shadow: none;          
  margin-top: -100px;    
  margin-bottom: 0.5rem;
}
.welcome-subtitle {
  font-size: 1.25rem;
  font-weight: 500;
  color: #000;                
  margin-top: 10px;   
  margin-bottom: 2rem;
}


/* ─ 시작 버튼 ─ */
.start-button {
  font-size: 1.125rem;
  padding: 0.75rem 2rem;
  background-color: #1e3a8a;
  color: white;
  border: none;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s ease;
}
.start-button:hover {
  background-color: #002c5f;
  transform: translateY(-2px);
}

/* 헤더 스타일 */
#header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #1e3a8a;
    color: white;
    padding: 10px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* 헤더 로고 이미지 */
#header img {
  height: 30px;
  margin-right: 15px;
}

.tab-container {
    display: flex;
    gap: 10px;
}

.tab-button {
    background-color: rgba(255,255,255,0.2);
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 8px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 500;
}

.tab-button:hover {
    background-color: rgba(255,255,255,0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.tab-button.active {
    background-color: rgba(255,255,255,0.9);
    color: #667eea;
    font-weight: bold;
}



/* 경상북도용 사이드바 */
#gyeongbuk-sidebar {
    position: fixed;
    top: 60px;
    left: 0;
    width: 300px;
    height: calc(100vh - 60px);
    background-color: white;
    box-shadow: 2px 0 8px rgba(0,0,0,0.2);
    transition: transform 0.3s ease-in-out;
    z-index: 1000;
    padding: 20px;
    transform: translateX(0);
}

#gyeongbuk-sidebar.open {
    transform: translateX(0);
}

#gyeongbuk-toggle-button {
    position: fixed;
    top: 80px;
    left: 300px;
    transform: translateX(-5px);
    z-index: 9999;
    background-color: white;
    border: 1px solid #ccc;
    width: 30px;
    height: 30px;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 6px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: left 0.3s ease-in-out;
}

/* 영천시용 사이드바 */
#yeongcheon-sidebar {
    position: fixed;
    top: 60px;
    left: 0;
    width: 300px;
    height: calc(100vh - 60px);
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    z-index: 10000;
    transition: transform 0.3s ease-in-out;
    transform: translateX(0);
}

#yeongcheon-toggle-button {
    position: fixed;
    top: 80px;
    left: 300px;
    transform: translateX(-5px);
    z-index: 10001;
    background-color: white;
    border: 1px solid #ccc;
    width: 30px;
    height: 30px;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 6px rgba(0,0,0,0.2);
    cursor: pointer;
    transition: left 0.3s ease-in-out;
}


.sidebar-section {
    margin-bottom: 20px;
    border-bottom: 1px solid #eee;
    padding-bottom: 15px;
}

.close-btn {
    position: absolute;
    top: 10px;
    right: 15px;
    border: none;
    background: none;
    font-size: 18px;
    cursor: pointer;
    color: #999;
}

.close-btn:hover {
    color: #333;
}


/* ─────────────────────────────────
   공통 사이드바 디자인
───────────────────────────────── */
#gyeongbuk-sidebar,
#yeongcheon-sidebar {
  position: fixed;
  top: 60px;                            /* 헤더 바로 아래 */
  left: 0;
  width: 320px;                        
  height: calc(100vh - 60px);
  background: #fff;                     /* 깨끗한 흰색 배경 */
  border-right: 1px solid #e0e0e0;      /* 부드러운 테두리 */
  box-shadow: 2px 0 8px rgba(0,0,0,0.1); /* 조금 가벼운 그림자 */
  border-top-right-radius: 8px;         /* 우측 모서리 둥글게 */
  border-bottom-right-radius: 8px;
  padding: 16px;                        /* 내부 여백 */
  overflow-y: auto;
  transition: transform 0.3s ease;
  transform: translateX(0);             /* JS에서 translateX(-260px)로 닫고, 0으로 열립니다 */
}

/* ─────────────────────────────────
   토글 버튼 디자인
───────────────────────────────── */
#gyeongbuk-toggle-button,
#yeongcheon-toggle-button {
  position: fixed;
  top: calc(60px + 16px);              /* 헤더 높이 + 사이드바 패딩 */
  left: 320px;                          /* 사이드바의 오른쪽 끝에 딱 붙여놓습니다 */
  width: 32px;
  height: 32px;
  background-color: #1e88e5;            /* 포인트 블루 */
  color: #fff;
  border: none;
  border-top-left-radius: 4px;          /* 버튼 왼쪽 모서리만 둥글게 */
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: left 0.3s ease, background-color 0.2s ease;
  z-index: 1100;
}
#gyeongbuk-toggle-button:hover,
#yeongcheon-toggle-button:hover {
  background-color: #1565c0;           /* hover 땐 좀 더 진한 블루로 */
}

/* ─────────────────────────────────
   토글 버튼: 오른쪽만 둥글게
───────────────────────────────── */
#gyeongbuk-toggle-button,
#yeongcheon-toggle-button {
  position: fixed;
  top: calc(60px + 16px);
  left: 320px;
  width: 40px;
  height: 60px;
  background-color: #ffffff;
  color: #333333;
  border: none;
  /* top-left, top-right, bottom-right, bottom-left */
  border-radius: 0 20px 20px 0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: left 0.3s ease, background-color 0.2s ease;
  z-index: 900;
}
#gyeongbuk-toggle-button:hover,
#yeongcheon-toggle-button:hover {
  background-color: #f5f5f5;
}




/* 부록 페이지 스타일 */
#appendix-content {
    padding: 40px;
    background-color: #f8f9fa;
    min-height: calc(100vh - 60px);
    max-width: 1200px;
    margin: 0 auto;
    font-family: '맑은 고딕', sans-serif;
}

.appendix-section {
    background-color: white;
    margin-bottom: 30px;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.appendix-section h2 {
    color: #1e3a8a;
    border-bottom: 3px solid #1e3a8a;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

.appendix-section h3 {
    color: #2563eb;
    margin-top: 25px;
    margin-bottom: 15px;
}

.appendix-section h4 {
    color: #3b82f6;
    margin-top: 20px;
    margin-bottom: 10px;
}

.formula-box {
    background-color: #f1f5f9;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    font-family: 'Courier New', monospace;
    text-align: center;
}

.highlight-box {
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 15px;
    margin: 15px 0;
}

.data-source {
    background-color: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    padding: 10px;
    margin: 10px 0;
    font-size: 0.9em;
}

.method-step {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
}



:root {
  --bslib-sidebar-main-bg: #f3f3f3;
}

body {
  font-family: '맑은 고딕', sans-serif;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: auto;
}

h4 {
  margin-top: 0;
  font-weight: bold;
}

/* 1) nav-tabs 자체의 밑줄 제거 */
.modern-tabs .nav-tabs {
  border-bottom: none !important;
}

/* 2) 각 탭 버튼 스타일 (기존에 쓰시던 대로) */
.modern-tabs .nav-tabs .nav-link {
    border: none !important;
    border-radius: 20px !important;
    margin-right: 8px !important;
    background-color: #f8f9fa !important;
    color: #6c757d !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}


.modern-tabs .nav-tabs .nav-link.active {
    background-color: #007bff !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(0,123,255,0.3) !important;
    /* 필요하다면 여기도 border:none 지정 가능 */
    border: none !important;
}

.modern-tabs .nav-tabs {
  margin-bottom: 40px !important;
}


"""

# # ====== [12] UI 구성 ======
# app_ui = ui.page_fluid(
#     ui.tags.style(custom_css),
    
#     # 헤더
#     ui.div(
#         ui.div(
#             # 좌측: 로고 + 제목
#             ui.div(
#                 tags.img(src="logo.png", height="28px", style="margin-right: 10px;"),
#                 ui.h3("영천시에서 함께살개냥", style="margin: 0; font-size: 18px; color: white;"),
#                 style="display: flex; align-items: center;"
#             ),
#             # 우측: 탭 버튼
#             ui.div(
#                 ui.HTML("""
#                     <div class="tab-container">
#                         <button class="tab-button active" id="tab-gyeongbuk"
#                                 onclick="setActiveTab('경상북도')">경상북도</button>
#                         <button class="tab-button" id="tab-yeongcheon"
#                                 onclick="setActiveTab('영천시')">영천시</button>
#                     </div>
#                 """),
#                 ui.tags.script("""
#                     function setActiveTab(tabName) {
#                         Shiny.setInputValue('top_tab', tabName);
#                         document.querySelectorAll('.tab-button').forEach(btn => {
#                             btn.classList.remove('active');
#                         });
#                         if (tabName === '경상북도') {
#                             document.getElementById('tab-gyeongbuk').classList.add('active');
#                         } else {
#                             document.getElementById('tab-yeongcheon').classList.add('active');
#                         }
#                     }
                    
#                     // 초기화 및 엔터키/체크박스 토글 기능
#                     document.addEventListener("DOMContentLoaded", function() {
#                         // 기본값을 경상북도로 설정
#                         setTimeout(function() {
#                             setActiveTab('경상북도');
#                         }, 100);
                        
#                         // 경상북도 관련 변수들
#                         let isFirstTimeGyeongbuk = true;
#                         let isFirstDetailsTime = true;
                        
#                         // 체크박스 변경 시뮬레이션 함수 (경상북도용)
#                         function simulateCheckboxChangeGyeongbuk() {
#                             const firstChecked = document.querySelector("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
#                             if (firstChecked) {
#                                 firstChecked.click(); // 해제
#                                 setTimeout(function() {
#                                     firstChecked.click(); // 다시 체크
#                                     setTimeout(function() {
#                                         const applyBtn = document.getElementById('gyeongbuk_apply_selection');
#                                         if (applyBtn) applyBtn.click(); // 분석 버튼 클릭
#                                     }, 100);
#                                 }, 100);
#                             }
#                         }
                        
#                         // 경상북도 분석 버튼 클릭 이벤트 (조건부 렌더링으로 변경)
#                         document.addEventListener('click', function(event) {
#                             if (event.target && event.target.id === 'gyeongbuk_apply_selection') {
#                                 const checkedBoxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
#                                 if (checkedBoxes.length > 0) {
#                                     // 최초 메인 창이 열릴 때만 체크박스 변경 시뮬레이션
#                                     if (isFirstTimeGyeongbuk) {
#                                         isFirstTimeGyeongbuk = false;
#                                         setTimeout(function() {
#                                             simulateCheckboxChangeGyeongbuk();
#                                         }, 700);
#                                     }
#                                 } else {
#                                     alert('분석할 지역을 먼저 선택해주세요.');
#                                 }
#                             }
                            
#                             // 지표 상세 버튼 클릭 처리
#                             if (event.target && event.target.textContent === '자세히 보기') {
#                                 const container = document.getElementById('gyeongbuk-details-container');
#                                 if (container && isFirstDetailsTime) {
#                                     isFirstDetailsTime = false;
#                                     setTimeout(function() {
#                                         simulateCheckboxChangeGyeongbuk();
#                                     }, 500);
#                                 }
#                             }
#                         });
                        
#                         // 엔터키 이벤트 (전체)
#                         document.addEventListener('keydown', function(event) {
#                             if (event.key === 'Enter' || event.keyCode === 13) {
#                                 const currentTab = document.querySelector('.tab-button.active');
#                                 if (currentTab && currentTab.id === 'tab-gyeongbuk') {
#                                     // 경상북도 탭에서 엔터키
#                                     const sidebar = document.getElementById('gyeongbuk-sidebar');
#                                     if (sidebar && sidebar.classList.contains('open')) {
#                                         const checkedBoxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
#                                         if (checkedBoxes.length > 0) {
#                                             const applyBtn = document.getElementById('gyeongbuk_apply_selection');
#                                             if (applyBtn) applyBtn.click();
#                                         }
#                                     }
#                                 } else if (currentTab && currentTab.id === 'tab-yeongcheon') {
#                                     // 영천시 탭에서 엔터키
#                                     const sidebar = document.getElementById('yeongcheon-sidebar');
#                                     const activeElement = document.activeElement;
#                                     if (sidebar && (sidebar.contains(activeElement) || activeElement.tagName === 'INPUT' || activeElement.tagName === 'SELECT')) {
#                                         // 숫자 입력 필드의 경우 blur 이벤트를 강제로 발생시켜 값 업데이트
#                                         if (activeElement.tagName === 'INPUT' && activeElement.type === 'number') {
#                                             activeElement.blur();
#                                             activeElement.focus();
#                                         }
                                        
#                                         // 약간의 지연 후 Shiny 신호 전송
#                                         setTimeout(function() {
#                                             Shiny.setInputValue('yeongcheon_enter_key_pressed', Math.random(), {priority: 'event'});
#                                         }, 50);
                                        
#                                         event.preventDefault();
#                                     }
#                                 }
#                             }
#                         });
#                     });
#                 """),
#                 style="margin-left: auto;"
#             ),
#             id="header"
#         ),
#         style="""
#             position: fixed;
#             top: 0;
#             left: 0;
#             width: 100%;
#             height: 60px;
#             z-index: 9999;
#         """
#     ),
#     ui.output_ui("main_content", style="padding-top: 60px;"),
    
# )



# ====== [12] UI 구성 ======
app_ui = ui.page_fluid(
    ui.tags.style(custom_css),
    
    # 헤더
    ui.div(
        ui.div(
            # 좌측: 로고 + 제목
            ui.div(
                tags.img(src="logo.png", height="28px", style="margin-right: 10px;"),
                ui.h3("영천에서 함께살개냥", style="margin: 0; font-size: 18px; color: white;"),
                style="display: flex; align-items: center;"
            ),
            # 우측: 탭 버튼 (부록 탭 추가)
            ui.div(
                ui.HTML("""
                    <div class="tab-container">
                        <button class="tab-button active" id="tab-gyeongbuk"
                                onclick="setActiveTab('경상북도')">경상북도</button>
                        <button class="tab-button" id="tab-yeongcheon"
                                onclick="setActiveTab('영천시')">영천시</button>
                        <button class="tab-button" id="tab-appendix"
                                onclick="setActiveTab('부록')">부록</button>
                    </div>
                """),
                ui.tags.script("""
                    function setActiveTab(tabName) {
                        Shiny.setInputValue('top_tab', tabName);
                        document.querySelectorAll('.tab-button').forEach(btn => {
                            btn.classList.remove('active');
                        });
                        if (tabName === '경상북도') {
                            document.getElementById('tab-gyeongbuk').classList.add('active');
                        } else if (tabName === '영천시') {
                            document.getElementById('tab-yeongcheon').classList.add('active');
                        } else if (tabName === '부록') {
                            document.getElementById('tab-appendix').classList.add('active');
                        }
                    }
                    
                    // 초기화 및 엔터키/체크박스 토글 기능
                    document.addEventListener("DOMContentLoaded", function() {
                        // 기본값을 경상북도로 설정
                        setTimeout(function() {
                            setActiveTab('경상북도');
                        }, 100);
                        
                        // 경상북도 관련 변수들
                        let isFirstTimeGyeongbuk = true;
                        let isFirstDetailsTime = true;
                        
                        // 체크박스 변경 시뮬레이션 함수 (경상북도용)
                        function simulateCheckboxChangeGyeongbuk() {
                            const firstChecked = document.querySelector("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
                            if (firstChecked) {
                                firstChecked.click(); // 해제
                                setTimeout(function() {
                                    firstChecked.click(); // 다시 체크
                                    setTimeout(function() {
                                        const applyBtn = document.getElementById('gyeongbuk_apply_selection');
                                        if (applyBtn) applyBtn.click(); // 분석 버튼 클릭
                                    }, 100);
                                }, 100);
                            }
                        }
                        
                        // 경상북도 분석 버튼 클릭 이벤트 (조건부 렌더링으로 변경)
                        document.addEventListener('click', function(event) {
                            if (event.target && event.target.id === 'gyeongbuk_apply_selection') {
                                const checkedBoxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
                                if (checkedBoxes.length > 0) {
                                    // 최초 메인 창이 열릴 때만 체크박스 변경 시뮬레이션
                                    if (isFirstTimeGyeongbuk) {
                                        isFirstTimeGyeongbuk = false;
                                        setTimeout(function() {
                                            simulateCheckboxChangeGyeongbuk();
                                        }, 700);
                                    }
                                } else {
                                    alert('분석할 지역을 먼저 선택해주세요.');
                                }
                            }
                            
                            // 지표 상세 버튼 클릭 처리
                            if (event.target && event.target.textContent === '자세히 보기') {
                                const container = document.getElementById('gyeongbuk-details-container');
                                if (container && isFirstDetailsTime) {
                                    isFirstDetailsTime = false;
                                    setTimeout(function() {
                                        simulateCheckboxChangeGyeongbuk();
                                    }, 500);
                                }
                            }
                        });
                        
                        // 엔터키 이벤트 (전체)
                        document.addEventListener('keydown', function(event) {
                            if (event.key === 'Enter' || event.keyCode === 13) {
                                const currentTab = document.querySelector('.tab-button.active');
                                if (currentTab && currentTab.id === 'tab-gyeongbuk') {
                                    // 경상북도 탭에서 엔터키
                                    const sidebar = document.getElementById('gyeongbuk-sidebar');
                                    if (sidebar && sidebar.classList.contains('open')) {
                                        const checkedBoxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']:checked");
                                        if (checkedBoxes.length > 0) {
                                            const applyBtn = document.getElementById('gyeongbuk_apply_selection');
                                            if (applyBtn) applyBtn.click();
                                        }
                                    }
                                } else if (currentTab && currentTab.id === 'tab-yeongcheon') {
                                    // 영천시 탭에서 엔터키
                                    const sidebar = document.getElementById('yeongcheon-sidebar');
                                    const activeElement = document.activeElement;
                                    if (sidebar && (sidebar.contains(activeElement) || activeElement.tagName === 'INPUT' || activeElement.tagName === 'SELECT')) {
                                        // 숫자 입력 필드의 경우 blur 이벤트를 강제로 발생시켜 값 업데이트
                                        if (activeElement.tagName === 'INPUT' && activeElement.type === 'number') {
                                            activeElement.blur();
                                            activeElement.focus();
                                        }
                                        
                                        // 약간의 지연 후 Shiny 신호 전송
                                        setTimeout(function() {
                                            Shiny.setInputValue('yeongcheon_enter_key_pressed', Math.random(), {priority: 'event'});
                                        }, 50);
                                        
                                        event.preventDefault();
                                    }
                                }
                            }
                        });
                    });
                """),
                style="margin-left: auto;"
            ),
            id="header"
        ),
        style="""
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 60px;
            z-index: 9999;
        """
    ),
    ui.output_ui("main_content", style="padding-top: 60px;"),
    
)


# ====== [13] 서버 함수 ======
def server(input, output, session):
    # 메모리 사용량 로깅
    log_memory_usage("서버 시작")
    
    # 캐싱된 데이터 (메모리 효율적)
    _cached_data = {}
    
    def get_cached_or_compute(key, compute_func, *args, **kwargs):
        """캐싱 기능이 있는 계산 함수"""
        if key not in _cached_data:
            _cached_data[key] = compute_func(*args, **kwargs)
            # 캐시 크기 제한
            if len(_cached_data) > 10:
                oldest_key = next(iter(_cached_data))
                del _cached_data[oldest_key]
                gc.collect()
        return _cached_data[key]
    
    app_started = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.start_app)
    def _():
        app_started.set(True)
    
    # 탭 변경 감지
    @reactive.Effect
    @reactive.event(input.top_tab)
    def on_tab_change():
        print(f"선택된 탭: {input.top_tab()}")
        log_memory_usage(f"탭 변경: {input.top_tab()}")

    # 메인 콘텐츠 UI
    @output()
    @render.ui
    def main_content():
        # 1) 시작 전: 웰컴 페이지
        if not app_started():
            # 환영 페이지
            return ui.div(
                ui.tags.div({"class":"welcome-background"}),
                ui.div({"class":"welcome-container"},
                    ui.h1("영천시 반려동물 친화 환경 분석",
                          {"class":"welcome-title"}),
                    ui.p("저수지 데이터 기반 반려동물 친화 환경 개선안 제안",
                          {"class":"welcome-subtitle"}),
                    ui.input_action_button("start_app", "시작하기",
                                           class_="start-button")
                )
            )
        
        # 2) 시작 후: 세 개의 탭을 분기
        if input.top_tab() == "경상북도":
            return gyeongbuk_ui()
        elif input.top_tab() == "영천시":
            return yeongcheon_ui()
        elif input.top_tab() == "부록":
            return appendix_ui()
        else:
            return ui.div("잘못된 탭 선택")

    # 경상북도 UI 함수
    def gyeongbuk_ui():
        return ui.TagList(
            # 지도
            ui.output_ui(
                "gyeongbuk_map",
                style="""
                    position: fixed;
                    top: 60px;
                    left: 0;
                    width: 100vw;
                    height: calc(100vh - 60px);
                    z-index: -1;
                """
            ),

                # 사이드바 토글 버튼
                ui.tags.button(
                    "〈",
                    id="gyeongbuk-toggle-button",
                    onclick="""
                        const sidebar = document.getElementById('gyeongbuk-sidebar');
                        const btn     = document.getElementById('gyeongbuk-toggle-button');
                        const isClosed = sidebar.style.transform === 'translateX(-320px)';
                        sidebar.style.transform = isClosed
                            ? 'translateX(0)'
                            : 'translateX(-320px)';
                        btn.innerText = isClosed ? '〈' : '〉';
                        btn.style.left = isClosed ? '320px' : '16px';
                    """),

                # 사이드바
                ui.div(
                    ui.h3("경상북도 지역 선택", style="margin-bottom: 20px;"),
                    ui.input_checkbox_group(
                        "gyeongbuk_selected_areas", 
                        "", 
                        choices={
                            area: ui.HTML(f"""
                                <span style="display: inline-flex; align-items: center;">
                                    {area}
                                    <span style="
                                        display: inline-block;
                                        width: 12px;
                                        height: 12px;
                                        background-color: {REGION_COLORS.get(area, '#808080')};
                                        border-radius: 50%;
                                        margin-left: 8px;
                                        border: 1px solid #ddd;
                                    "></span>
                                </span>
                            """) for area in unique_gyeongbuk_areas
                        },
                        selected=[]
                    ),
                    
                    ui.div(
                        ui.tags.button(
                            "모두선택",
                            onclick="""
                                const checkboxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']");
                                checkboxes.forEach(checkbox => {
                                    if (!checkbox.checked) {
                                        checkbox.click();
                                    }
                                });
                            """,
                            style="width: 44%; padding: 8px; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-size: 12px; font-weight: bold; cursor: pointer; margin-right: 4%;"
                        ),
                        ui.tags.button(
                            "모두해제",
                            onclick="""
                                const checkboxes = document.querySelectorAll("input[type='checkbox'][name='gyeongbuk_selected_areas']");
                                checkboxes.forEach(checkbox => {
                                    if (checkbox.checked) {
                                        checkbox.click();
                                    }
                                });
                            """,
                            style="width: 44%; padding: 8px; background-color: #f44336; color: white; border: none; border-radius: 4px; font-size: 12px; font-weight: bold; cursor: pointer;"
                        ),
                        style="margin-top: 10px; margin-bottom: 15px; display: flex;"
                    ),
                    
                    ui.input_action_button("gyeongbuk_apply_selection", "선택 지역 분석하기", 
                                        style="width: 92%; margin-top: 10px; padding: 10px;  background-color: #1e3a8a;  color: white; border: none; border-radius: 5px; font-weight: bold;"),
                    id="gyeongbuk-sidebar",
                    class_="open"
                ),

                # 팝업 창들
                ui.output_ui("gyeongbuk_popup"),
                ui.output_ui("gyeongbuk_details")
            )
        
    # 영천시 UI 함수
    def yeongcheon_ui():
        return ui.TagList(
            # 지도
            ui.output_ui(
                "yeongcheon_map",
                style="""
                    position: fixed;
                    top: 60px;
                    left: 0;
                    width: 100vw;
                    height: calc(100vh - 60px);
                    z-index: -1;
                """
            ),
                # 사이드바 토글 버튼
                ui.tags.button(
                    "〈",
                    id="yeongcheon-toggle-button",
                    onclick="""
                        const sidebar = document.getElementById('yeongcheon-sidebar');
                        const btn     = document.getElementById('yeongcheon-toggle-button');
                        const isClosed = sidebar.style.transform === 'translateX(-320px)';
                        sidebar.style.transform = isClosed
                            ? 'translateX(0)'
                            : 'translateX(-320px)';
                        btn.innerText = isClosed ? '〈' : '〉';
                        btn.style.left = isClosed ? '320px' : '16px';
                    """
                ),

                # 사이드바
                ui.div(
                    ui.h3("영천시 저수지 분석", style="margin-bottom: 20px;"),
                    ui.div(
                        ui.h4("저수지 필터링"),
                        ui.input_select("yeongcheon_area", "읍면동 선택", choices=unique_areas, selected="전체"),
                        ui.input_numeric("yeongcheon_top_n", "상위 저수지 개수", value=10, min=1, max=len(df_yeongcheon) if not df_yeongcheon.empty else 10),
                        class_="sidebar-section"
                    ),
                    ui.div(
                        ui.h4("가중치 설정"),
                        ui.input_slider("yeongcheon_weight_area", "면적 가중치", min=0, max=1, value=0.3, step=0.05),
                        ui.input_slider("yeongcheon_weight_perimeter", "둘레 가중치", min=0, max=1, value=0.3, step=0.05),
                        ui.input_slider("yeongcheon_weight_distance", "거리 가중치", min=0, max=1, value=0.2, step=0.05),
                        ui.input_slider("yeongcheon_weight_facilities", "시설수 가중치", min=0, max=1, value=0.2, step=0.05),
                        ui.input_action_button("yeongcheon_apply_filters", "입력",
                        style=(
                            "width: 92%; "
                            "margin-top: 10px; "
                            "padding: 10px; "
                            "background-color: #1e3a8a; "
                            "color: white; "
                            "border: none; "
                            "border-radius: 5px; "
                            "font-weight: bold;"
                        )
                        ),
                        class_="sidebar-section"
                    ),
                    id="yeongcheon-sidebar"
                ),

                # 지도 타입 선택
                ui.div(
                    ui.input_radio_buttons(
                        "yeongcheon_map_type", "지도 종류 선택",
                        choices={"normal": "일반 지도", "satellite": "위성 지도"},
                        selected="normal", inline=True
                    ),
                    style="position: fixed; bottom: 20px; right: 20px; z-index: 9999; background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); max-width: 300px;"
                ),

                # 동적 리스트와 차트
                ui.output_ui("yeongcheon_dynamic_list"),
                ui.output_ui("yeongcheon_dynamic_chart"),

                # 설명 박스
                ui.div(
                    ui.output_ui("yeongcheon_description"),
                    style="""
                        position: fixed;
                        top: 80px;
                        right: 20px;
                        z-index: 9999;
                        background-color: rgba(255,255,255,0.95);
                        padding: 15px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                        max-width: 350px;
                    """
                )
        )








    # 부록 UI 함수
    def appendix_ui():
        return ui.div(
            ui.div(
                # 제목 섹션 (카드 형태 없이)
                ui.div(
                    ui.h1("부록 : 분석 방법론 및 지표 산출 가이드", 
                        style="text-align: center; color: #1e3a8a; margin-bottom: 40px; padding: 30px 0; border-bottom: 3px solid #1e3a8a;"),
                ),

                # 경상북도 분석 섹션
                ui.div(
                    ui.h2("경상북도 반려동물 친화도 종합 분석"),
                    
                    ui.h3("1. 레이더 차트 구성 요소"),
                    ui.p("경상북도의 반려동물 친화도를 5개 핵심 지표로 평가합니다:"),
                    ui.tags.ul(
                        ui.tags.li("산책 환경"),
                        ui.tags.li("반려동물 시설"),
                        ui.tags.li("교통 안전"),
                        ui.tags.li("치안"),
                        ui.tags.li("대기 환경")
                    ),

                    ui.h3("2. 각 지표별 산출 방법"),
                    
                    ui.h4("2.1 산책 환경"),
                    ui.p("공원 면적이 넓을수록 좋은 산책 환경을 갖추었음을 의미합니다."),
                    ui.div(
                        "1인당 공원면적 = 총 공원면적(㎡) ÷ 총 인구수",
                        class_="formula-box"
                    ),
                    ui.div(
                        "레이더 차트 정규화 점수 = (해당 지역 1인당 공원면적) ÷ (최대 1인당 공원면적)",
                        class_="formula-box"
                    ),

                    ui.h4("2.2 반려동물 시설"),
                    ui.p("반려동물 관련 시설이 많을수록 편리한 반려동물 환경을 제공합니다."),
                    ui.div(
                        "1인당 시설수 = 총 반려동물 시설수 ÷ 총 인구수",
                        class_="formula-box"
                    ),
                    ui.div(
                        "레이더 차트 정규화 점수 = (해당 지역 1인당 시설수) ÷ (최대 1인당 시설수)",
                        class_="formula-box"
                    ),

                    ui.h4("2.3 교통 안전"),
                    ui.p("교통사고가 적을수록 산책 안전도가 높음을 의미합니다."),
                    ui.div(
                        "1인당 사고율 = 평균 교통사고 건수 ÷ 총 인구수",
                        class_="formula-box"
                    ),
                    ui.div(
                        "레이더 차트 정규화 점수 = 1 ÷ (1인당 사고율)",
                        class_="formula-box"
                    ),

                    ui.h4("2.4 치안"),
                    ui.p("범죄 발생률이 낮을수록 안전한 산책 환경을 제공합니다."),
                    ui.div(
                        "1인당 범죄율 = 총 범죄건수 ÷ 총 인구수",
                        class_="formula-box"
                    ),
                    ui.div(
                        "레이더 차트 정규화 점수 = 1 ÷ (1인당 범죄율)",
                        class_="formula-box"
                    ),

                    ui.h4("2.5 대기 환경"),
                    ui.p("PM2.5, PM10, O3, CO, NO2 농도를 종합하여 계산하고, 농도가 낮을수록 대기 환경이 좋음을 의미합니다."),
                    ui.div(
                        "종합 오염도 = PM2.5_정규화 + PM10_정규화 + O3_정규화 + CO_정규화 + NO2_정규화",
                        class_="formula-box"
                    ),
                    ui.div(
                        "레이더 차트 정규화 점수 = 1 ÷ (종합 오염도)",
                        class_="formula-box"
                    ),

                    ui.div(
                        ui.h4("중요 포인트"),
                        ui.p("• 레이더 차트에서 모든 지표는 0~1 사이로 정규화되어 공정한 비교가 가능합니다."),
                        ui.p("• 지표 상세 분석에서 산책 환경과 반려동물 시설은 높을수록 좋고, 교통 안전, 치안, 대기 환경은 낮을수록 좋습니다."),
                        ui.p("• 여러 지역을 선택하여 상대적 비교가 가능합니다."),
                        class_="highlight-box"
                    ),

                    class_="appendix-section"
                ),

                # 영천시 분석 섹션
                ui.div(
                    ui.h2("영천시 저수지 개발 적합도 분석"),
                    
                    ui.h3("1. 개발 적합도 점수 구성"),
                    ui.p("반려동물 산책로로서의 저수지 개발 적합성을 4개 요소로 평가합니다:"),
                    
                    ui.div(
                        ui.h4("면적 지표 (기본 가중치: 30%)"),
                        ui.p("• 넓은 면적일수록 충분한 산책 공간 제공"),
                        class_="method-step"
                    ),

                    ui.div(
                        ui.h4("둘레 지표 (기본 가중치: 30%)"),
                        ui.p("• 긴 둘레일수록 다양한 산책로 확보 가능"),
                        class_="method-step"
                    ),

                    ui.div(
                        ui.h4("접근성 지표 (기본 가중치: 20%)"),
                        ui.p("• 인구 밀집 지역과의 거리가 가까울수록 접근성 우수"),
                        ui.p("• 영천시 내 학교, 약국, 병원, 마트 위치 데이터를 수집하여 KMeans 군집분석을 통해 도출한 주요 중심지와의 최단거리 계산"),
                        ui.p("• 거리가 가까울수록 높은 점수 (역산 적용)"),
                        class_="method-step"
                    ),

                    ui.div(
                        ui.h4("편의시설 지표 (기본 가중치: 20%)"),
                        ui.p("• 반경 2km 내 반려동물 관련 시설 수"),
                        ui.p("• 더 많은 편의시설이 있을수록 높은 점수"),
                        ui.p("• Haversine 공식으로 정확한 거리 계산"),
                        class_="method-step"
                    ),

                    ui.h3("2. 개발 적합도 점수 계산 공식"),
                    ui.div(
                        "개발 적합도점수 = (w₁ × 면적_정규화) + (w₂ × 둘레_정규화) + (w₃ × 거리_정규화) + (w₄ × 시설수_정규화)",
                        class_="formula-box"
                    ),
                    ui.p("여기서 w₁ + w₂ + w₃ + w₄ = 1 (가중치 합계 = 100%)"),

                    ui.h3("3. 정규화 방법 (Min-Max 정규화)"),
                    ui.div(
                        "정규화 점수 = (값 - 최솟값) ÷ (최댓값 - 최솟값)",
                        class_="formula-box"
                    ),
                    ui.p("• 결과: 0~1 사이의 값으로 표준화"),
                    ui.p("• 거리 지표는 1 - 정규화값으로 계산 (가까울수록 높은 점수)"),

                    ui.h3("4. 거리 계산 (Haversine 공식)"),
                    ui.p("지구의 곡률을 고려한 정확한 거리 계산을 위해 Haversine 공식을 사용합니다:"),
                    ui.div(
                        "a = sin²(Δφ/2) + cos φ₁ × cos φ₂ × sin²(Δλ/2)",
                        ui.br(),
                        "c = 2 × atan2(√a, √(1−a))",
                        ui.br(),
                        "d = R × c",
                        ui.br(),
                        "여기서 R = 6,371km (지구 반지름), φ = 위도, λ = 경도",
                        class_="formula-box"
                    ),

                    ui.h3("5. 사용자 맞춤 설정"),
                    ui.div(
                        ui.h4("가중치 조정"),
                        ui.p("• 사용자가 직접 각 지표의 가중치를 0~1 사이에서 조정 가능"),
                        ui.p("• 시스템이 자동으로 전체 가중치를 100%로 정규화"),
                        ui.p("• 실시간으로 개발 적합도 순위가 업데이트됨"),
                        class_="highlight-box"
                    ),

                    ui.div(
                        ui.h4("필터링 옵션"),
                        ui.p("• 특정 읍면동으로 분석 범위 제한 가능"),
                        ui.p("• 상위 N개 저수지만 선별하여 분석"),
                        ui.p("• 가중치 변경 시 즉시 결과 반영"),
                        class_="highlight-box"
                    ),

                    class_="appendix-section"
                ),

                id="appendix-content"
            )
        )


    # ===== 경상북도 관련 서버 함수들 =====
    selected_regions_for_analysis = reactive.Value([])
    
    @reactive.effect
    @reactive.event(input.gyeongbuk_apply_selection)
    def update_gyeongbuk_analysis_regions():
        selected_regions_for_analysis.set(input.gyeongbuk_selected_areas() or [])
        log_memory_usage("경상북도 분석 실행")
    
    @output()
    @render.ui
    def gyeongbuk_map():
        return ui.HTML(create_gyeongbuk_map(selected_regions=selected_regions_for_analysis()))

    @output()
    @render.ui
    def gyeongbuk_popup():
        sel = selected_regions_for_analysis()
        if not sel:
            return ui.div(style="display: none;")
        
        try:
            cache_key = f"radar_chart_{hash(tuple(sorted(sel)))}"
            fig = get_cached_or_compute(
                cache_key,
                plot_radar_chart,
                park_fp=DATA_DIR / "시군별_공원_면적.xlsx",
                acc_fp=DATA_DIR / "경상북도 시도별 교통사고 건수.xlsx",
                facility_fp=DATA_DIR / "한국문화정보원_전국 반려동물 동반 가능 문화시설 위치 데이터_20221130.csv",
                pop_fp=DATA_DIR / "경상북도 주민등록.xlsx",
                crime_fp=DATA_DIR / "경찰청_범죄 발생 지역별 통계.xlsx",
                pollution_fp=DATA_DIR / "월별_도시별_대기오염도.xlsx",
                selected_regions=sel
            )
            
            fig.update_layout(
                polar=dict(
                    domain=dict(x=[0.15, 0.75], y=[0.1, 0.9]),
                    radialaxis=dict(visible=True, range=[0, 1], side="clockwise", angle=90),
                    angularaxis=dict(rotation=90, direction="clockwise")
                ),
                showlegend=True,
                legend=dict(
                    orientation='v',
                    x=1.05,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle'
                ),
                width=520,
                height=330,
                margin=dict(t=30, b=20, l=20, r=140)
            )
            
            radar_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={'displayModeBar': False})
        except Exception as e:
            radar_html = f"<div>레이더 차트 생성 오류: {e}</div>"
        
        return ui.div(
            ui.tags.button(
                "×", 
                onclick="document.getElementById('gyeongbuk-popup-container').style.display = 'none';",
                style="""
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    border: none;
                    background: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                """
            ),
            ui.div(
                ui.h4("선택 지역 비교", style="margin: 0; display: inline-block;"),
                ui.tags.button(
                    "자세히 보기", 
                    onclick="""
                        const container = document.getElementById('gyeongbuk-details-container');
                        const btn = this;
                        if (container.style.display === 'none' || container.style.display === '') {
                            container.style.display = 'block';
                            btn.innerText = '닫기';
                            btn.style.backgroundColor = '#f44336';
                        } else {
                            container.style.display = 'none';
                            btn.innerText = '자세히 보기';
                            btn.style.backgroundColor = '#1e3a8a';
                        }
                    """,
                    style="""
                        margin-left: 10px;
                        padding: 4px 8px;
                        width: 70px;
                        background-color: #1e3a8a;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: bold;
                        cursor: pointer;
                        text-align: center;
                    """
                ),
                style="margin-top: 20px; margin-bottom: 15px; display: flex; align-items: center;"
            ),
            ui.div(
                ui.HTML(radar_html),
                style="height: 380px; margin-bottom: 10px; overflow: hidden;"
            ),
            id="gyeongbuk-popup-container",
            style="""
                position: fixed; 
                top: 80px; 
                right: 20px; 
                width: 520px; 
                height: 420px; 
                background-color: white; 
                padding: 20px; 
                box-shadow: 0 2px 12px rgba(0,0,0,0.3); 
                z-index: 9999; 
                border-radius: 12px; 
                overflow: hidden;
                display: block;
            """
        )

    @output()
    @render.ui
    def gyeongbuk_details():
        sel = selected_regions_for_analysis()
        if not sel:
            return ui.div()
        
        # 각 탭별 차트 생성
        try:
            # 대기오염 차트
            cache_key = f"pollution_{hash(tuple(sorted(sel)))}"
            pollution_df = get_cached_or_compute(
                cache_key,
                analyze_air_pollution_data,
                DATA_DIR / "월별_도시별_대기오염도.xlsx"
            )
            
            if pollution_df is not None and not pollution_df.empty:
                pollutant_cols = [c for c in pollution_df.columns if c.endswith('_평균')]
                norm = pollution_df.copy()
                for col in pollutant_cols:
                    norm[col] = (norm[col] / norm[col].max()).astype(np.float32)
                norm['total_pollution'] = norm[pollutant_cols].sum(axis=1)
                norm_sel = norm[norm['시군구'].isin(sel)].sort_values('total_pollution', ascending=False)
                
                pollution_fig = go.Figure()
                for _, row in norm_sel.iterrows():
                    region = row['시군구']
                    hover_parts = [f"<b>{region}</b><br>"]
                    hover_parts.append(f"총 오염도: {row['total_pollution']:.3f}<br>")
                    for col in pollutant_cols:
                        pollutant_name = col.split('_')[0]
                        hover_parts.append(f"{pollutant_name}: {row[col]:.3f}<br>")
                    
                    pollution_fig.add_trace(go.Bar(
                        x=[region], y=[row['total_pollution']], name=region,
                        marker_color=REGION_COLORS.get(region, '#808080'),
                        hovertemplate=''.join(hover_parts) + '<extra></extra>',
                        showlegend=True
                    ))
                
                pollution_fig.update_layout(
                    width=580, height=350, showlegend=True,
                    legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                    yaxis=dict(title='총 대기오염 지수'),
                    xaxis=dict(tickangle=-45, automargin=True),
                    margin=dict(t=30, b=60, l=40, r=120),
                    template='plotly_white'
                )
                pollution_html = pollution_fig.to_html(full_html=False, include_plotlyjs="cdn", config={'displayModeBar': False})
            else:
                pollution_html = "<div>대기오염 데이터를 불러올 수 없습니다.</div>"

            # 범죄율 차트
            crime_df = get_cached_or_compute(
                f"crime_{hash(tuple(sorted(sel)))}",
                analyze_crime_rate,
                DATA_DIR / "경찰청_범죄 발생 지역별 통계.xlsx",
                DATA_DIR / "경상북도 주민등록.xlsx"
            )
            crime_sel = crime_df[crime_df['region'].isin(sel)]
            crime_fig = px.bar(crime_sel, x='region', y='범죄율', color='region',
                              color_discrete_map=REGION_COLORS, labels={'범죄율':'1인당 범죄율','region':''})
            crime_fig.update_layout(width=580, height=350, showlegend=True,
                                   legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                   xaxis=dict(automargin=True, tickangle=-45),
                                   margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            crime_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 범죄율: %{y:.5f}<extra></extra>')
            crime_html = crime_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

            # 교통사고 차트  
            traffic_df = get_cached_or_compute(
                f"traffic_{hash(tuple(sorted(sel)))}",
                analyze_accident_data,
                DATA_DIR / "경상북도 시도별 교통사고 건수.xlsx"
            )
            traffic_sel = traffic_df[traffic_df['시군'].isin(sel)]
            traffic_fig = px.bar(traffic_sel, x='시군', y='사고비율', color='시군',
                                color_discrete_map=REGION_COLORS, labels={'사고비율':'1인당 평균 사고','시군':''})
            traffic_fig.update_layout(width=580, height=350, showlegend=True,
                                     legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                     xaxis=dict(automargin=True, tickangle=-45),
                                     margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            traffic_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 사고 건수: %{y:.5f}<extra></extra>')
            traffic_html = traffic_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

            # 공원면적 차트
            park_df = get_cached_or_compute(
                f"park_{hash(tuple(sorted(sel)))}",
                analyze_park_area,
                DATA_DIR / "시군별_공원_면적.xlsx"
            )
            park_sel = park_df[park_df['시군'].isin(sel)]
            park_fig = px.bar(park_sel, x='시군', y='공원면적비율', color='시군',
                             color_discrete_map=REGION_COLORS, labels={'공원면적비율':'1인당 공원면적','시군':''})
            park_fig.update_layout(width=580, height=350, showlegend=True,
                                  legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                  xaxis=dict(automargin=True, tickangle=-45),
                                  margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            park_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 공원면적: %{y:.2f}㎡<extra></extra>')
            park_html = park_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

            # 반려동물 시설 차트
            facility_df = get_cached_or_compute(
                f"facility_{hash(tuple(sorted(sel)))}",
                analyze_population_facility_ratio,
                DATA_DIR / "한국문화정보원_전국 반려동물 동반 가능 문화시설 위치 데이터_20221130.csv",
                DATA_DIR / "경상북도 주민등록.xlsx"
            )
            facility_sel = facility_df[facility_df['region'].isin(sel)]
            facility_fig = px.bar(facility_sel, x='region', y='per_person', color='region',
                                 color_discrete_map=REGION_COLORS, labels={'per_person':'1인당 시설 수','region':''})
            facility_fig.update_layout(width=580, height=350, showlegend=True,
                                      legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle'),
                                      xaxis=dict(automargin=True, tickangle=-45),
                                      margin=dict(t=30, b=60, l=40, r=120), template='plotly_white')
            facility_fig.update_traces(hovertemplate='<b>%{x}</b><br>1인당 시설 수: %{y:.6f}<extra></extra>')
            facility_html = facility_fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

        except Exception as e:
            pollution_html = crime_html = traffic_html = park_html = facility_html = f"<div>차트 생성 오류: {e}</div>"

        return ui.div(
            ui.tags.button(
                "×", 
                onclick="""
                    document.getElementById('gyeongbuk-details-container').style.display = 'none';
                """,
                style="""
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    border: none;
                    background: none;
                    font-size: 20px;
                    cursor: pointer;
                    color: #666;
                """
            ),
            ui.h4("지표 상세 분석", style="margin-top: 20px;"),
            ui.div(
                ui.navset_tab(
                    ui.nav_panel("대기 환경", ui.HTML(pollution_html)),
                    ui.nav_panel("치안", ui.HTML(crime_html)),
                    ui.nav_panel("교통", ui.HTML(traffic_html)),
                    ui.nav_panel("산책 환경", ui.HTML(park_html)),
                    ui.nav_panel("반려동물 시설", ui.HTML(facility_html)),
                    selected="대기 환경"
                ),
                class_="modern-tabs",
                style="margin-top: 10px; transform: scale(0.8); transform-origin: top left; width: 110%; height: 110%;"

            ),
            id="gyeongbuk-details-container",
            style="position: fixed; top: 520px; right: 20px; width: 520px; height: 420px; overflow: hidden; background-color: white; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.3); z-index: 9998; border-radius: 12px; display: none;"
        )

    # ===== 영천시 관련 서버 함수들 =====
    yeongcheon_selected_marker = reactive.Value(None)
    yeongcheon_show_list = reactive.Value(False)
    yeongcheon_show_chart = reactive.Value(False)
    yeongcheon_current_data = reactive.Value(pd.DataFrame())
    yeongcheon_button_clicks = reactive.Value({})

    @reactive.Effect
    @reactive.event(input.yeongcheon_apply_filters)  
    def handle_yeongcheon_apply():
        if df_yeongcheon.empty:
            return
        
        log_memory_usage("영천시 필터 적용")
        
        # 먼저 저수지 선택 완전히 초기화
        yeongcheon_selected_marker.set(None)
        
        if input.yeongcheon_area() == "전체":
            filtered = df_yeongcheon.copy()
        else:
            filtered = df_yeongcheon[df_yeongcheon['행정동명'] == input.yeongcheon_area()].copy()
        
        # 가중치 적용
        w_area = input.yeongcheon_weight_area()
        w_perimeter = input.yeongcheon_weight_perimeter()
        w_distance = input.yeongcheon_weight_distance()
        w_facilities = input.yeongcheon_weight_facilities()
        
        total = w_area + w_perimeter + w_distance + w_facilities
        if total == 0:
            w_area, w_perimeter, w_distance, w_facilities = 0.3, 0.3, 0.2, 0.2
            total = 1
        else:
            w_area /= total
            w_perimeter /= total
            w_distance /= total
            w_facilities /= total

        filtered['적합도점수'] = (
            w_area * filtered['면적_정규화'] +
            w_perimeter * filtered['둘레_정규화'] +
            w_distance * filtered['거리_정규화'] +
            w_facilities * filtered['시설수_정규화']
        ).astype(np.float32)
        
        top_data = filtered.nlargest(max(1, input.yeongcheon_top_n()), '적합도점수').reset_index(drop=True)
        top_data = optimize_dataframe_memory(top_data)
        
        yeongcheon_current_data.set(top_data)
        yeongcheon_show_list.set(True)
        yeongcheon_show_chart.set(True)
        yeongcheon_button_clicks.set({})
        
        # 메모리 정리
        del filtered
        gc.collect()
        
        print(f"분석 실행됨 - 저수지 선택 초기화됨")

    # 엔터키 눌림 처리 추가 (영천시)
    @reactive.Effect
    @reactive.event(input.yeongcheon_enter_key_pressed)
    def handle_yeongcheon_enter_key():
        print("영천시 엔터키 감지됨 - 입력 버튼과 동일한 동작 실행")
        if df_yeongcheon.empty:
            return
        
        log_memory_usage("영천시 엔터키 입력")
        
        # 먼저 저수지 선택 완전히 초기화
        yeongcheon_selected_marker.set(None)
            
        if input.yeongcheon_area() == "전체":
            filtered = df_yeongcheon.copy()
        else:
            filtered = df_yeongcheon[df_yeongcheon['행정동명'] == input.yeongcheon_area()].copy()
        
        # 가중치 적용
        w_area = input.yeongcheon_weight_area()
        w_perimeter = input.yeongcheon_weight_perimeter()
        w_distance = input.yeongcheon_weight_distance()
        w_facilities = input.yeongcheon_weight_facilities()
        
        total = w_area + w_perimeter + w_distance + w_facilities
        if total == 0:
            w_area, w_perimeter, w_distance, w_facilities = 0.3, 0.3, 0.2, 0.2
            total = 1
        else:
            w_area /= total
            w_perimeter /= total
            w_distance /= total
            w_facilities /= total

        filtered['적합도점수'] = (
            w_area * filtered['면적_정규화'] +
            w_perimeter * filtered['둘레_정규화'] +
            w_distance * filtered['거리_정규화'] +
            w_facilities * filtered['시설수_정규화']
        ).astype(np.float32)
        
        top_data = filtered.nlargest(max(1, input.yeongcheon_top_n()), '적합도점수').reset_index(drop=True)
        top_data = optimize_dataframe_memory(top_data)
        
        yeongcheon_current_data.set(top_data)
        yeongcheon_show_list.set(True)
        yeongcheon_show_chart.set(True)
        yeongcheon_button_clicks.set({})
        
        # 메모리 정리
        del filtered
        gc.collect()
        
        print(f"엔터키로 영천시 입력 실행됨 - 데이터 길이: {len(top_data)}, 저수지 선택 초기화됨")

    @output
    @render.ui
    def yeongcheon_map():
        if yeongcheon_show_list.get() and not yeongcheon_current_data.get().empty:
            return ui.HTML(create_yeongcheon_map(
                yeongcheon_selected_marker.get(), 
                input.yeongcheon_map_type(), 
                yeongcheon_current_data.get(),
                input.yeongcheon_area()
            ))
        else:
            return ui.HTML(create_yeongcheon_map(
                map_type=input.yeongcheon_map_type(), 
                locations=pd.DataFrame(),
                selected_area=input.yeongcheon_area()
            ))

    @output
    @render.ui
    def yeongcheon_dynamic_list():
        if not yeongcheon_show_list.get() or yeongcheon_current_data.get().empty:
            return ui.div()
        
        top_data = yeongcheon_current_data.get()
        
        return ui.div(
            ui.tags.button("✕", class_="close-btn", onclick="Shiny.setInputValue('yeongcheon_close_list', Math.random());"),
            ui.div(
                ui.h2(f"개발 적합도 상위 {len(top_data)}개 저수지", style="font-size: 18px; margin-bottom: 15px;"),
                *[ui.input_action_button(f"yeongcheon_btn_{i}", 
                                       label=f"{i+1}. {row['시설명']}", 
                                       style="margin-bottom: 5px; width: 100%;")
                  for i, row in top_data.iterrows()],
                style="max-height: 350px; overflow-y: auto;"
            ),
            style="position: fixed; top: 80px; left: 380px; z-index: 9998; background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 320px;"
        )

    @output
    @render.ui
    def yeongcheon_dynamic_chart():
        if not yeongcheon_show_chart.get() or yeongcheon_current_data.get().empty:
            return ui.div()
        
        top_data = yeongcheon_current_data.get()
        
        return ui.div(
            ui.tags.button("✕", class_="close-btn", onclick="Shiny.setInputValue('yeongcheon_close_chart', Math.random());"),
            ui.div(
                create_barplot(top_data),
                style="margin-top: 25px;"
            ),
            style="position: fixed; top: 480px; left: 380px; z-index: 9998; background-color: rgba(255,255,255,0.95); padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); width: 320px;"
        )

    @reactive.Effect
    @reactive.event(input.yeongcheon_close_list)
    def handle_yeongcheon_close_list():
        yeongcheon_show_list.set(False)

    @reactive.Effect
    @reactive.event(input.yeongcheon_close_chart)
    def handle_yeongcheon_close_chart():
        yeongcheon_show_chart.set(False)

    # 저수지 버튼 클릭 처리
    @reactive.Effect  
    def handle_yeongcheon_button_clicks():
        if not yeongcheon_show_list.get() or yeongcheon_current_data.get().empty:
            return
            
        top_data = yeongcheon_current_data.get()
        current_clicks = yeongcheon_button_clicks.get()
        
        for i, row in top_data.iterrows():
            btn_name = f"yeongcheon_btn_{i}"
            if hasattr(input, btn_name):
                current_value = input[btn_name]()
                if current_value and current_value > current_clicks.get(btn_name, 0):
                    yeongcheon_selected_marker.set(row['시설명'])
                    current_clicks[btn_name] = current_value
                    yeongcheon_button_clicks.set(current_clicks.copy())
                    print(f"저수지 선택됨: {row['시설명']}")
                    break

    @output
    @render.ui
    def yeongcheon_description():
        name = yeongcheon_selected_marker.get()
        
        if not name or df_yeongcheon.empty:
            return ui.p("저수지를 선택해 주세요.")
        
        row = df_yeongcheon[df_yeongcheon['시설명'] == name]
        if row.empty:
            return ui.p("정보를 찾을 수 없습니다.")
        
        row = row.iloc[0]
       
        return ui.div(
            ui.h3(name),
            ui.p(f"주소: {row['소재지지번주소']}"),
            ui.p(f"행정동: {row['행정동명']}"),
            ui.p(f"지도상 명칭: {row.get('지도상명칭', row.get('지도상 명칭', '정보 없음'))}"),
            ui.p(f"면적: {row['면적']:.2f} m²"),
            ui.p(f"둘레: {row['둘레']:.2f} m"),
            ui.p(f"인구 밀집지역과의 거리: {row['중심거리_km']:.2f} km"),
            ui.p(f"개발 적합도 점수: {row['적합도점수']:.2f}")
        )

    # 세션 종료 시 메모리 정리
    @reactive.Effect
    def cleanup():
        if '_cached_data' in locals() or '_cached_data' in globals():
            _cached_data.clear()
        gc.collect()
        log_memory_usage("세션 종료")
        
        session.on_ended(cleanup)

# 앱 생성 시 메모리 로깅
log_memory_usage("앱 초기화 시작")

here = os.path.dirname(__file__)
static_path = str(WWW_DIR)

app = App(app_ui, server, static_assets=static_path)

log_memory_usage("앱 초기화 완료")