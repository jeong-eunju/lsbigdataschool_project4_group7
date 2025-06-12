# 영천시 반려동물 친화 환경 분석 대시보드

경상북도 및 영천시의 반려동물 친화 환경을 대시보드 형태로 분석할 수 있는 Shiny-Python 대시보드입니다.

## 주요 기능

### 경상북도 탭
- 22개 시·군구 중복 선택  
- 레이더 차트: 산책 환경, 반려동물 시설, 교통 안전, 치안, 대기 환경  
- 세부 지표 막대차트: 대기오염, 범죄율, 교통사고, 공원 면적, 시설 밀도  

### 영천시 탭
- 읍·면·동별 필터링  
- 면적·둘레·거리·편의시설 가중치 조정  
- 상위 N개 저수지 순위 리스트 및 막대차트  
- 지도 마커 클릭 시 상세 정보 표시  

### 부록 탭
- 분석 지표 및 산출 방법론 정리  
- 수식 및 정규화 방법 안내  

## 기술 스택
- Python 3.9+  
- shiny for Python  
- Pandas, NumPy, GeoPandas, Shapely  
- Folium (지도)  
- Plotly (차트)  
- Matplotlib (폰트 설정)  
- CSS 커스터마이징  

## 설치 및 실행

```bash
git clone https://github.com/your-org/yeongcheon-pet-friendly.git
cd yeongcheon-pet-friendly

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

## 데이터 폴더 구조
```bash
data/
├─ 경상북도_영천시_저수지및댐.xlsx
├─ DA_EMD_202307.shp 및 관련 shapefile 파일
├─ DA_SIG_202307.shp 및 관련 shapefile 파일
├─ 시군별_공원_면적.xlsx
├─ 경찰청_범죄 발생 지역별 통계.xlsx
├─ 경상북도 주민등록.xlsx
├─ 전국 반려동물 동반 가능 문화시설 위치 데이터_20221130.csv
└─ 월별_도시별_대기오염도.xlsx
```

