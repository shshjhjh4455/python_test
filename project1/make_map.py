import json
import pandas as pd
import folium

csv_data = pd.read_csv('data/customer_payment_data.csv')

m = folium.Map([43, -102], zoom_start=3)

folium.Choropleth(
    geo_data='data/ctp_rvn.zip.geojson', # 경계선 좌표값이 담긴 데이터
    #data=state_data, # Series or DataFrame 넣으면 된다
    columns=['State', 'Unemployment'], # DataFrame의 어떤 columns을 넣을지
    key_on='feature.id', # id 값을 가져오겠다; feature.id : feature 붙여줘야 함 (folium의 정해진 형식)
    fill_color='BuPu',
    fill_opacity=0.5, # 색 투명도
    line_opacity=0.5, # 선 투명도
    legend_name='Unemployment rate (%)' # 범례
).add_to(m)
m
