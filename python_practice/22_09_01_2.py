import pandas as pd
import folium

df = pd.read_csv('data/older_population.csv')
print(df.info()) 
print(df.head())

geo_data = 'data/seoul-dong.geojson'

center =[37.541,126.986] # 서울시 중앙좌표
m = folium.Map(location=center, zoom_start=12)


#동 단위 노령 인구를 Choropleth로 표현하기
folium.Choropleth(geo_data=geo_data,
                    data=df,
                    columns=['동','인구'],
                    key_on='feature.properties.동',
                    fill_color='YlGn',
                    legend_name='노령 인구수').add_to(m)

m

df_gu = df.groupby(['구'])['인구'].sum().to_frame().reset_index()
print(df_gu.head())

center = [37.541,126.986] 
m = folium.Map(location=center,zoom_start=10)