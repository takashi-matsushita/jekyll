---
layout: post
categories: [Kaggle,]
tags: [GIS,]
title: Energy consumption in the Netherlands
author: Takashi MATSUSHITA
---
[Kaggle](https://www.kaggle.com) にある[オランダの年別エネルギー消費量データ](https://www.kaggle.com/lucabasa/dutch-energy/)を見てみる.

choropleth map を作成するためにオランダの shapefile を[こちら](http://www.diva-gis.org/gdata)から取得.
また、郵便番号と都市名から州へマッピングするための情報を、[ここ](https://postcodebijadres.nl/postcodes-nederland) と [ここ](https://www.geonames.org/postalcode-search.html?country=NL) から取得.

電力データの読み込み.
```python
import pandas as pd

years = list(range(2010, 2020))
firms = ('enexis', 'liander', 'stedin')
net_manager = {'enexis': 'Enexis B.V.', 'liander': 'Liander N.V.', 'stedin': 'Stedin'}

elec = []
for firm in firms:
  for year in years:
    if firm in ('enexis', 'liander'):
      path = 'Electricity/{}_electricity_0101{}.csv'.format(firm, year)
    else:
      path = 'Electricity/{}_electricity_{}.csv'.format(firm, year)
    print('inf> reading {}'.format(path))
    df = pd.read_csv(path)
    df['year'] = year
    df['net_manager'] = net_manager[firm]
    elec.append(df)

df = pd.concat(elec)
```
スマートメータの数、自家発電量、有効接続数を計算.
```python
df['smartmeters'] = df.smartmeter_perc/100*df.num_connections
df['in_house'] = (100.-df.delivery_perc)/100.*df.annual_consume
df['active_connections'] = df.perc_of_active_connections/100.*df.num_connections
```
都市名と郵便番号から州名へ変換.
```python
def zip2province(city, code):
  for k, v in province.items(): 
    if code[:4] in v: return k
  print('war> unknown zip code {}'.format(code))
  if city in cities:
    return cities[city]
  return 'NA'

df['province'] = df.apply(lambda x: zip2province(x['city'], x['zipcode_from']), axis=1)
```

電力供給元毎に総電力消費量の経年変化を表示する.
```python
import matplotlib.pyplot as plt
plt.style.use('bmh') 

data = df.groupby(['net_manager', 'year']).agg({'annual_consume': 'sum'})
for k, v in net_manager.items():
  plt.plot(data.xs(v, level='net_manager'), label=v, marker='o')
plt.xlabel('Year')
plt.ylabel('Energy consumption [kWh]')
plt.legend()
plt.ylim(ymin=0)
plt.tight_layout()
```
![NLD energy consumption by proviers]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/nld-energy-consumption.png)
同様に、接続毎の電力消費量、スマートメータの普及率、自家発電率の経年変化を見てみる.
![NLD energy consumption by proviers]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/nld-consumption-connection.png)
![NLD energy consumption by proviers]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/nld-fraction-smartmeter.png)
![NLD energy consumption by proviers]({{ site.url }}/{{ site.baseurl }}/assets/img/posts/nld-inhouse-generation.png)


次に、2019年のデータを使用して接続毎電力消費量の choropleth map を作成.
```python
import geoviews as gv
gv.extension('bokeh')

data = df[df.year==2019].groupby(['province']).agg({'annual_consume': 'sum', 'active_connections': 'sum'})
data = data.annual_consume/data.active_connections

datum = pd.DataFrame(data)
datum['province'] = datum.index
datum.reset_index(drop=True, inplace=True)  
datum.columns = ['ratio', 'province']
gdf = gpd.GeoDataFrame(pd.merge(nld[['NAME_1','geometry']], datum, left_on='NAME_1', right_on='province'))
options = {'title': '[kWh]'}
fig = gv.Polygons(gdf, vdims=['NAME_1', 'ratio']).opts(
    tools=['hover'], width=500, height=400, color_index='ratio',
    colorbar=True, toolbar='above', xaxis=None, yaxis=None).opts(title="接続毎電力消費量 [kWh] (2019)", colorbar_opts=options)
gv.save(fig, 'nld-consumption-connection.html')
```
{% include figures/nld-consumption-connection.html %}
同様に、スマートメータ普及率、自家発電率の choropleth map を作成.
{% include figures/nld-fraction-smartmeter.html %}
{% include figures/nld-inhouse-generation.html %}

スマートメータの普及は全体としては進んでいるが、地域毎のバラツキが多きい. 自家発電の普及も同様.

今回使用したコードは[こちら](https://github.com/takashi-matsushita/lab/blob/master/gis/nld-electricity.py).
