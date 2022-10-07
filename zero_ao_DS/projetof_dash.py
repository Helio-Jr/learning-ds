from turtle import color
import streamlit as st
import pandas    as pd
import numpy     as np
import seaborn as sns
import plotly.express as px 
import geopandas
import folium

from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster

# ------------------------------------------
# settings
# ------------------------------------------

st.set_page_config( layout='wide' )
sns.set()

# ------------------------------------------
# Helper functions
# ------------------------------------------
@st.cache( allow_output_mutation=True )
def get_data( path ):
    data = pd.read_csv( path )
    return data

@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile

def set_attributes( data ):
    data['date'] = pd.to_datetime(df['date'])
    data['month'] = data['date'].apply(lambda x: str(x)[5:7])
    data['season'] = data['month'].apply(lambda x: season(x))

    return data

def drop_rows(data):
    #Excluindo imóvel com 33 banheiros
    data = data.drop(15870)
    #Excluindo a primeira aparição do imóvel com id repetido
    data = data.drop(2496)

    return data

def season(month):
    month = int(month)
    if month <= 2 or month == 12:
        return 'Winter'
    elif 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    else:
        return 'Fall'

def selling_price(x):
    if x['best_median_price'] > x['price']:
        price = x['price'] + (x['price']*30/100)
        if x['condition'] == 5:
            price = price + price*15/100
    else:
        price = x['price'] + (x['price']*10/100)

    return price

def data_overview(df):
    st.title('Data Overview')
    st.sidebar.title('Data Overview')

    input_id = st.sidebar.number_input('Digite o ID do imóvel que deseja:', key = 0)

    df_id = df.copy()
    if input_id == []:
        df_id = df.copy()
    else:
        df_id = df.loc[df['id'] == input_id, :]
        if df_id.shape[0] == 0:
            df_id = df.copy()

    st.write(df_id)

    c1, c2 = st.columns((1, 4))  

    c1.header('Data types')
    c1.write((df.dtypes).astype(str))

    df_num = df.drop(['id', 'date', 'month', 'season'], axis=1) 

    df_min = pd.DataFrame(df_num.apply(np.min))
    df_max = pd.DataFrame(df_num.apply(np.max))
    df_range = pd.DataFrame(df_num.apply(lambda x: x.max() - x.min()))
    df_median = pd.DataFrame(df_num.apply(np.median))
    df_mean = pd.DataFrame(df_num.apply(np.mean))
    df_std = pd.DataFrame(df_num.apply(np.std))

    desc_stats = pd.concat([df_min, df_max, df_range, df_median, df_mean, df_std], axis=1)
    desc_stats.columns = ['min', 'max', 'range', 'median', 'mean', 'std']
    desc_stats = desc_stats.T

    c2.header('Statistic table')
    c2.write(desc_stats)

def h1_h2(df):
    c1, c2 = st.columns((1, 1))  

    df_agrouped = df[['price', 'waterfront']].groupby('waterfront').mean().reset_index()

    c1.header('Hipótese 1: Imóveis que possuem vista para água, são pelo menos 30% mais caros, na média.')
    c1.write(df_agrouped)
    c1.write('Hipótese confirmada. Em média, o valor dos imóveis com vista para o mar são mais que 3x maiores aos imóveis sem esse privilégio.')

    df_agrouped = df.loc[df['yr_built'] < 1955, 'price'].mean()
    df_agrouped2 = df.loc[df['yr_built'] >= 1955, 'price'].mean()

    c2.header('Hipótese 2: Imóveis com data de construção menor que 1955, são 50% mais baratos, na média.')
    c2.write(f'Média de preço dos imóveis construídos antes de 1955: {df_agrouped}')
    c2.write(f'Média de preço dos imóveis construídos a partir de 1955: {df_agrouped2}')
    c2.write('Hipótese falsa. A diferença média de preço é muito menor que a especulada.')

def h3_h4(df):
    c1, c2 = st.columns((1, 1))  

    df_agrouped = df.loc[df['bathrooms'] == 3, ['price', 'month']].groupby('month').mean().sort_values('month').reset_index()
    fig = px.line(df_agrouped, x='month', y='price')

    c1.header('Hipótese 3: Imóveis com 3 banheiros tem um crescimento MoM (Month over Month) de 15%')
    c1.plotly_chart( fig, use_container_width=True )
    c1.write('Hipótese falsa. A média de preço dos imóveis com 3 banheiros mês a mês é irregular e imprevisível.')

    df_agrouped = df[['price', 'yr_built']].groupby('yr_built').mean().reset_index()
    fig = px.line( df_agrouped, x='yr_built', y='price' )

    c2.header('Hipótese 4: O crescimento do preço dos imóveis YoY (Year over Year) é de 10%')
    c2.plotly_chart( fig, use_container_width=True )
    c2.write('Hipótese falsa. A média de preço ano após ano também é irregular e imprevisível.')


def h5_h6(df):
    c1, c2 = st.columns((1, 1))  

    df_agrouped = df.loc[df['sqft_basement'] == 0, 'sqft_lot'].mean()
    df_agrouped2 = df.loc[df['sqft_basement'] != 0, 'sqft_lot'].mean()

    c1.header('Hipótese 5: Imóveis com porão possuem o lote 50% maiores do que os imóveis sem porão.')
    c1.write(f'Média de tamanho em pés quadrados dos lotes sem porão: {round(df_agrouped,3)}')
    c1.write(f'Média de tamanho em pés quadrados dos lotes com porão: {round(df_agrouped2,3)}')
    c1.write('Hipótese falsa. Na verdade, em média, os imóveis com porão são mais baratos do que os sem porão.')

    df_agrouped = df[['price', 'yr_renovated']].groupby('yr_renovated').mean().sort_values('yr_renovated').reset_index()
    df_agrouped2 = df.loc[(df['yr_built'] == 1934) | (df['yr_built'] >= 1940), ['price', 'yr_built']].groupby('yr_built').mean().sort_values('yr_built').reset_index()
    df_agrouped.columns = ['year', 'mean_price_yr_renovated']
    df_agrouped2.columns = ['year', 'mean_price_yr_built']
    df_agrouped = pd.merge(df_agrouped, df_agrouped2, on='year', how='inner')

    c2.header('Hipótese 6: Imóveis renovados são em média 20% mais baratos que os imóveis construidos no mesmo ano')
    c2.write(df_agrouped)
    c2.write('Hipótese falsa. Depende do ano.')

def h7_h8(df):
    c1, c2 = st.columns((1, 1))  

    df_agrouped = df[['price', 'condition']].groupby('condition').mean().sort_values('condition').reset_index()
    fig = px.bar( df_agrouped, x='condition', y='price')

    c1.header('Hipótese 7: Imóveis variam pelo menos 30% de média de preço por condição')
    c1.plotly_chart( fig, use_container_width=True )
    c1.write('Hipótese falsa. Entretanto, da pra notar um aumento de 15% da média do preço da condição 4 para 5')

    df_agrouped = df[['price', 'floors']].groupby('floors').mean().sort_values('floors').reset_index()
    fig = px.bar( df_agrouped, x='floors', y='price')

    c2.header('Hipótese 8: Imóveis variam 20% de média de preço por número de andares')
    c2.plotly_chart( fig, use_container_width=True )
    c2.write('Hipótese falsa. Embora ocorra um crescimento de média de preço entre imóveis de 1 a 2 pisos, isso não se repete até o máximo de pisos.')

def h9_h10(df):
    c1, c2 = st.columns((1, 1))  

    df_agrouped = df.loc[:, ['price', 'bedrooms']].groupby('bedrooms').mean().sort_values('bedrooms').reset_index()
    fig = px.bar( df_agrouped, x='bedrooms', y='price')

    c1.header('Hipótese 9: Imóveis com mais de 2 quartos são 70% mais caros que os que possuem apenas 1')
    c1.plotly_chart( fig, use_container_width=True )
    c1.write('Hipótese falsa. Embora ocorra um crescente aumento da média de preços até imóveis com 7 quartos, a diferença de média dos imóveis com apenas 1 quarto não chega a 70%, comparado aos com 3 quartos, por exemplo.')

    df_agrouped = df[['price', 'grade']].groupby('grade').mean().sort_values('grade').reset_index()
    fig = px.bar( df_agrouped, x='grade', y='price')

    c2.header('Hipótese 10: Imóveis variam 4% ou mais o preço por nota de design, em média')
    c2.plotly_chart( fig, use_container_width=True )
    c2.write('Hipótese verdadeira.')

def hypothesis(df):
    st.title('Hypothesis')
    h1_h2(df)
    h3_h4(df)
    h5_h6(df)
    h7_h8(df)
    h9_h10(df)

def question1(df):
    #Quais são os imóveis que a House Rocket deveria comprar e por qual preço?

    st.header('Quais são os imóveis que a House Rocket deveria comprar e por qual preço?')

    df_agrouped = data[['zipcode', 'price']].groupby('zipcode').median().sort_values('price').reset_index()
    df_filtrado = data.loc[:, ['id', 'lat', 'long', 'zipcode', 'season', 'condition', 'price']]

    df1 = pd.merge(df_filtrado, df_agrouped, on='zipcode', how='inner')          
    df1.columns = ['id', 'lat', 'long', 'zipcode', 'season', 'condition', 'price', 'median_price']

    df1['buy?'] = 0

    for i in range(len(df1)):
        if df1.loc[i, 'condition'] > 3 and df1.loc[i, 'price'] < df1.loc[i, 'median_price']:
            df1.loc[i, 'buy?'] = 'yes'
        else:
            df1.loc[i, 'buy?'] = 'no'

    df_compra = df1[df1['buy?'] == 'yes']
    df_compra.loc[:,'percent_diff'] = round((df_compra.loc[:,'median_price'] - df_compra.loc[:,'price']) * 100/df_compra.loc[:,'price'], 3)
    df_compra = df_compra.drop('buy?', axis=1).sort_values('percent_diff', ascending=False).reset_index().drop('index', axis=1)
    df_compra = df_compra.loc[df_compra['percent_diff'] >= 5, :]

    density_map = folium.Map( location=[df['lat'].mean(), df['long'].mean() ],
                                default_zoom_start=15 ) 

    marker_cluster = MarkerCluster().add_to( density_map )
    for name, row in df_compra.iterrows():
        folium.Marker( [row['lat'], row['long'] ], 
            popup='Sold R${0}. Condition: {2} Percent_diff: {3}. ID: {1}'.format( row['price'], row['id'], row['condition'], row['percent_diff']), fill_color='YlOrRd').add_to( marker_cluster )

    c1, c2 = st.columns((1, 1))
    c1.write(df_compra)

    with c2:
        folium_static( density_map )

def question2(df):
    #Uma vez a casa comprada, qual o melhor momento para vendê-las e por qual preço?

    st.header('Uma vez a casa comprada, qual o melhor momento para vendê-las e por qual preço?')

    # df_agrouped = data[['zipcode', 'season', 'price']].groupby(['zipcode', 'season']).median().sort_values('zipcode').reset_index()
    # df_agrouped = df_agrouped.rename(columns={'price':'season_median_price'})

    # zipcodes_list = list(data['zipcode'].unique())

    # row_list = []

    # for i in zipcodes_list:
    #     df_zipcode = df_agrouped.loc[df_agrouped['zipcode'] == i, :]
        
    #     for index, row in df_zipcode.iterrows():
    #         if row['season_median_price'] == int(df_zipcode['season_median_price'].max()):
    #             row_list.append(row)
            
    # df_agp = pd.DataFrame(row_list)
    # df_agp.columns = ['zipcode', 'best_season', 'best_median_price']

    # df_entrega = data[['id', 'zipcode', 'condition', 'season', 'price']]
    # df_final = pd.merge(df_entrega, df_agp, on='zipcode', how='inner')
    # df_final['selling_price'] = df_final.apply(lambda x: selling_price(x), axis = 1)
    # df_final['profit'] = df_final.apply(lambda x: x['selling_price'] - x['price'], axis = 1)

    #st.write(df_final)

    df_agrouped = df_compra[['zipcode', 'season', 'price']].groupby(['zipcode', 'season']).median().sort_values('zipcode').reset_index()
    df_agrouped = df_agrouped.rename(columns={'price':'season_median_price'})

    zipcodes_list = list(df_compra['zipcode'].unique())

    row_list = []

    for i in zipcodes_list:
        df_zipcode = df_agrouped.loc[df_agrouped['zipcode'] == i, :]
        
        for index, row in df_zipcode.iterrows():
            if row['season_median_price'] == int(df_zipcode['season_median_price'].max()):
                row_list.append(row)
                break
            if row['zipcode'] == 98010:
                row_list.append(row)
                break
            
    df_agp = pd.DataFrame(row_list)

    df_agp.columns = ['zipcode', 'best_season', 'best_median_price']

    df_entrega = df_compra[['id', 'zipcode', 'condition', 'season', 'price']]
    df_final = pd.merge(df_entrega, df_agp, on='zipcode', how='left')
    df_final['selling_price'] = df_final.apply(lambda x: selling_price(x), axis = 1)
    df_final['profit'] = df_final.apply(lambda x: x['selling_price'] - x['price'], axis = 1)

    df_profit = pd.DataFrame(df_final[['price', 'selling_price', 'profit']].apply(np.sum)).T
    st.write(df_final.sort_values('zipcode'))

    return df_profit

def business_questions(df):
    st.title('Business questions')
    st.sidebar.title( 'Business questions' )
    st.sidebar.subheader( 'Select Max Price' )

    # filters
    min_price = int( df['price'].min() )
    max_price = int( df['price'].max() )
    avg_price = int( df['price'].mean() )

    f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )

    data = df[df['price'] < f_price]

    question1(data)
    profit = question2(data)

    st.header('Tabela de lucro')

    st.write(df_profit)
# ------------------------------------------
# Business questions
# ------------------------------------------

#Quais são os imóveis que a House Rocket deveria comprar e por qual preço?

st.title('Business questions')
st.sidebar.title( 'Business questions' )
st.sidebar.subheader( 'Select Max Price' )
st.header('Quais são os imóveis que a House Rocket deveria comprar e por qual preço?')

# filters
min_price = int( df['price'].min() )
max_price = int( df['price'].max() )
avg_price = int( df['price'].mean() )

f_price = st.sidebar.slider( 'Price', min_price, max_price, avg_price )

data = df[df['price'] < f_price]

df_agrouped = data[['zipcode', 'price']].groupby('zipcode').median().sort_values('price').reset_index()
df_filtrado = data.loc[:, ['id', 'lat', 'long', 'zipcode', 'season', 'condition', 'price']]

df1 = pd.merge(df_filtrado, df_agrouped, on='zipcode', how='inner')          
df1.columns = ['id', 'lat', 'long', 'zipcode', 'season', 'condition', 'price', 'median_price']

df1['buy?'] = 0

for i in range(len(df1)):
    if df1.loc[i, 'condition'] > 3 and df1.loc[i, 'price'] < df1.loc[i, 'median_price']:
        df1.loc[i, 'buy?'] = 'yes'
    else:
        df1.loc[i, 'buy?'] = 'no'

df_compra = df1[df1['buy?'] == 'yes']
df_compra.loc[:,'percent_diff'] = round((df_compra.loc[:,'median_price'] - df_compra.loc[:,'price']) * 100/df_compra.loc[:,'price'], 3)
df_compra = df_compra.drop('buy?', axis=1).sort_values('percent_diff', ascending=False).reset_index().drop('index', axis=1)
df_compra = df_compra.loc[df_compra['percent_diff'] >= 5, :]

density_map = folium.Map( location=[df['lat'].mean(), df['long'].mean() ],
                              default_zoom_start=15 ) 

marker_cluster = MarkerCluster().add_to( density_map )
for name, row in df_compra.iterrows():
    folium.Marker( [row['lat'], row['long'] ], 
        popup='Sold R${0}. Condition: {2} Percent_diff: {3}. ID: {1}'.format( row['price'], row['id'], row['condition'], row['percent_diff']), fill_color='YlOrRd').add_to( marker_cluster )

c1, c2 = st.columns((1, 1))
c1.write(df_compra)

with c2:
    folium_static( density_map )

#Uma vez a casa comprada, qual o melhor momento para vendê-las e por qual preço?

# df_agrouped = data[['zipcode', 'season', 'price']].groupby(['zipcode', 'season']).median().sort_values('zipcode').reset_index()
# df_agrouped = df_agrouped.rename(columns={'price':'season_median_price'})

# zipcodes_list = list(data['zipcode'].unique())

# row_list = []

# for i in zipcodes_list:
#     df_zipcode = df_agrouped.loc[df_agrouped['zipcode'] == i, :]
    
#     for index, row in df_zipcode.iterrows():
#         if row['season_median_price'] == int(df_zipcode['season_median_price'].max()):
#             row_list.append(row)
        
# df_agp = pd.DataFrame(row_list)
# df_agp.columns = ['zipcode', 'best_season', 'best_median_price']

# df_entrega = data[['id', 'zipcode', 'condition', 'season', 'price']]
# df_final = pd.merge(df_entrega, df_agp, on='zipcode', how='inner')
# df_final['selling_price'] = df_final.apply(lambda x: selling_price(x), axis = 1)
# df_final['profit'] = df_final.apply(lambda x: x['selling_price'] - x['price'], axis = 1)

st.header('Uma vez a casa comprada, qual o melhor momento para vendê-las e por qual preço?')
#st.write(df_final)

df_agrouped = df_compra[['zipcode', 'season', 'price']].groupby(['zipcode', 'season']).median().sort_values('zipcode').reset_index()
df_agrouped = df_agrouped.rename(columns={'price':'season_median_price'})

zipcodes_list = list(df_compra['zipcode'].unique())

row_list = []

for i in zipcodes_list:
    df_zipcode = df_agrouped.loc[df_agrouped['zipcode'] == i, :]
    
    for index, row in df_zipcode.iterrows():
        if row['season_median_price'] == int(df_zipcode['season_median_price'].max()):
            row_list.append(row)
            break
        if row['zipcode'] == 98010:
            row_list.append(row)
            break
        
df_agp = pd.DataFrame(row_list)

df_agp.columns = ['zipcode', 'best_season', 'best_median_price']

df_entrega = df_compra[['id', 'zipcode', 'condition', 'season', 'price']]
df_final = pd.merge(df_entrega, df_agp, on='zipcode', how='left')
df_final['selling_price'] = df_final.apply(lambda x: selling_price(x), axis = 1)
df_final['profit'] = df_final.apply(lambda x: x['selling_price'] - x['price'], axis = 1)

df_profit = pd.DataFrame(df_final[['price', 'selling_price', 'profit']].apply(np.sum)).T
st.write(df_final.sort_values('zipcode'))

st.header('Tabela de lucro')

st.write(df_profit)

if __name__ == "__main__":
    path = 'kc_house_data.csv'
    df = get_data(path)
    set_attributes(df)