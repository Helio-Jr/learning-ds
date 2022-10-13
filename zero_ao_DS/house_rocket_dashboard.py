from email.policy import default
from turtle import color
import streamlit as st
import pandas    as pd
import numpy     as np
import seaborn as sns
import plotly.express as px 
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
def get_data( path ): #Leitura do dataset no formato csv e retorno dele no formado dataframe
    data = pd.read_csv( path )
    return data

def set_attributes( data ): #Ajustando e criando colunas necessárias para a elaboração do projeto
    data['date'] = pd.to_datetime(df['date'])
    data['month'] = data['date'].apply(lambda x: str(x)[5:7])
    data['season'] = data['month'].apply(lambda x: season(x))

    return data

def drop_rows(data): #Removendo as linhas com base nas premissas
    #Excluindo imóvel com 33 banheiros
    data = data.drop(15870)
    #Excluindo a primeira aparição do imóvel com id repetido
    data = data.drop(2496)

    return data

def season(month): #Função para determinar a sazonalidade a partir do mês
    month = int(month)
    if month <= 2 or month == 12:
        return 'Winter'
    elif 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    else:
        return 'Fall'

def selling_price(x): #Determinando o preço de venda a partir da condição e maior preço médio de sazonalidade por zipcode
    if x['best_median_price'] > x['price']:
        price = x['price'] + (x['price']*30/100)
        if x['condition'] == 5:
            price = price + price*15/100
    else:
        price = x['price'] + (x['price']*10/100)

    return price

# ------------------------------------------
# Data Overview
# ------------------------------------------

def data_overview(df):
    st.title('Welcome to House Rocket Dashboard')
    st.header('Data Overview')
    st.sidebar.title('Data Overview')

    input_id = st.sidebar.number_input('Digite o ID do imóvel que deseja:', key = 0) #Colocando a opção do usuário digitar um número (Que deve ser um id)

    df_id = df.copy()
    if input_id == []: 
        df_id = df.copy()
    else:
        df_id = df.loc[df['id'] == input_id, :] #Filtrando o dataframe para mostrar apenas as informações do id requisitado
        if df_id.shape[0] == 0:
            df_id = df.copy()

    st.write(df_id) #Apresentando o dataframe na tela

    c1, c2 = st.columns((1, 4))  

    c1.header('Data types')
    c1.write((df.dtypes).astype(str)) #Apresentando os tipos de variáveis de cada coluna do dataframe na tela

    df_num = df.drop(['id', 'date', 'month', 'season'], axis=1) #Selecionando as colunas úteis pra analisar estatísticamente

    #Criando um dataframe para cada ferramenta estatíscas escolhidas (Valor Mínimo, Valor Máximo, Range, Mediana, Média e Desvio Padrão)
    df_min = pd.DataFrame(df_num.apply(np.min))
    df_max = pd.DataFrame(df_num.apply(np.max))
    df_range = pd.DataFrame(df_num.apply(lambda x: x.max() - x.min()))
    df_median = pd.DataFrame(df_num.apply(np.median))
    df_mean = pd.DataFrame(df_num.apply(np.mean))
    df_std = pd.DataFrame(df_num.apply(np.std))

    desc_stats = pd.concat([df_min, df_max, df_range, df_median, df_mean, df_std], axis=1) #Juntando os dataframes
    desc_stats.columns = ['min', 'max', 'range', 'median', 'mean', 'std']
    desc_stats = desc_stats.T

    c2.header('Descriptive statistics table')
    c2.write(desc_stats)

# ------------------------------------------
# Hypothesis
# ------------------------------------------

def h1_h2(df): #Testando hipóteses 1 e 2
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

def h3_h4(df): #Testando hipóteses 3 e 4
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

def h5_h6(df): #Testando hipóteses 5 e 6
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

def h7_h8(df): #Testando hipóteses 7 e 8
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

def h9_h10(df): #Testando hipóteses 9 e 10
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

def hypothesis(df): #Apresentando todas as hipóteses na tela
    st.title('Hypothesis')
    h1_h2(df)
    h3_h4(df)
    h5_h6(df)
    h7_h8(df)
    h9_h10(df)

# ------------------------------------------
# Business Questions
# ------------------------------------------

def question1(data, df): #Resolvendo a primeira questão de negócio
    #Quais são os imóveis que a House Rocket deveria comprar e por qual preço?
    st.header('Quais são os imóveis que a House Rocket deveria comprar e por qual preço?')

    df_agrouped = data[['zipcode', 'price']].groupby('zipcode').median().sort_values('price').reset_index()
    df_filtrado = data.loc[:, ['id', 'lat', 'long', 'zipcode', 'season', 'condition', 'waterfront', 'bathrooms', 'bedrooms', 'price']]

    df1 = pd.merge(df_filtrado, df_agrouped, on='zipcode', how='inner')          
    df1.columns = ['id', 'lat', 'long', 'zipcode', 'season', 'condition', 'waterfront', 'bathrooms', 'bedrooms', 'price', 'median_price']

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

    #Criando e dimensionando o mapa
    density_map = folium.Map( location=[df['lat'].mean(), df['long'].mean() ],
                                default_zoom_start=15 ) 

    #Plotando os pontos com imóveis sugeridos para venda no mapa
    marker_cluster = MarkerCluster().add_to( density_map )
    for name, row in df_compra.iterrows():
        folium.Marker( [row['lat'], row['long'] ], 
            popup='Sold R${0}. Condition: {2} Percent_diff: {3}. ID: {1}'.format( row['price'], row['id'], row['condition'], row['percent_diff']), fill_color='YlOrRd').add_to( marker_cluster )

    c1, c2 = st.columns((1, 1))

    c1.write(df_compra)
    c1.subheader(f'Número de imóveis recomendados para compra com essas condições: {df_compra.shape[0]-1}')

    with c2:
        folium_static( density_map ) 

    return df_compra

def question2(df_compra): #Respondendo a segunda questão de negócio
    #Uma vez a casa comprada, qual o melhor momento para vendê-las e por qual preço?
    try:

        st.header('Uma vez a casa comprada, qual o melhor momento para vendê-las e por qual preço?')

        df_agrouped = df_compra[['zipcode', 'season', 'price']].groupby(['zipcode', 'season']).median().sort_values('zipcode').reset_index()
        df_agrouped = df_agrouped.rename(columns={'price':'season_median_price'})

        zipcodes_list = list(df_compra['zipcode'].unique())

        row_list = []

        for i in zipcodes_list: #Loop que percorrerá todos os zipcodes
            df_zipcode = df_agrouped.loc[df_agrouped['zipcode'] == i, :]
            
            for index, row in df_zipcode.iterrows(): #Loop que adicionará na lista "row_list" a linha do zipcode quando o preço médio da sazonalidade é igual o preço médio máximo dentre todas as sazonalidades daquele zipcode
                if row['season_median_price'] == int(df_zipcode['season_median_price'].max()):
                    row_list.append(row)
                    break
                if row['zipcode'] == 98010:
                    row_list.append(row)
                    break
                
        df_agp = pd.DataFrame(row_list)

        df_agp.columns = ['zipcode', 'best_season', 'best_median_price']

        df_entrega = df_compra[['id', 'zipcode', 'condition', 'season', 'waterfront', 'bathrooms', 'bedrooms', 'price']]
        df_final = pd.merge(df_entrega, df_agp, on='zipcode', how='left')
        df_final['selling_price'] = df_final.apply(lambda x: selling_price(x), axis = 1)
        df_final['profit'] = df_final.apply(lambda x: x['selling_price'] - x['price'], axis = 1)

        st.write(df_final.sort_values('profit', ascending=False))   
        return df_final

    except:
        st.subheader('Não existe nenhum imóvel que corresponda com os valores solicitados')

def business_questions(df): #Exibindo o resultado das duas questões de negócio + apresentando o lucro obtido caso a empresa aplicasse o método de precificar e vender os imóveis dentro dos imóveis sugeridos para compra
    st.title('Business questions')
    st.sidebar.title( 'Business questions' )
    st.sidebar.subheader( 'Altere os filtros para selecionar o imóvel ideal:' )

    # filters
    min_price = int( df['price'].min() )
    max_price = int( df['price'].max() )
    avg_price = int( df['price'].mean() )

    bathrooms_options = df['bathrooms'].unique()
    bathrooms_options.sort()
    bedroomns_options = df['bedrooms'].unique()
    bedroomns_options.sort()

    f_price = st.sidebar.slider( 'Preço em $:', min_price, max_price, avg_price )
    f_waterfront = st.sidebar.checkbox('Apresentar apenas imóveis com vista para o mar?')
    f_bathrooms = st.sidebar.selectbox('Selecione o número máximo de banheiros:', bathrooms_options, 2)
    f_bedrooms = st.sidebar.selectbox('Selecione o número máximo de quartos:', bedroomns_options, 2)

    if f_waterfront:
        data = df[(df['price'] <= f_price) & (df['waterfront'] == 1) & (df['bathrooms'] <= f_bathrooms) & (df['bedrooms'] <= f_bedrooms)]
    else:
        data = df[(df['price'] <= f_price) & (df['bathrooms'] <= f_bathrooms) & (df['bedrooms'] <= f_bedrooms)]


    df_compra = question1(data, df)

    try:
        df_final = question2(df_compra)

        if df_final.shape[0] != 0:
            df_profit = pd.DataFrame(df_final[['price', 'selling_price', 'profit']].apply(np.sum)).T

            st.header('Tabela de lucro')

            st.write(df_profit)
    except:
        st.subheader('Altere os filtros para visualizar os imóveis desejados')


if __name__ == "__main__": #Função principal para execução
    #Extraction
    path = 'kc_house_data.csv'
    df = get_data(path)

    #Transformation
    set_attributes(df)
    df = drop_rows(df)
    data_overview(df)
    hypothesis(df)
    business_questions(df)

