from select import select
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px

st.set_page_config(layout='wide')

# read data
@st.cache( allow_output_mutation=True )
def get_data( path):
    data = pd.read_csv( path , sep = ',')
    data['date'] = pd.to_datetime( data['date'] )
    return data

df = get_data( 'kc_house_data.csv' )
df['price_m2'] = df['price']/(df['sqft_lot']*0.092903)

select_atrib = st.sidebar.multiselect('Enter Columns', df.columns)
select_zipcode = st.sidebar.multiselect('Enter zipcode', df['zipcode'].unique())

st.title('Data Overview')

if select_atrib != [] and select_zipcode != []:
    data = df.loc[df['zipcode'].isin(select_zipcode), select_atrib]
elif select_atrib != [] and select_zipcode == []:
    data = df.loc[:, select_atrib]
elif select_atrib == [] and select_zipcode != []:
    data = df.loc[df['zipcode'].isin(select_zipcode), :]
else: 
    data = df.copy()

st.write(data.head())

c1, c2 = st.columns((1,1))

df1 = data.loc[:,['zipcode', 'id']].groupby('zipcode').count()
df2 = data.loc[:,['zipcode', 'price']].groupby('zipcode').mean()
df3 = data.loc[:,['zipcode', 'sqft_living']].groupby('zipcode').mean()
df4 = data.loc[:,['zipcode', 'price_m2']].groupby('zipcode').mean()

m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
m3 = pd.merge(m2, df4, on='zipcode', how='inner').reset_index()

m3.columns = ['zipcode', 'total', 'price','sqft_living', 'price_m2']
c1.dataframe(m3)

num_at = data.select_dtypes(['float64', 'int64'])
mean = pd.DataFrame(num_at.apply(np.mean))
median = pd.DataFrame(num_at.apply(np.median))
std = pd.DataFrame(num_at.apply(np.std))
min = pd.DataFrame(num_at.apply(np.min))
max = pd.DataFrame(num_at.apply(np.max))
range = pd.DataFrame(num_at.apply(lambda column: column.max() - column.min()))

desc = pd.concat([mean, median, std, min, max, range], axis = 1).reset_index()
desc.columns = ['atributtes','mean', 'median', 'std', 'min','max', 'range']

c2.dataframe(desc)
