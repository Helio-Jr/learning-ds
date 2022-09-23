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