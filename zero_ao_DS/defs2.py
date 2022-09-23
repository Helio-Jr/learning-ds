import time
from geopy.geocoders import Nominatim

geolocator = Nominatim( user_agent='geopiExercises' )

def get_longlat( x ):
    index, row = x
    response = geolocator.reverse( row['query'] )
    address = response.raw['address']
    try:
        postcode = address['postcode'] if 'postcode' in address else 'NA'
        return postcode

    except:
        return None
