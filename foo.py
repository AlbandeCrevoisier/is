from data.satellite import *
import json

parcels = json.load(open("data/cadastre-06105-parcelles.json"))
crop = geoJSON_to_satellite_view(parcels['features'])
