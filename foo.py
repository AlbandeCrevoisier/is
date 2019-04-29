import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json
from data.utils import *

commune = json.load(open('data/cadastre-06105-parcelles.json'))['features']

m = load_model('model.h5')

for parcel in commune:
    g = shape(parcel['geometry'])
    xmin, ymin, xmax, ymax = g.bounds
    col_min, col_max, row_min, row_max = get_tiles_list(xmin, ymin, xmax, ymax)
    for c in range(int(col_min), int(col_max + 1)):
        for r in range(int(row_min), int(row_max + 1)):
            img = plt.imread(urlopen(get_tiles_url(c, r)), format='JPEG')
            plt.imshow(img)
            plt.show()
            print(m.predict([[img / 255]]) > 0.5)

