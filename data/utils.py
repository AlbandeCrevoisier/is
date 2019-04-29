import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.ops import unary_union
from pyproj import Proj, transform
from urllib.request import urlopen
from subprocess import run


def geoJSON_to_satellite_download(parcels):
    """Download the 256x256x3 thumbnails of a list of parcels."""
    parcels_obj = []
    for parcel in parcels:
        parcels_obj.append(shape(parcel['geometry']))
    big_parcel = unary_union(parcels_obj)
    xmin, ymin, xmax, ymax = big_parcel.bounds
    col_min, col_max, row_min, row_max = get_tiles_list(xmin, ymin, xmax, ymax)
    for col in range(int(col_min), int(col_max) + 1):
        for row in range(int(row_min), int(row_max) + 1):
            run(['curl', "-o", "p_" + str(col) + '_' + str(row) + ".jpeg",
                 get_tiles_url(col, row)])


def geoJSON_to_satellite_view(parcels):
    parcels_obj = []
    for parcel in parcels:
        parcels_obj.append(shape(parcel['geometry']))
    big_parcel = unary_union(parcels_obj)

    xmin, ymin, xmax, ymax = big_parcel.bounds
    min_col, max_col, min_row, max_row = get_tiles(xmin, ymin, xmax, ymax)

    img = None
    for row in range(int(min_row), int(max_row) + 1):
        img0 = None
        for col in range(int(min_col), int(max_col) + 1):
            image = plt.imread(urlopen(get_tiles_url(col, row)), format='JPEG')
            if img0 is None:
                img0 = image
            else:
                img0 = np.concatenate((img0, image), axis=1)
        if img is None:
            img = img0
        else:
            img = np.concatenate((img, img0), axis=0)
    return img


def get_tiles_list(xmin, ymin, xmax, ymax):
    min_col, min_row = convert_to_tile(xmin, ymin)
    max_col, max_row = convert_to_tile(xmax, ymax)
    if min_col > max_col:
        temp = min_col
        min_col = max_col
        max_col = temp
    if min_row > max_row:
        temp = min_row
        min_row = max_row
        max_row = temp
    return min_col, max_col, min_row, max_row


def convert_to_tile(x, y):
    X, Y = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), x, y)
    X0 = -20037508.3427892476320267
    Y0 = 20037508.3427892476320267
    Resolution = 0.2985821417
    col = (X - X0) / (256 * Resolution)
    row = (Y0 - Y) / (256 * Resolution)
    return col, row


def get_tiles_url(col, row):
    # Note the use of `pratique` key
    return f"http://wxs.ign.fr/pratique/geoportail/wmts?service=WMTS"\
          f"&request=GetTile" \
          f"&version=1.0.0&layer=ORTHOIMAGERY.ORTHOPHOTOS" \
          f"&tilematrixset=PM&tilematrix=19&" \
          f"tilecol={col}&" \
          f"tilerow={row}&" \
          f"layer=ORTHOIMAGERY.ORTHOPHOTOS&format=image/jpeg&style=normal"

