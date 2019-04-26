import cv2
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from pyproj import Proj, transform
from urllib.request import urlopen




def geoJSON_to_satellite_view(list_parcelles):
    parcelles_obj_list = []
    for parcelle in list_parcelles:
        parcelles_obj_list.append(shape(parcelle['geometry']))
    big_parcelle = unary_union(parcelles_obj_list)

    minx, miny, maxx, maxy = big_parcelle.bounds
    min_col, max_col, min_row, max_row = get_titles_list(minx, miny, maxx, maxy)

    img = None
    for row in range(int(min_row), int(max_row+1)):
        img0 = None
        for col in range(int(min_col), int(max_col + 1)):
            image = url_to_image(get_tiles_url(col, row))
            if img0 is None:
                img0 = image
            else:
                img0 = np.concatenate((img0, image), axis=1)
        if img is None:
            img = img0
        else:
            img = np.concatenate((img, img0), axis=0)
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros((height, width), dtype=np.uint8)

    list_points = []
    for x, y in big_parcelle.exterior.coords:
        pix_x, pix_y = convert_to_pixel(int(min_col), int(min_row), x, y)
        list_points.append([int(round(pix_x)), int(round(pix_y))])

    points = np.array([list_points])
    cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(img, img, mask=mask)

    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    cv2.imwrite("/tmp/croped.png", cropped)
    return cropped


def get_titles_list(minx, miny, maxx, maxy):
    min_col, min_row = convert_to_title(minx, miny)
    max_col, max_row = convert_to_title(maxx, maxy)
    if(min_col>max_col):
        temp = min_col
        min_col = max_col
        max_col = temp
    if(min_row>max_row):
        temp = min_row
        min_row = max_row
        max_row = temp
    return min_col, max_col, min_row, max_row


def convert_to_pixel(min_col, min_row, x, y):
    X, Y = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), x, y)
    X0 = -20037508.3427892476320267
    Y0 = 20037508.3427892476320267
    Resolution = 0.2985821417
    X1 = X0 + min_col *(256 * Resolution)
    Y1 = Y0 - min_row *(256 * Resolution)

    pixel_x = (X - X1)/Resolution
    pixel_y = (Y1-Y)/Resolution
    return pixel_x, pixel_y


def convert_to_title(x, y):
    X, Y = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), x, y)
    X0 = -20037508.3427892476320267
    Y0 = 20037508.3427892476320267
    Resolution = 0.2985821417
    TILECOL = (X - X0) / (256 * Resolution)
    TILEROW = (Y0 - Y) / (256 * Resolution)
    return TILECOL, TILEROW


def get_tiles_url(col, row):
    return f"http://wxs.ign.fr/f01up8cve8mfujd90xgwnjaj/geoportail/wmts?service=WMTS&request=GetTile" \
          f"&version=1.0.0&layer=ORTHOIMAGERY.ORTHOPHOTOS" \
          f"&tilematrixset=PM&tilematrix=19&" \
          f"tilecol={col}&" \
          f"tilerow={row}&" \
          f"layer=ORTHOIMAGERY.ORTHOPHOTOS&format=image/jpeg&style=normal"

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)
    return image