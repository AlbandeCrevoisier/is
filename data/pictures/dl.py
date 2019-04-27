from subprocess import run

for i in range(191246, 191360):
    for j in range(272365, 272477):
        run(['curl', "-o", "p_" + str(i) + '_' + str(j) + ".jpeg",
             "http://wxs.ign.fr/pratique/geoportail/wmts?SERVICE=WMTS&request=GetTile&version=1.0.0&layer=ORTHOIMAGERY.ORTHOPHOTOS&tilematrixset=PM&tilerow="
              + str(i) + "&tilematrix=19&tilecol=" + str(j)
              + "&layer=ORTHOIMAGERY.ORTHOPHOTOS&format=image/jpeg&style=normal"])
