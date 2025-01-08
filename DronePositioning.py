import math
import pandas as pd
import os
import h5py
import datetime
import numpy as np

def calc_dist(origin, end):
    '''
    Calcula la distancia entre dos puntos en coordenadas geográficas (Latitude, Longitude, Height)
    y retorna una tupla con la distancias en coordenadas rectangulares (+X, +Y, +Z) del punto final con respecto al
    punto inicial.

    * origin y end son tuplas de 3 elementos (coordenadas geográficas)
    * Se retorna una tupla de 3 elementos (coordenadas rectangulares)
    '''

    r = 6372.8  # radio de la tierra en km

    lat_o = origin[0]
    lon_o = origin[1]
    alt_o = origin[2]
    lat_f = end[0]
    lon_f = end[1]
    alt_f = end[2]

    dLat = math.radians(lat_f - lat_o)
    dLon = math.radians(lon_f - lon_o)
    lat_o = math.radians(lat_o)
    lat_f = math.radians(lat_f)

    # Calculo del bearing
    diffLong = math.radians(lon_f - lon_o)
    x = math.sin(diffLong) * math.cos(lat_f)
    y = math.cos(lat_o) * math.sin(lat_f) - \
        (math.sin(lat_o) * math.cos(lat_f) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    # Calculo de la distancia entre punto y punto
    a = math.sin(dLat / 2) ** 2 + math.cos(lat_o) * \
        math.cos(lat_f) * math.sin(dLon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    distance = r * c

    # Calculo de la distancia en cada eje dependiendo del bearing y la distancia p-p
    distX = distance * math.sin(math.radians(bearing)) * 1000  # metros
    distY = distance * math.cos(math.radians(bearing)) * 1000
    distZ = alt_f - alt_o

    return (distX, distY, distZ)

def readHdf5Drone(main_path, cableLength, sphereRadius, radarPosition):


    '''
    Lee un archivo .hdf5 conteniendo el dataset de angulos y coordenadas de GPS, elabora el posicionamiento
    del drone con respecto al radar como origen en coordenadas cartesianas (X - Este, Y - Norte), y el de la
    esfera con respecto a la posicion del drone en base a los angulos roll y pitch leidos, retorna un
    DataFrame de Pandas con las posiciones de drone y la esfera.

    * Variable PATH_HDF5_DRONE indica la ruta del archivo .hdf5 a leer
    * Se retorna un dataframe de Pandas

    '''

    L = cableLength - sphereRadius
    path_hdf5 = main_path + '/drone_hdf5'

    rows = []

    files_h5 = os.listdir(path_hdf5)
    file_h5 = h5py.File(path_hdf5 + "/" + files_h5[0], mode='r')

    for reg in file_h5["dset"]:

        # datetm = datetime(int(reg[1]), int(reg[2]), int(reg[3]), int(reg[4]), int(reg[5]), int(reg[6]), int(reg[7]))

        coords_drone = (reg[8], reg[9], reg[10])  # Lat, Long, Alt del drone
        yaw_drone = reg[14]
        dist_drone = calc_dist(radarPosition, coords_drone)

        roll = reg[11]
        pitch = reg[12]
        yaw = reg[13]

        roll = math.radians(float(roll))
        pitch = math.radians(float(pitch))

        if (roll != 0.0) and (pitch != 0.0) and (reg[1] >= 2022):
            # datetm = datetime(int(reg[1]), int(reg[2]), int(reg[3]), int(reg[4]), int(reg[5]), int(reg[6]), int(reg[7]))
            datetm = datetime.datetime(int(reg[1]), int(reg[2]), int(reg[3]), int(reg[4]), int(reg[5]), int(reg[6]))
            # Algoritmo de posicionamiento

            # Relacion entre a y b
            a = math.cos(pitch)/math.cos(roll)
            # print(a)

            # Se halla A en terminos de b
            A = math.sqrt((a * math.sin(roll)) ** 2 + (a * math.sin(pitch)) ** 2)
            # print(A)

            # Se halla theta
            B = math.cos(pitch)
            theta = math.atan(A / B)
            # print(math.degrees(theta))

            # Se halla A y B
            A1 = L * math.sin(theta)
            B1 = L * math.cos(theta)
            # print(A1, B1)

            # Hallamos phi
            D = math.sin(pitch)
            C = math.sin(roll)
            phi = math.atan(D / C)
            # print(phi)

            # Correccion
            delta = phi + np.radians(360 - yaw_drone)

            # E y F
            E = A1 * math.cos(delta)
            F = A1 * math.sin(delta)

            x_esf = dist_drone[0] + F
            y_esf = dist_drone[1] + E
            z_esf = dist_drone[2] - B1

            range_dro = math.sqrt((dist_drone[0]) ** 2 + (dist_drone[1]) ** 2)
            range_esf = math.sqrt(x_esf ** 2 + y_esf ** 2)

            pos_esf = (x_esf, y_esf, z_esf)

            rows.append({"timestamp": datetm,
                         "x_drone": dist_drone[0],
                         "y_drone": dist_drone[1],
                         "z_drone": dist_drone[2],
                         "r_drone": range_dro,
                         "x_esfera": x_esf,
                         "y_esfera": y_esf,
                         "z_esfera": z_esf,
                         "r_esfera": range_esf,
                         "yaw_drone": yaw_drone})

    return pd.DataFrame(rows)