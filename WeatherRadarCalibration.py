import datetime
import os
import h5py
import numpy as np
import math
import pandas as pd
from scipy.constants import speed_of_light
from DronePositioning import readHdf5Drone


class WeatherRadarCalibration:

    def __init__(self):

        #Variables del experimento

        #Posición de SOPHy (Lat, Long, Alt en msnm)
        self.radarLatitude = -11.95198611
        self.radarLongitude = -76.87634722
        self.radarHeight = 522

        self.RadarPosition = (self.radarLatitude, self.radarLongitude, self.radarHeight )

        #Umbral de potencia en dB para reconocer ecos del dron/esfera
        self.powerThreshold = -48.2

        #Resolución en rango de SOPHy
        self.rangeResolution = 15.0

        #Longitud del cable que sostiene la esfera
        self.cableLength = 30.0

        #Radio de la esfera en metros
        self.sphereRadius = 0.178
        #Radar Cross Section
        self.crossSection = math.pi * self.sphereRadius ** 2


        #Ruta dentro del directorio principal que contiene los directorios de AZ por experimento
        self.pathWithinExp = 'param-D_RHI_T_0.1us_0.3Kmm'

        #Parámetros de SOPHy

        self.radarFrequency = 9.375 * 1e9                            # GHz
        self.lambdaSOPHy =  speed_of_light / self.radarFrequency     # m

        self.pulseWidth = 0.1 * 1e-6                                 # us
        self.transmittedPower = 44.0                                 # dBm
        self.transmittedPowerW = 10**((self.transmittedPower-30)/10) # W
        self.lnaGain = 70.0                                          # dB
        self.beamWidth = math.radians(1.8)                           # rad

        self.k_M = 0.93                    #Coeficiente de refraccion del medio



        #Indice del range bin a partir de donde puede estar el drone (empírico - observable) 101 equivale a 220 m aprox,
        #para acotar la región de interes dentro del dataset

        self.idx_min_range_bin = 101  # Indice del primer range bin de deteccion (Para experimentos donde el drone volo a una misma distancia)

    def loadExperiments(self):

        filepath = '/home/gibs/Documentos/radar_cal/SOPHy_calibracion_2024'   # Ruta del directorio de experimentos

        exps = [os.path.join(filepath, i) for i in sorted(os.listdir(filepath))]

        logFilePath = [os.path.join(i,'drone_hdf5') for i in exps]

        return exps[0]


    def get_azimuths_for_exp(self, path):

        '''
        Obtiene las rutas de los barridos por distintos azimuth de un experimento de calibracion
        Entrada --> Ruta de experimento
        Salida --> Lista con rutas por cada AZ del experimento
        '''

        list_azimuths = []

        for az in os.listdir(path):
            list_azimuths.append(az)

        return list_azimuths

    def get_time_for_hdf5(self, file):
        utc_time = datetime.datetime.fromtimestamp(file['Data']['time'][0]).replace(microsecond=0)
        return utc_time

    def get_powerH(self, file):
        hPower = np.asarray(file['Data']['power']['H'])  # Canal H (FINAL)
        # hPower = np.array(file['Data']['data_param']['channel00'])  # Canal H
        # hPower = np.array(file['Data']['data_param']['channel00'])  # Canal H
        return hPower

    def get_powerV(self, file):
        vPower = np.asarray(file['Data']['power']['V'])  # Canal V
        # vPower = np.array(file['Data']['data_param']['channel01'])  # Canal H
        return vPower

    def get_elevation_h5f(self, file):
        elevation = np.asarray(file['Metadata']['elevation'])  # Elevacion
        return elevation

    def get_range_h5f(self, file):
        range = np.asarray(file['Metadata']['range'])  # Rango
        return range

    def get_azimuth_h5f(self, file):
        azimuth = np.asarray(file['Metadata']['azimuth']).mean()
        return azimuth

    def failed_experiment(self, ele_array):

        if (len(ele_array) > 80):
            return True
        return False

    def remove_failed_profiles(self, dPower, threshold):

        '''
        Remueve los perfiles fallados (perfiles con un valor atipico de potencia en todos los range bins).
        Entrada --> Dataset original, Valor de umbral (W)
        Salida --> Dataset filtrado

        '''

        filteredD = dPower.copy()

        # Selecciona los perfiles con NaN/Valor atipico
        first_from_row = filteredD[:, 0]
        idx_fail = np.where(first_from_row > threshold)
        # print(h5_file, idx_fail)

        # Los rellena con un valor de potencia que no intervenga
        for idx in idx_fail[0]:
            filteredD[idx, :] = 1e-7

        # idx_wrong = np.where(np.isnan(first_from_row) == False)

        return filteredD


    def is_sphere_drone_detected(self, dPower, rangeEl):

        """
        Retorna True si del dataset se reconocen ecos correspondientes al drone y la esfera separados por un espacio
        (ecos de potencia menor). Retorna False si no se reconoce el patron ecos fuertes - espacio de ecos debiles - ecos fuertes
        dentro del dataset.
        """

        n_gaps = 0
        end = False

        dPowerdB = 10 * np.log10(dPower)  # dataset en dBm

        # filtro de valores de potencia en base a los valores estimados en un limite referencial (ELE y RAN) donde puede estar el dron/esfera
        x, y = np.where(dPowerdB[(int(rangeEl / 2) - 10):48,
                        self.idx_min_range_bin:self.idx_min_range_bin + 7] > self.powerThreshold)

        # Se empieza buscando desde uno de los primeros perfiles empezando del mas alto

        x += (int(rangeEl / 2) - 10)
        y += self.idx_min_range_bin

        x = list(sorted(set(x)))  # se guardan los perfiles con valores de potencia sospechosos (drone y esfera)

        for i in range(len(x)):

            try:
                # si hay perfiles consecutivos con valores de potencia reconocibles
                if (x[i] + 1) == x[i + 1]:
                    # print("next")
                    if n_gaps >= 1:
                        end = True
                else:
                    # Se detecta el espacio entre el drone y la esfera
                    # print("gap")
                    n_gaps += 1

            except:
                pass

        return end

    def getSphereDroneEchoes(self, power, ele, ran, azi, rangeEl):

        """
        Retorna un dataframe con las posiciones de los ecos asociados al drone y a la esfera en base a los archivos
        hdf5 de potencia y a las consideraciones por cada experimento.

        """

        idx_separation = 0
        powerdB = 10 * np.log10(power)  # Dataset en dB

        # Selecciona los bins en el array con valores de potencia que superen el umbral (POSIBLES ECOS DEL DRONE Y ESFERA)
        rows, cols = np.where(powerdB[(int(rangeEl / 2) - 10):48,
                              self.idx_min_range_bin:self.idx_min_range_bin + 7] > self.powerThreshold)

        # Correccion de los indices tomando en cuenta los offsets para filas y columnas
        rows += (int(rangeEl / 2) - 10)
        cols += self.idx_min_range_bin

        rows_single = sorted(list(rows))
        # print(rows_single)
        # return rows, cols

        # Separacion de los perfiles asociados a los ecos del drone y la esfera
        for profile in range(len(rows_single)):

            try:
                if ((rows_single[profile] + 1) != rows_single[profile + 1] and (
                        rows_single[profile] != rows_single[profile + 1])):
                    idx_separation = profile

            except:
                pass

        profiles_drone = sorted(list(set(rows_single[:idx_separation + 1])))
        profiles_sphere = sorted(list(set(rows_single[idx_separation + 1:])))
        doper = np.array([rows, cols])

        drone_echos = []
        esfera_echos = []

        # Chequea los datos de cada eco asociado al drone y esfera

        for echo in doper.transpose():

            power_echo_db = powerdB[echo[0]][echo[1]]
            power_echo_lnr = 10 ** (power_echo_db / 10) * 100
            range_echo = ran[echo[1]]
            ele_echo = ele[echo[0]]
            r_echo_limits = (round(range_echo - self.rangeResolution/ 2, 4),
                             round(range_echo + self.rangeResolution/ 2, 4))

            if (echo[0] in profiles_drone):

                # print("Drone", round(power_echo,2), 10**(power_echo/10), ele_echo, r_echo_limits)
                drone_echos.append({"power_echo_db_H": power_echo_db,
                                    "power_echo_lnr_H": power_echo_lnr,
                                    "ele_echo": ele_echo,
                                    "range_limits": r_echo_limits,
                                    "azimuth": azi,
                                    "coord": (echo[0], echo[1])})

            elif (echo[0] in profiles_sphere):

                # print("Esfera", round(power_echo,2), 10**(power_echo/10), ele_echo, r_echo_limits)
                esfera_echos.append({"power_echo_db_H": power_echo_db,
                                     "power_echo_lnr_H": power_echo_lnr,
                                     "ele_echo": ele_echo,
                                     "range_limits": r_echo_limits,
                                     "azimuth": azi,
                                     "coord": (echo[0], echo[1])})

        df_drone = pd.DataFrame(drone_echos)
        df_esfera = pd.DataFrame(esfera_echos)

        return df_drone, df_esfera

    def get_max_echo(self, echoes):

        idx_max = echoes["power_echo_lnr_H"].idxmax()
        return echoes.loc[idx_max]


    def getWeightingFunctions(self, pos_esf, pos_dro, esf_echo, yaw_drone):


        sigma_r = 0.35 * self.pulseWidth * speed_of_light / 2


        sigma_xy = math.degrees(self.beamWidth) / 2.36 # BW en rad

        OFFSET = 360 - yaw_drone - 4
        # Valor de offset angular de compensación de Yaw (Azimutal)

        rangeSphere = math.sqrt(pos_esf[3] ** 2 + pos_esf[2] ** 2)  # Rango hasta el target (Directo)
        heightSphere = pos_esf[2]  # Altura del target (msnm) (Proyeccion z)
        distanceSphere = pos_esf[3]  # Rango horizontal (Proyeccion xy)

        gamma = np.rad2deg(np.arctan((pos_dro[1]) / (pos_dro[0])))
        # theta = OFFSET - gamma
        theta = 122 - gamma
        alfa = OFFSET - np.rad2deg(np.arctan(pos_esf[1] / pos_esf[0]))

        # r_o = (esf_echo['range_limits'][0] + esf_echo['range_limits'][1]) / 2 - self.range_offset

        # Centro del volumen de resolución del eco de máxima potencia
        r_o = (esf_echo['range_limits'][0] * 1000 + esf_echo['range_limits'][
            1] * 1000) / 2 - self.rangeResolution

        theta_X_bar = esf_echo["azimuth"]
        theta_Y_bar = esf_echo["ele_echo"]

        theta_X = theta
        theta_Y = math.degrees(math.atan(heightSphere / distanceSphere))

        # Range Weighting Function
        Wr = math.exp(-((rangeSphere - r_o) ** 2) / (2 * sigma_r ** 2))

        # Beam Weighting Function
        Wb = math.exp(-((theta_X - theta_X_bar) ** 2) / (2 * sigma_xy ** 2) - ((theta_Y - theta_Y_bar) ** 2) / (
                    2 * sigma_xy ** 2))

        return Wr, Wb

    def getHardTConstant(self, echo_sphere, pos_esf, rwf, bwf):

        if (isinstance(echo_sphere, pd.Series)):
            r_power = echo_sphere["power_echo_lnr_H"] / 100

        else:
            r_power = echo_sphere / 100

        range = pos_esf[1]
        gLNA_lnr = 10 ** (self.lnaGain / 10)

        #Ltotal = self.lossesConnH.get() + self.lossesCircH.get() + self.lossesWG1H.get() + self.lossesWG2H.get()
        #print(Ltotal)

        # return ((r_power * range**4 * (10**(Ltotal/10)))/(self.transmittedPower.get()*self.crossSection*gLNA_lnr*rwf*bwf))
        return ((r_power * range ** 4) / (self.transmittedPowerW * self.crossSection * gLNA_lnr * rwf * bwf))

    def getSoftTConstant(self, htc):


        return (16 * math.log(2) * self.lambdaSOPHy ** 4 * 10 ** 18) / (
                    htc * speed_of_light * math.pi * self.beamWidth ** 2 * math.pi ** 5 * self.k_M)

    def calculateConstants(self, list_experiments):

        radar_constants = []

        #Itera entre todos los experimentos de un directorio
        for exp in list_experiments:

            list_azimuths =  self.get_azimuths_for_exp(os.path.join(exp, self.pathWithinExp))
            path_hdf5 = os.path.join(exp, self.pathWithinExp)

            df_positions = readHdf5Drone(exp, self.cableLength, self.sphereRadius,self.RadarPosition)

            ref_files = os.listdir(os.path.join(path_hdf5,list_azimuths[0]))
            first_h5_file = h5py.File(os.path.join(path_hdf5, list_azimuths[0], ref_files[0]), 'r')
            elevation_exp = self.get_elevation_h5f(first_h5_file)
            range_elevation_exp = len(elevation_exp)

            #print(range_elevation_exp)

            hardConstantTotal = 0.0
            softConstantTotal = 0.0

            constantCount = 0           #Cuenta por cada RHI aprobado

            for az in sorted(list_azimuths):

                for file in sorted(os.listdir(os.path.join(path_hdf5,az))):

                    h5_file = h5py.File(os.path.join(path_hdf5, az, file), 'r')

                    powerH = self.get_powerH(h5_file)
                    #powerV = self.get_powerV(h5_file)

                    elevation_arr = self.get_elevation_h5f(h5_file)
                    range_arr = self.get_range_h5f(h5_file)
                    azimuth_arr = self.get_azimuth_h5f(h5_file)

                    timestamp = self.get_time_for_hdf5(h5_file)
                    #print(timestamp)

                    # Eliminando los perfiles fallidos
                    powerH_corr = self.remove_failed_profiles(powerH, 0.1)

                    #print(powerH_corr)

                    if(self.is_sphere_drone_detected(powerH_corr, range_elevation_exp)) and self.failed_experiment(elevation_arr) == False :

                        #print("Smash")
                        droneH, sphereH = self.getSphereDroneEchoes(powerH_corr, elevation_arr, range_arr, azimuth_arr,
                                                               range_elevation_exp)


                        # Se recogen los ecos de máxima potencia (asociados a la posición del objeto)
                        if (len(droneH) > 0 and len(sphereH) > 0):

                            try:
                                #print(file)
                                max_echo_drone_H = self.get_max_echo(droneH)
                                max_echo_sphere_H = self.get_max_echo(sphereH)

                                #print(max_echo_sphere_H)

                                df_second = df_positions.loc[ df_positions["timestamp"].dt.strftime('%#d/%#m/%Y %#H:%M:%S') == timestamp.strftime('%#d/%#m/%Y %#H:%M:%S'),:]


                                #Se promedian todas las coincidencias en dicho segundo
                                x_drone_mean = df_second['x_drone'].mean()
                                y_drone_mean = df_second['y_drone'].mean()
                                z_drone_mean = df_second['z_drone'].mean()
                                r_drone_mean = df_second['r_drone'].mean()

                                x_esf_mean = df_second['x_esfera'].mean()
                                y_esf_mean = df_second['y_esfera'].mean()
                                z_esf_mean = df_second['z_esfera'].mean()
                                r_esf_mean = df_second['r_esfera'].mean()

                                yaw_drone_mean = df_second['yaw_drone'].mean()

                                pos_esf = (x_esf_mean, y_esf_mean, z_esf_mean, r_esf_mean)

                                pos_dro = (x_drone_mean, y_drone_mean, z_drone_mean, r_drone_mean)

                                wR, wB = self.getWeightingFunctions(pos_esf, pos_dro, max_echo_sphere_H, yaw_drone_mean)

                                #print(file)

                                #print(f'Wr:{wR} Wb:{wB}')
                                #print(file)

                                if(wR > 10e-5 and wB > 10e-5):


                                    hardConstant = self.getHardTConstant(max_echo_sphere_H,pos_esf,wR,wB)
                                    #print(hardConstant)
                                    softConstant = self.getSoftTConstant(hardConstant)

                                    #print(f'HTC:{hardConstant}  STC:{softConstant}')

                                    hardConstantTotal += hardConstant
                                    softConstantTotal += softConstant

                                    constantCount += 1


                            except:

                                pass
                                #print("----")

            hardConstantTotal /= float(constantCount)
            softConstantTotal /= float(constantCount)

            softConstantExpDb = 10 * np.log10(softConstantTotal)

            radar_constants.append(softConstantExpDb)

        return np.array(radar_constants)