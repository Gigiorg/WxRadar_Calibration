import WeatherRadarCalibration
import os


#Ruta del directorio donde se encuentran los directorios por experimentos:
filepath = '/home/gibs/Documentos/radar_cal/SOPHy_calibracion_2024'

#Instancia del objeto de procesamiento para calibracion
ex_cal = WeatherRadarCalibration.WeatherRadarCalibration()

#Lista de paths por experimento
exps = [os.path.join(filepath, i) for i in sorted(os.listdir(filepath))]

#Array con listas de constantes obtenidas por experimento procesado (dB)
rcc_constant = ex_cal.calculateConstants(exps)

print(rcc_constant.mean())




