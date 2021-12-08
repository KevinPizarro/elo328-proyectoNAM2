import numpy as np

test_value = 1546841

arreglo_prueba=np.array([10, 7, 4, 5, 6]) 

aux = arreglo_prueba[-1]
print('El valor de aux es: %s bpm' %aux)
print('breathing rate is: %s bpm' %arreglo_prueba)

arreglo_prueba[:-1] = arreglo_prueba[1:]## lo que se hace aca es correr todo el arreglo de derecha a izquierda un espacio y copiar el ultimo en la ultima posicion.
print('Despues del corrimiento: %s bpm' %arreglo_prueba)
arreglo_prueba[:-1] = aux

print('Despues del cambio: %s bpm' %arreglo_prueba)

print("Start")
camData = np.random.normal(size=50)
camSig = camData - np.nanmean(camData)
print('Valor de camData al crearlo con el random: %s bpm' %camData)
camData[-1] = test_value
print('Valor posterior a actualizacion de camData: %s bpm' %camData)
print('Valor de camSig: %s bpm' %camSig)

print("End")