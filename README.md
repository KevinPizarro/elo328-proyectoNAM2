# Proyecto NAM2

Proyecto NAM Longevidad Adulto Mayor 2: Análisis de técnicas de monitoreo de Respiración y Saturación de Oxígeno.

# Opcion 1 : Obtencion de la curva de respiracion y saturacion de oxigeno a traves de PPG - ROI Rostro
	1. Proceso de obtencion de Region de interes (Rostro, Frente y boca)
	- Lo primero fue cargar los clasificadores Haarcascade de los .xml para rostro, ojos y boca.
	- Se pasa el frame de BGR a escala de grises.
	- Se aplica el clasificador cargado para reconocer rostro.
	- Dentro de la imagen del rostro reconocido se aplica el clasificador para ojos.
	- Con algunos cálculos matemáticos se obtiene la region sobre los ojos, es decir la frente.
	- Dentro de la misma imagen del rostro se aplica el clasificador para boca.
	- Con algunos cálculos matemáticos se obtiene la región de la boca reconocida solo en la mitad inferior del rostro.
	
	2.  Proceso de análisis del ROI


	- Lo primero es pasar el RGB a escala de grises
	- Se calcula la media aritmética de todos los pixeles de la ROI detectada.
	- Se empiezan a tomar de 25 datos de imagen por segunda a procesar para generar la curva de respiración.
	- Se le aplica la fft a los datos obtenidos de tal forma de obtener la frecuencia.
	- Se calcula los bpm y se muestran en la interfaz, en caso de encontrar inconsistencia se manda 0.
	
	- Trabajo realizado por Johanny Espinoza, Victor Cortés y Rudolf Hartmann.

# Opcion 2 : Obtencion de la curva de respiracion y saturacion de oxigeno a traves de rPPG y SVR - ROI Mano
	- Lo primero es obtener la region de interes en la mano, se plantea dejar la mano en un espacio en especifico delimitado.
	  Se le aplica segmentacion por treshholding a traves de Otsu y luego se extrae el valor promedio de cada canal RGB de la ROI.
	- Calculo de la saturacion de oxigeno a traves de ventanas de tiempo de 50 datos, a traves del metodo de RoR a partir de los canal RB
	- Se calcula el breathing rate en BPM a partir de la formación de pulso usando el algoritmo POS y ocupando la biblioteca HeartPy.
	- Se destaca que hay dependencia de el fondo utilizado y la luminosidad por la naturaleza del método de rPPG.
	- Trabajo realizado por Gustavo Silva y Kevin Pizarro.

# Link a [repositorio](https://gitlab.com/elo328/proyecto-nam2).
