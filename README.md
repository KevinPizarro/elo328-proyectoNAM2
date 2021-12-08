# Proyecto NAM2

Proyecto NAM Longevidad Adulto Mayor 2: Análisis de técnicas de monitoreo de Respiración y Saturación de Oxígeno.

# Opcion 1 : Obtencion de la curva de respiracion y saturacion de oxigeno a traves de PPG
	-Comentar-
	- Trabajo realizado por Johanny Espinoza, Victor Cortés y Rudolf Hartmann.

# Opcion 2 : Obtencion de la curva de respiracion y saturacion de oxigeno a traves de rPPG y SVR
	- Lo primero es obtener la region de interes en la mano, se plantea dejar la mano en un espacio en especifico delimitado.
	  Se le aplica segmentacion por treshholding a traves de Otsu y luego se extrae el valor promedio de cada canal RGB de la ROI.
	- Calculo de la saturacion de oxigeno a traves de ventanas de tiempo de 50 datos, a traves del metodo de RoR a partir de los canal RB
	- Se calcula el breathing rate en BPM a partir de la formación de pulso usando el algoritmo POS y ocupando la biblioteca HeartPy.
	- Se destaca que hay dependencia de el fondo utilizado y la luminosidad por la naturaleza del método de rPPG.
	- Trabajo realizado por Gustavo Silva y Kevin Pizarro.

# Link a [repositorio](https://gitlab.com/elo328/proyecto-nam2).