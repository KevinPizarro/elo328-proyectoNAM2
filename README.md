# Proyecto NAM2

Proyecto NAM Longevidad Adulto Mayor 2: Análisis de técnicas de monitoreo de Respiración y Saturación de Oxígeno.

# Opcion 1 : Obtencion de la curva de respiracion y saturacion de oxigeno a traves de PPG
	- Lo primero es obtener la region del rostro, para despues obtener la region de la frente y de la boca.

# Opcion 2 : Obtencion de la curva de respiracion y saturacion de oxigeno a traves de rPPG y SVR
	- Lo primero es obtener la region de interes en la mano, se plantea dejar la mano en un espacio en especifico delimitado.
	  Se le aplica segmentacion por treshholding a traves de Otsu y luego se extrae el valor promedio de cada canal RGB de la ROI.
	- Calculo de la saturacion de oxigeno a traves de ventanas de tiempo de 50 datos, a traves del metodo de RoR a partir de los canal RB
