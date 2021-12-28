import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os, subprocess
#como modo prueba se usara:
from holamundo import hola
#Se debe importar los modulos para poder ser utilizados en la interfaz
#import NAM2_o2_v7
#import spO2
#import ppg_fft
    
def centrar(i):
    ancho_ventana = 800
    alto_ventana = 800
    x_ventana = i.winfo_screenwidth() // 2 - ancho_ventana // 2
    y_ventana = i.winfo_screenheight() // 2 - alto_ventana // 2
    posicion = str(ancho_ventana) + "x" + str(alto_ventana) + "+" + str(x_ventana) + "+" + str(y_ventana)
    i.geometry(posicion)

def ventanaPPG():
    ventanaPPG= tk.Toplevel(window)
    centrar(ventanaPPG)
    ventanaPPG.title("NAM 2 Longevidad Adulto Mayor: Rostro y PPG")
    ventanaPPG.geometry("800x800")
    ventanaPPG.resizable(width=False, height=False)
    ventanaPPG.iconphoto(False, tk.PhotoImage(file='logo.png'))
    img=ImageTk.PhotoImage(Image.open("fondoppg.png"))
    Imagen= tk.Label(ventanaPPG, image= img).pack(side = "bottom", fill = "both", expand = "yes")
    #Boton de menú
    bot1=Image.open("menu.png")
    bot1= ImageTk.PhotoImage(bot1)
    boton1=tk.Button(ventanaPPG, image=bot1, command=window.deiconify)
    boton1.place(x=130,y=700)
    boton1.pack
    #Boton para detener
    bot2=Image.open("detener.png")
    bot2= ImageTk.PhotoImage(bot2)
    boton2=tk.Button(ventanaPPG, image=bot2, command=ventanaPPG)
    boton2.place(x=130,y=170)
    boton2.pack
    #Boton para iniciar
    bot3=Image.open("iniciar.png")
    bot3= ImageTk.PhotoImage(bot3)
    boton3=tk.Button(ventanaPPG, image=bot3, command=hola) #en tk.Button(ventanaSVR, image=bot3, command=hola ) se debe implementar la funcion de medicion por PPG
    #forma correcta seria
    #boton3=tk.Button(ventanaSVR, image=bot3, command= ppg_fft)
    #boton3=tk.Button(ventanaSVR, image=bot3, command= spO2)
    boton3.place(x=130,y=70)
    boton3.pack
    #Boton para cerrar
    bot4=Image.open("cerrar.png")
    bot4= ImageTk.PhotoImage(bot4)
    boton4=tk.Button(ventanaPPG, image=bot4, command=window.destroy)
    boton4.place(x=500,y=700)
    boton4.pack

    ventanaPPG.mainloop()
    
def ventanaSVR():
    ventanaSVR= tk.Toplevel()
    window.withdraw
    centrar(ventanaSVR)
    ventanaSVR.title("NAM 2 Longevidad Adulto Mayor: Palma y SVR")
    ventanaSVR.geometry("800x800")
    ventanaSVR.resizable(width=False, height=False)
    ventanaSVR.iconphoto(False, tk.PhotoImage(file='logo.png'))
    img=ImageTk.PhotoImage(Image.open("fondosvr.png"))
    Imagen= tk.Label(ventanaSVR, image= img).pack(side = "bottom", fill = "both", expand = "yes")
    #Boton de menú
    bot1=Image.open("menu.png")
    bot1= ImageTk.PhotoImage(bot1)
    boton1=tk.Button(ventanaSVR, image=bot1, command=window.deiconify)
    boton1.place(x=130,y=700)
    boton1.pack
    #Boton para detener
    bot2=Image.open("detener.png")
    bot2= ImageTk.PhotoImage(bot2)
    boton2=tk.Button(ventanaSVR, image=bot2, command=ventanaSVR)
    boton2.place(x=130,y=170)
    boton2.pack
    #Boton para iniciar
    bot3=Image.open("iniciar.png")
    bot3= ImageTk.PhotoImage(bot3)
    boton3=tk.Button(ventanaSVR, image=bot3, command=hola) #en tk.Button(ventanaSVR, image=bot3, command=hola) se debe implementar la funcion de medicion por SVR
    #forma correcta seria
    # boton3=tk.Button(ventanaSVR, image=bot3, NAM2_o2_v7 )
    boton3.place(x=130,y=70)
    boton3.pack
    #Boton para cerrar
    bot4=Image.open("cerrar.png")
    bot4= ImageTk.PhotoImage(bot4)
    boton4=tk.Button(ventanaSVR, image=bot4, command=window.destroy)
    boton4.place(x=500,y=700)
    boton4.pack

    ventanaSVR.mainloop()

def ventana2():
    window2= tk.Toplevel(window)
    window.withdraw()
    centrar(window2)
    window2.title("NAM 2 Longevidad Adulto Mayor")
    window2.geometry("800x800")
    window2.resizable(width=False, height=False)
    window2.iconphoto(False, tk.PhotoImage(file='logo.png'))
    img=ImageTk.PhotoImage(Image.open("fondomedio22.png"))
    Imagen= tk.Label(window2, image= img).pack(side = "bottom", fill = "both", expand = "yes")
    #Boton para opcion 1
    bot1=Image.open("botonSVR.png")
    bot1= ImageTk.PhotoImage(bot1)
    boton1=tk.Button(window2, image=bot1, command=ventanaSVR)
    boton1.place(x=130,y=600)
    boton1.pack
    #Boton para opcion 2
    bot2=Image.open("botonPPG.png")
    bot2= ImageTk.PhotoImage(bot2)
    boton2=tk.Button(window2, image=bot2, command=ventanaPPG)
    boton2.place(x=500,y=600)
    boton2.pack
    window2.mainloop()

window= Tk()
window.title("NAM 2 Longevidad Adulto Mayor")
window.geometry("800x800")
window.resizable(width=False, height=False)
window.iconphoto(False, tk.PhotoImage(file='logo.png'))
centrar(window)
#Fondo de la ventana
img=ImageTk.PhotoImage(Image.open("fondo.png"))
Imagen= tk.Label(window, image= img).pack(side = "bottom", fill = "both", expand = "yes")
#Agregar boton
botimg=Image.open("boton1.png")
botimg= ImageTk.PhotoImage(botimg)
boton=tk.Button(window, image=botimg, command=ventana2)
boton.place(x=300,y=650)
boton.pack

window.mainloop()




