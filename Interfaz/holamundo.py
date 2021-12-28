import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os, subprocess

def hola():
    
    wi= Tk()
    ancho_ventana = 700
    alto_ventana = 400
    x_ventana = wi.winfo_screenwidth() // 2 - ancho_ventana // 2
    y_ventana = wi.winfo_screenheight() // 2 - alto_ventana // 2
    posicion = str(ancho_ventana) + "x" + str(alto_ventana) + "+" + str(x_ventana) + "+" + str(y_ventana+100)
    wi.geometry(posicion)
    wi.title("NAM 2 Longevidad Adulto Mayor")
    wi.geometry("700x400")
    wi.resizable(width=False, height=False)
    wi.iconphoto(False, tk.PhotoImage(file='logo.png'))
    
