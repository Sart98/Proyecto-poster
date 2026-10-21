# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


print("hello world!")

#Estabelcemos la ruta principal mediante el siguiente comando

carpeta_base = Path(__file__).resolve().parents[2]

#Ahora, definimos la ruta de la carpeta de datos

carpeta_datos = carpeta_base / "datos"

#Ahora, definimos cada archivo de datos que vamos a utilizar en el proyecto

base_RI = carpeta_datos / "RI.csv"

base_tasa_de_cambio = carpeta_datos / "TC.csv"

#0. LIMPIEZA DE DATOS

#Ahora, empezamos con la limpieza de datos
#Primero, dado que pandas trabaja con el formato gringo (comas para miles, puntos para decimales), vamos a cambiar el formato de los datos de la base de datos de RI y TC

df_TC = pd.read_csv(
    'datos/TC.csv',
    sep = ";",  # Le dice a pandas: "el separador de columnas es el punto y coma"
    decimal=',',  # Le dice a pandas: "la coma es el separador decimal"
    thousands='.',  # Le dice a pandas: "el punto es el separador de miles"
)

#Ahora, cambiamos el nombre de las columnas para que sean más fáciles de manejar en el código

df_TC = df_TC.rename(
    columns={
        'Periodo(MMM DD, AAAA)': 'fecha',
        'Tasa Representativa del Mercado (TRM)': 'tasa',}
)

#Ahora, hacemos lo mismo con la base de datos de RI

df_RI = pd.read_csv(
    'datos/RI.csv',
    sep = ";",  # Le dice a pandas: "el separador de columnas es el punto y coma"
    decimal=',',  # Le dice a pandas: "la coma es el separador decimal"
    thousands='.',  # Le dice a pandas: "el punto es el separador de miles"
)

df_RI = df_RI.rename(
    columns={
        "Periodo(MMM, AAAA)": 'fecha',
        "Reservas internacionales netas": 'RIN'}
)

df_TC['fecha'] = pd.to_datetime(df_TC['fecha']) #Esto crea un indice interno, desde el cual juntaremos la base de datos
df_TC = df_TC.set_index('fecha').sort_index() #Sort_index() ordena el indice de menor a mayor, para que no haya problemas al hacer el resample
df_TC = df_TC.loc['1991-12-01':'2026-09-30'] #Filtra los datos para que solo empiece desde diciembre de 1991. La tasa de cambio empieza desde el 27 de noviembre del 91, pero dado que mensualizaremos, filtramos esos datos. Por su parte, RI empieza desde 1960, entonces se hace lo mismo

df_RI['fecha'] = pd.to_datetime(df_RI['fecha'])
df_RI = df_RI.set_index('fecha').sort_index()
df_RI = df_RI.loc['1991-12-01':] #Acá no ponemos limite superior, dado que las RI acaban en septiembre, en TC si lo hacemos dado que también tiene datos de octubre de 2026.


#Ahora, vamos a hacer un resample de la base de datos de TC para que tenga la misma frecuencia que la base de datos de RI
#La razón de porque se hace con la función last() (ultimo dato de cada mes) y no mean() (promedio mensual) es porque
#dado que las RI se digitan a final de mes, al usar el promedio, estariamos midiendo distintos cortes de tiempo, haciendo que sea inconsistente. Por eso se usa last()
df_TC = df_TC.resample('ME').last()

#Unimos ambos dataframes en uno solo, para poder crear el modelo. Esto se hace mediante el uso de un indice compartido: la fecha

df_RI_TC = pd.concat([df_RI, df_TC], axis=1)

#Aislamos la Tasa de cambio nominal como una serie, para así analizarla y sacarle la FAC

serie_TC = df_RI_TC['tasa']

print(serie_TC.describe()) #Datos estadísticos de la serie de tiempo

plt.figure(figsize=(10, 5))
plt.plot(serie_TC)
plt.title("Tasa de cambio nominal, 1991-2026")
plt.xlabel("fecha")
plt.ylabel("tasa")
plt.grid(True)
plt.show()

#Hacemos la FAC y FACP de la serie de tiempo de la tasa de cambio nominal

#Establecemos los ejes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_TC.dropna(),
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
)
axes[0].set_title('FAC - Tasa de Cambio Mensual')


plot_pacf(
    serie_TC.dropna(),
    lags=24,
    alpha=0.05,
    method='ywm',
    ax=axes[1],
)
axes[1].set_title('FACP - Tasa de Cambio Mensual')

# Mostrar la gráfica
plt.tight_layout()
plt.show()
