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



# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(serie_TC.dropna(), regression='ct') # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(serie_TC.dropna(), regression="ct", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

#Segun los aspectos de los test, la serie no es estacionaria, por lo que se procede a aplicar logaritmo y luego diferenciarla 

serie_TC_log = np.log(serie_TC)

# Gráfica 
plt.figure(figsize=(10, 5))
plt.plot(serie_TC_log)
plt.title("Serie con logaritmo aplicado - Tasa de cambio nominal, 1991-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

#Ahora, aplicamos la diferenciación a la serie logarítmica para hacerla estacionaria

serie_TC_log_diff = serie_TC_log.diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(serie_TC_log_diff)
plt.title("Serie diferenciada del logaritmo - Tasa de cambio nominal, 1991-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

#Realizamos la prueba de estacionariedad ADF y KPSS a la serie diferenciada del logaritmo

# Test de Augmented Dickey Fuller (ADF)
adf_result2 = adfuller(serie_TC_log_diff.dropna(), regression='c') # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result2[0])
print("p-valor:", adf_result2[1])
print("Rezagos usados:", adf_result2[2]) 
print("Observaciones:", adf_result2[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result2[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result2 = kpss(serie_TC_log_diff.dropna(), regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result2[0])
print("p-valor:", kpss_result2[1])
print("Rezagos usados:", kpss_result2[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result2[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

#Ahora, para meter la variable exogena (reservas internacionales netas) debemos asegurarnos de que también sean estacionarias, para así evitar una regresión espuria. Por lo que se procede a hacer la misma prueba de estacionariedad a la serie de RI

#Primero, nos toca volver las RI una serie 

serie_RI = df_RI_TC['RIN']

print(serie_RI.describe()) #Datos estadísticos de la serie de tiempo

plt.figure(figsize=(10, 5))
plt.plot(serie_RI)
plt.title("Reservas internacionales netas, 1991-2026")
plt.xlabel("fecha")
plt.ylabel("tasa")
plt.grid(True)
plt.show()

#Realizamos el rest de raiz unitaria sobre la serie de RI, si no es estacionaria, procedemos a difernciar y aplicar logaritmo

adf_result3 = adfuller(serie_RI.dropna(), regression='ct') # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result3[0])
print("p-valor:", adf_result3[1])
print("Rezagos usados:", adf_result3[2]) 
print("Observaciones:", adf_result3[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result3[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result3 = kpss(serie_RI.dropna(), regression="ct", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result3[0])
print("p-valor:", kpss_result3[1])
print("Rezagos usados:", kpss_result3[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result3[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

#Dado que la serie de RI no es estacionaria, procedemos a aplicar logaritmo y diferenciación a la serie de RI

serie_RI_log = np.log(serie_RI)

plt.figure(figsize=(10, 5))
plt.plot(serie_RI_log)
plt.title("Serie con logaritmo aplicado - Reservas internacionales netas, 1991-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

serie_RI_log_diff = serie_RI_log.diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(serie_RI_log_diff)
plt.title("Serie con logaritmo y diferenciada - Reservas internacionales netas, 1991-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

#Ahora, realizamos la prueba de estacionariedad ADF y KPSS a la serie diferenciada del logaritmo de RI

adf_result4 = adfuller(serie_RI_log_diff.dropna(), regression='ct') # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result4[0])
print("p-valor:", adf_result4[1])
print("Rezagos usados:", adf_result4[2]) 
print("Observaciones:", adf_result4[3])
print("Valores críticos:")
for nivel, valor in adf_result4[4].items():
    print(f"{nivel}: {valor}")

if adf_result4[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result4 = kpss(serie_RI_log_diff.dropna(), regression="ct", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result4[0])
print("p-valor:", kpss_result4[1])
print("Rezagos usados:", kpss_result4[2])
print("Valores críticos:")
for nivel, valor in kpss_result4[3].items():
    print(f"{nivel}: {valor}")

if kpss_result4[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

#Dado que la serie no es estacionaria, no utilizaremos la serie de RI como variable exógena, dado que no es estacionaria. Por lo que se procede a hacer un modelo SARIMAX sin variables exógenas.

#Ahora, usaremos la FAC y la FACP de la serie diferenciada del logaritmo de la tasa de cambio nominal para determinar los parámetros p y q del modelo SARIMAX

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_TC_log_diff.dropna(),
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
)
axes[0].set_title('FAC - Tasa de Cambio Mensual diferenciada y logaritmada')


plot_pacf(
    serie_TC_log_diff.dropna(),
    lags=24,
    alpha=0.05,
    method='ywm',
    ax=axes[1],
)
axes[1].set_title('FACP - Tasa de Cambio Mensual diferenciada y logaritmada')

# Mostrar la gráfica
plt.tight_layout()
plt.show()

#Dado que la serie de tasa de cambio nominal tiene comportamiento de ruido blanco, cambiamos a una serie de tasa de cambio real, basada en el IPC

df_TCR = pd.read_csv(
    'datos/TCR.csv',
    sep = ";",  # Le dice a pandas: "el separador de columnas es el punto y coma"
    decimal=',',  # Le dice a pandas: "la coma es el separador decimal"
    thousands='.',  # Le dice a pandas: "el punto es el separador de miles"
)

#Ahora, cambiamos el nombre de las columnas para que sean más fáciles de manejar en el código

df_TCR = df_TCR.rename(
    columns={
        'Periodo(MMM DD, AAAA)': 'fecha',
        'Tasa Representativa del Mercado (TRM)': 'tasa',}
)

print(df_TCR.head()) #Imprimimos los primeros 5 datos de la base de datos para verificar que se haya cargado correctamente

