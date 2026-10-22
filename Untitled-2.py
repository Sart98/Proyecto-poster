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
from statsmodels.tsa.ar_model import AutoReg


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
    sep = ",",  # Le dice a pandas: "el separador de columnas es el punto y coma"
    decimal='.',  # Le dice a pandas: "la coma es el separador decimal"
)

#Ahora, cambiamos el nombre de las columnas para que sean más fáciles de manejar en el código

df_TCR = df_TCR.rename(
    columns={
        'Fecha': 'fecha',
        'TCR': 'tasa',}
)

df_TCR['fecha'] = pd.to_datetime(df_TCR['fecha']) #Esto crea un indice interno
df_TCR = df_TCR.set_index('fecha').sort_index() #Sort_index() ordena el indice de menor a mayor
#Convertimos la columna de tasa en una serie.

serie_TCR = df_TCR['tasa']

print(serie_TCR.describe()) #Datos estadísticos de la serie de tiempo

plt.figure(figsize=(10, 5))
plt.plot(serie_TCR)
plt.title("Tasa de cambio real, 1986-2026")
plt.xlabel("fecha")
plt.ylabel("tasa")
plt.grid(True)
plt.show()

#Hacemos la FAC y FACP de la serie de tiempo de la tasa de cambio real

#Establecemos los ejes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_TCR.dropna(),
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
)
axes[0].set_title('FAC - Tasa de Cambio Real')


plot_pacf(
    serie_TCR.dropna(),
    lags=24,
    alpha=0.05,
    method='ywm',
    ax=axes[1],
)
axes[1].set_title('FACP - Tasa de Cambio Real')

# Mostrar la gráfica
plt.tight_layout()
plt.show()

#Ahora, realizamos las pruebas de raiz unitaria ADF y KPSS a la serie de TCR, para ver si es estacionaria o no. En caso de no serlo, se procede a aplicar logaritmo y diferenciación a la serie.

adf_result5 = adfuller(serie_TCR.dropna(), regression='c') # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result5[0])
print("p-valor:", adf_result5[1])
print("Rezagos usados:", adf_result5[2]) 
print("Observaciones:", adf_result5[3])
print("Valores críticos:")
for nivel, valor in adf_result5[4].items():
    print(f"{nivel}: {valor}")

if adf_result5[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result5 = kpss(serie_TCR.dropna(), regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result5[0])
print("p-valor:", kpss_result5[1])
print("Rezagos usados:", kpss_result5[2])
print("Valores críticos:")
for nivel, valor in kpss_result5[3].items():
    print(f"{nivel}: {valor}")

if kpss_result5[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

#Como no es estacionaria, procedemos a aplicar logaritmo y diferenciación a la serie de TCR

serie_TCR_log = np.log(serie_TCR)

plt.figure(figsize=(10, 5))
plt.plot(serie_TCR_log)
plt.title("Serie con logaritmo aplicado - Tasa de cambio real, 1986-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

serie_TCR_log_diff = serie_TCR_log.diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(serie_TCR_log_diff)
plt.title("Serie diferenciada y con logaritmo aplicado - Tasa de cambio real, 1986-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

#Ahora, realizamos la prueba de estacionariedad ADF y KPSS a la serie diferenciada del logaritmo de TCR

adf_result6 = adfuller(serie_TCR_log_diff.dropna(), regression='c') # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result6[0])
print("p-valor:", adf_result6[1])
print("Rezagos usados:", adf_result6[2]) 
print("Observaciones:", adf_result6[3])
print("Valores críticos:")
for nivel, valor in adf_result6[4].items():
    print(f"{nivel}: {valor}")

if adf_result6[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result6 = kpss(serie_TCR_log_diff.dropna(), regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result6[0])
print("p-valor:", kpss_result6[1])
print("Rezagos usados:", kpss_result6[2])
print("Valores críticos:")
for nivel, valor in kpss_result6[3].items():
    print(f"{nivel}: {valor}")

if kpss_result6[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

#Ahora, dado que es estacionario, realizamos la FAC y FACP de la serie diferenciada del logaritmo de TCR para determinar los parámetros p y q del modelo SARIMAX

#1 IDENTIFICACIÓN 

#Establecemos los ejes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie_TCR_log_diff.dropna(),
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0],
)
axes[0].set_title('FAC - Tasa de Cambio Real Diferenciada y Logaritmada')


plot_pacf(
    serie_TCR_log_diff.dropna(),
    lags=24,
    alpha=0.05,
    method='ywm',
    ax=axes[1],
)
axes[1].set_title('FACP - Tasa de Cambio Real Diferenciada y Logaritmada')

# Mostrar la gráfica
plt.tight_layout()
plt.show()

#Dado que tanto la FAC como la FACP muestran que solo la primera autocorrelación es significativa, se procede a hacer un modelo ARIMA (1,1,1)

#2. ESTIMACIÓN

#Se utiliza un modelo ARIMA (1,1,1)

modelo1_ARIMA = SARIMAX(
    serie_TCR_log,
    order=(1, 1, 1),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

Resultados_ARIMA = modelo1_ARIMA.fit(disp=False)

print(Resultados_ARIMA.summary())

#Dado que el modelo ARIMA (1,1,1) no resultvo significativo en sus parametros, intentamos un ARIMA (0,1,1)

modelo1_MA = SARIMAX(
    serie_TCR_log,
    order=(0, 1, 1),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

Resultados_MA = modelo1_MA.fit(disp=False)

print(Resultados_MA.summary())

#Ahora, probaremos un ARIMA(1,1,0) para ver si es mejor que el ARIMA(0,1,1)

Modelo_AR = SARIMAX(
    serie_TCR_log,
    order=(1, 1, 0),
    trend="n",
    enforce_stationarity=False,
    enforce_invertibility=False
)

Resultados_AR = Modelo_AR.fit(disp=False)

print(Resultados_AR.summary())




#Aunque ambos modelos ARIMA(0,1,1) y ARIMA(1,1,0) son significativos, el ARIMA(1,1,0) tiene un AIC y BIC más bajos, por lo que se procede a hacer la validación de supuestos con este modelo

#3. VALIDACIÓN DE SUPUESTOS

# Residuales del modelo
residuales = Resultados_AR.resid.dropna()

# Los residuales son un objeto tipo Pandas.Series
type(residuales)

# Gráfica de los residuales (deberían comportarse como ruido blanco)
plt.figure(figsize=(10, 5))
plt.plot(residuales.iloc[1:-1])
plt.title("Residuales del modelo AR(1)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Descripción de los residuales 
print(residuales.describe())

# Encontrar la fecha del valor máximo en tus residuos
fecha_maxima = residuales.idxmax()
print("Fecha del choque extremo:", fecha_maxima)

#Este primer dato atipico de residuo ocurre por un error de estimación, y no es algo atado en si al modelo.
#Esto se conoce como efecto de inicialización del modelo

ljung_box = acorr_ljungbox(
    residuales,
    lags=[6, 12, 18, 24],
    return_df=True
)

print("Prueba Ljung-Box")
print(ljung_box)
#Ninguno de los p-value salio menor a 0.05, por lo que no hay autocorrelación serial con los residuos.

#Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales, nlags=12) 

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

#Como se puede ver, existe heterocedasticidad en los residuos, más esto se puede atribuir a la naturaleza de la serie de tiempo, dado que la tasa de cambio real es una serie que presenta cambios abruptos, por lo que no es raro que los residuos presenten heterocedasticidad. Sin embargo, esto no afecta la validez del modelo, dado que los residuos no presentan autocorrelación serial.

# Prueba de normalidad Jarque-Bera

jb_test = jarque_bera(residuales.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

#Como se puede ver, los residuos no presentan normalidad, sin embargo, esto puede ser atribuido al efecto de inicialización del modelo, dado que el primer residuo es un valor atípico. Por lo que se procede a eliminar el primer residuo y volver a hacer la prueba de normalidad.
residuales_sin_primer = residuales.iloc[1:]
jb_test_sin_primer = jarque_bera(residuales_sin_primer)


print("Prueba Jarque-Bera (sin primer residuo)")
print("Jarque-Bera statistic:", jb_test_sin_primer.statistic)
print("Jarque-Bera p-value:", jb_test_sin_primer.pvalue)


#Como se puede ver, los residuos no presentan normalidad, sin embargo, esto puede ser atribuido a la naturaleza de la serie de tiempo, dado que la tasa de cambio real es una serie que presenta cambios abruptos, por lo que no es raro que los residuos no presenten normalidad. Sin embargo, esto no afecta la validez del modelo, dado que los residuos no presentan autocorrelación serial ni heterocedasticidad.

#4. PRONÓSTICO
#Dado que los errores presentan anormalidad, utilizaremos bootstrapping para realizar los pronosticos
p_lags = 12  
model = AutoReg(serie_TCR, lags=p_lags).fit()

#Cambiamos los residuos y los parámetros del modelo a variables para poder usarlas en el pronóstico
params = model.params.values  

residuals = model.resid.dropna()
history = serie_TCR.values[-p_lags:]

# ==========================================
# 2. SIMULACIÓN BOOTSTRAP DE RESIDUOS
# ==========================================
np.random.seed(42)
n_simulations = 1000  # Número de trayectorias simuladas
h_horizon = 10        # Horizonte de pronóstico (ej. 24 períodos hacia adelante)

forecasts = np.zeros((h_horizon, n_simulations))

for i in range(n_simulations):
    # Remuestreo con reemplazo de los residuos históricos
    boot_residuals = np.random.choice(residuals, size=h_horizon, replace=True)
    
    simulated_path = []
    current_history = list(history)
    
    for step in range(h_horizon):
        lag_values = current_history[-p_lags:][::-1]
        # Ecuación del modelo AR + residuo simulado
        pred = params[0] + np.dot(params[1:], lag_values) + boot_residuals[step]
        simulated_path.append(pred)
        current_history.append(pred)
        
    forecasts[:, i] = simulated_path

# ==========================================
# 3. CÁLCULO DE INTERVALOS Y GRÁFICO
# ==========================================
# Generar fechas futuras 
future_dates = pd.date_range(start=serie_TCR.index[-1], periods=h_horizon + 1, freq='MS')[1:]

forecast_median = np.median(forecasts, axis=1)
forecast_lower = np.percentile(forecasts, 5, axis=1)   # Percentil 5% (Límite inferior)
forecast_upper = np.percentile(forecasts, 95, axis=1)  # Percentil 95% (Límite superior)

plt.figure(figsize=(12, 6))
plt.plot(serie_TCR.index[-60:], serie_TCR.values[-60:], label='TCR Histórica', color='black')
plt.plot(future_dates, forecast_median, label='Pronóstico Bootstrap (Mediana)', color='blue', linestyle='--')
plt.fill_between(future_dates, forecast_lower, forecast_upper, color='blue', alpha=0.2, label='IC 90% Bootstrap')

plt.title('Pronóstico Bootstrap de la Tasa de Cambio Real (TCR) de Colombia')
plt.xlabel('Fecha')
plt.ylabel('TCR')
plt.legend()
plt.grid(True)
plt.show()