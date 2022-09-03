"""
Regresion Linear
    Implementacion del algoritmo de regresion lineal.
    
Autor:
    Alejandro Domi­nguez Lugo
    A01378028
    
Fecha:
    01 de septiembre de 2022
    
"""

#----------------------------------Libreri­as-----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#-----------------------------Variables Globales-------------------------------
# Errores de la evaluación de cada nuevo grupo de parametros
errores = []
# Factor de escalamiento para los valores de entrada
factores = []
    
#------------------------------------Main--------------------------------------
def hyp(parametros, X):
    """
    hyp:
        Realiza la suma de resultados de la evaluación de la funcion de
        hipotesis para los parametros actuales.
        
    Argumentos:
        parametros (lst):
            Contiene los parametros para cada elemento de X.
        
        X (lst):
            Contiene los valores de entrada.
    
    Return:
        Suma de la evaluacion de la hipotesis.
        
    """
    
    suma = 0
    
    # Evaluar hipotesis para cada valor de X
    for i in range(len(parametros)):
        suma += parametros[i] * X[i]
    
    return suma

def calc_error(parametros, X, Y):
    """
    calc_error:
        Calcula el error para cada nueva hipotesis.
    
    Argumentos:
        parametros (lst):
            Contiene los parametros para cada elemento de X.
        
        X (lst):
            Contiene los valores de entrada.
        
        Y (lst):
            Contiene los valores de salida
            
    Return:
        -
    
    """
    
    global errores
    suma = 0
    
    # Evalúa la hipotesis actual y calcula los errores entre los valores
    # resultantes de la hipotesis y los valores reales de salida.
    for i in range(len(X)):
        hypothesis = hyp(parametros, X[i])
        error = hypothesis - Y[i]
        suma += error ** 2
    
    media = suma/len(X)
    
    # Agrega errores a la lista de errores
    errores.append(media)
    
def descensoGradiente(parametros, X, Y, alfa):
    """
    descensoGradiente:
        Calcula los nuevos valores de los parametros con una aproximacion de 
        descenso por el gradiente.
        
    Argumentos:
        parametros (lst):
            Contiene los parametros para cada elemento de X.
        
        X (lst):
            Contiene los valores de entrada.
        
        Y (lst):
            Contiene los valores de salida
        
        alfa (float):
            Tasa de aprendizaje.
    
    Return:
        aux (lst):
            Nuevos parametros calculados
            
    """
    
    aux = list(parametros)
    
    # Hacer calculo de error de la hipotesis actual y hacer correccion de 
    # parametros.
    for j in range(len(parametros)):
        suma = 0
        
        for i in range(len(X)):
            error = hyp(parametros, X[i]) - Y[i]
            suma += error * X[i][j]
        
        aux[j] = parametros[j] - alfa * (1 / len(X)) * suma
    
    return aux

def escalar(X):
    """
    escalar:
        Escala los datos para que halla convergencia en el descenso por el 
        gradiente.
        
    Argumentos:
        X (lst):
            Contiene los valores de entrada.
    
    Return:
        X (lst):
            Valores de entrada escalados.
            
    """
    
    global factores
    suma = 0
    X = np.asarray(X).T.tolist()
    
    # Escalar valores de X
    for i in range(1, len(X)):
        for j in range(len(X)):
            suma += X[i][j]
            
        prom = suma / len(X[i])
        valMax = max(X[i])
        
        for k in range(len(X[i])):
            X[i][k] = (X[i][k] - prom) / valMax
        
        factores.append((prom, valMax))
    
    return np.asarray(X).T.tolist()

def prep(X):
    """
    prep:
        Prepara los valores de X para tener las dimensiones correctas
    
    Argumentos:
        X (lst):
            Contiene los valores de entrada.
    
    Return:
        X (lst):
            Valores de entrada redimensionados.
            
    """
    
    # Redimensionar y agregar columna para constante
    for i in range(len(X)):
    	if isinstance(X[i], list):
    		X[i]=  [1]+X[i]
    	else:
    		X[i]=  [1,X[i]]
    
    return X

def show():
    """
    show:
        Muestra el error del modelo a traves de las epocas
    
    Argumentos:
        -
    
    Return:
        -
        
    """
    fig, ax = plt.subplots()
    plt.plot(errores, color = "red", linewidth = 2.5, label = "Error")
    plt.axhline(y = 0, xmin = 0.045, xmax = 0.955, color = "green", 
                linewidth = 1.5, label = "Error cero")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Error")
    ax.set_title("Error por epoca")
    ax.legend()
    plt.show()

def fit(parametros, X, Y, alfa = 0.001, maxEpoch = 100):
    """
    fit:
        Realiza una regresion lineal
        
    Argumentos:
        parametros (lst):
            Contiene los parametros para cada elemento de X.
        
        X (lst):
            Contiene los valores de entrada.
        
        Y (lst):
            Contiene los valores de salida
        
        alfa (float) = 0.001:
            Tasa de aprendizaje.
        
        maxEpoch (int) = 100:
            Ciclos máximos para encontrar los parametros finales.
        
    Return:
        parametros (lst):
            Parametros finales de la regresion lineal

    """
    
    global errores
    global factores
    errores = []
    factores = []
    
    # Preparar valores de entrada
    X = escalar(prep(X))
    
    # Calcular parametros hasta encontrar finales o alcanzar maximo de epocas
    epochs = 0
    while epochs != maxEpoch:
        parametrosViejos = list(parametros)
        parametros = descensoGradiente(parametros, X, Y, alfa)
        calc_error(parametros, X, Y)
        epochs += 1
        print("Epoca: ", epochs)
        print("Parametros: ", parametrosViejos, 
              "Error: ", errores[epochs - 1])
        
        if(parametrosViejos == parametros):
            break
    
    show()
    
    print("Parametros finales: ", parametros, 
          "Error final: ", errores[epochs - 1])
    
    return parametros

#-----------------------------------Pruebas------------------------------------
# Descargar dataset
wine = pd.read_csv("/home/lex/Escritorio/MomentoRetroM2/wine.data")
wine.columns = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", 
                "Magnesium", "Total phenols", "Flavanoids", 
                "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", 
                "Hue", "OD280/OD315 of diluted wines", "Proline"]

#__________________________________Prueba 1____________________________________
# Entrada y salida propuestos
entrada = "Flavanoids"
salida = "Alcohol"

# Obtener valores
x = wine[entrada].values.tolist()
y = wine[salida].values.tolist()

X = np.array(x)
Y = np.array(y)

parametros = [0,0]

# Hacer regresion lineal
parametros = fit(parametros, x, y, maxEpoch=4000)

# Calcular valores de salida con funcion calculada
yCalc = ((X - factores[0][0])/factores[0][1]) * parametros[1] + parametros[0]

# Mostrar resultados de regresion
fig, ax = plt.subplots()
plt.scatter(X, Y, color = "#4085BF", marker = "H")
plt.plot(X, yCalc, color = "#40BFBA", linewidth = 2.5)
ax.set_xlabel(entrada)
ax.set_ylabel(salida)
ax.set_title("Alcohol por flavonoides")
ax.legend()
plt.show()

#__________________________________Prueba 2____________________________________
# Entrada y salida propuestos
entrada = "Proline"
salida = "Alcohol"

# Obtener valores
x = wine[entrada].values.tolist()
y = wine[salida].values.tolist()

X = np.array(x)
Y = np.array(y)

parametros = [0,0]

# Hacer regresion lineal
parametros = fit(parametros, x, y, maxEpoch=3000)

# Calcular valores de salida con funcion calculada
yCalc = ((X - factores[0][0])/factores[0][1]) * parametros[1] + parametros[0]

# Mostrar resultados de regresion
fig, ax = plt.subplots()
plt.scatter(X, Y, color = "#5C4EB1", marker = "H")
plt.plot(X, yCalc, color = "#8E4EB1", linewidth = 2.5)
ax.set_xlabel(entrada)
ax.set_ylabel(salida)
ax.set_title("Alcohol por prolina")
ax.legend()
plt.show()

#__________________________________Prueba 3____________________________________
# Entrada y salida propuestos
entradas = ["Flavanoids", "Proline"]
salida = "Alcohol"

# Obtener valores
x = wine[entradas].values.tolist()
y = wine[salida].values.tolist()

X = np.array(x)
Y = np.array(y)

parametros = [0, 0, 0]

# Hacer regresion lineal
parametros = fit(parametros, x, y, maxEpoch=2500)

# Calcular valores de salida con funcion calculada
yCalcA = ((X[:, 0] - factores[0][0])/factores[0][1]) * parametros[1] + parametros[0]
yCalcB = ((X[:, 1] - factores[1][0])/factores[1][1]) * parametros[2] + parametros[0]

# Mostrar resultados de regresion
fig, ax = plt.subplots()
plt.scatter(X[:, 0], Y, color = "#B54AB1", marker = "H")
plt.plot(X[:, 0], yCalcA, color = "#B54A7B", linewidth = 2.5)
ax.set_xlabel(entradas[0])
ax.set_ylabel(salida)
ax.set_title("Alcohol por flavonoides")
ax.legend()
plt.show()

fig, ax = plt.subplots()
plt.scatter(X[:, 1], Y, color = "#C23D40", marker = "H")
plt.plot(X[:, 1], yCalcB, color = "#C27D3D", linewidth = 2.5)
ax.set_xlabel(entradas[1])
ax.set_ylabel(salida)
ax.set_title("Alcohol por prolina")
ax.legend()
plt.show()