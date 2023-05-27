#!/usr/bin/env python
# coding: utf-8

# Importamos las librerías necesarias para proceder con el KNN 

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Cargamos el archivo que en este caso es Cleaned-data que fue descargado desde Kaggle

# In[120]:


df = pd.read_csv('C:\Prueba\Cleaned-Data.csv')


# Hacemos una pequeña prueba para comprobar que esté funcionando 

# In[121]:


df.head()


# Observamos la forma del marco de datos con el siguiente comando (df.shape) (Indica el número de filas y columnas que tiene)

# In[122]:


df.shape


# Ahora lo que hacemos es preparar los datos para entrenar un modelo de aprendizaje automático

# In[123]:


X = df.drop('Contact_Yes',axis=1).values
y = df['Contact_Yes'].values


# Ahora dividimos el conjunto de datos en conjunto de entrenamiento y prueba

# In[124]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42,)


# Ahora procedemos en configurar las matrices para almacenar las precisiones de enternamiento y prueba

# In[125]:


neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# Ahora generamos la gráfica

# In[126]:


plt.title('Gráfica sobre los contactos del COVID')
plt.plot(neighbors, test_accuracy, label='Precisión de datos de prueba')
plt.plot(neighbors, train_accuracy, label='"" de datos de entrenamiento')
plt.legend()
plt.xlabel('Números de neighbors')
plt.ylabel('Precisión')
plt.show()


# In[ ]:




