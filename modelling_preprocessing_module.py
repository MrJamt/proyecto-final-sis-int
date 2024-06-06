# archivo: preprocesamiento.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def cargar_datos(ruta_csv):
    data = pd.read_csv(ruta_csv)
    return data


def eliminar_columnas(data, columnas):
    return data.drop(columns=columnas)


def manejar_valores_faltantes(data, columna, valor):
    data[columna] = data[columna].fillna(valor)
    return data


def separar_caracteristicas(data, caracteristicas_numericas, caracteristicas_categoricas):
    numeric_data = data[caracteristicas_numericas]
    categorical_data = data[caracteristicas_categoricas]
    return numeric_data, categorical_data


def normalizar_datos(numeric_data):
    scaler = StandardScaler()
    numeric_data_norm = scaler.fit_transform(numeric_data)
    numeric_data_norm = pd.DataFrame(numeric_data_norm, columns=numeric_data.columns)
    return numeric_data_norm


def codificar_datos(categorical_data):
    le = LabelEncoder()
    categorical_data_norm = categorical_data.apply(le.fit_transform)
    return categorical_data_norm


def codificar_objetivo(data, columna_objetivo):
    y = data[columna_objetivo].apply(lambda x: 1 if x == 'satisfied' else 0)
    return y


def preprocesar_datos(ruta_csv):
    caracteristicas_numericas = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                                 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
                                 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
                                 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
                                 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    caracteristicas_categoricas = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

    data = cargar_datos(ruta_csv)
    data = eliminar_columnas(data, ['id'])
    data = manejar_valores_faltantes(data, 'Arrival Delay in Minutes', 0)
    numeric_data, categorical_data = separar_caracteristicas(data, caracteristicas_numericas,
                                                             caracteristicas_categoricas)
    numeric_data_norm = normalizar_datos(numeric_data)
    categorical_data_norm = codificar_datos(categorical_data)
    y = codificar_objetivo(data, 'satisfaction')

    data_preprocesado = pd.concat([numeric_data_norm, categorical_data_norm], axis=1)
    return data_preprocesado, y

