# PredictBuild

**PredictBuild** es una solución de valoración automatizada de inmuebles orientada al sector inmobiliario (Real Estate). El objetivo es ayudar a los agentes inmobiliarios a estimar el precio de venta de un piso de forma rápida y precisa, introduciendo los datos del inmueble en una aplicación y recibiendo al instante un precio estimado basado en modelos de Machine Learning.

---

## Contexto del Proyecto

Cuando un agente inmobiliario recibe un nuevo cliente interesado en vender su propiedad, necesita hacer una valoración del inmueble. PredictBuild digitaliza y automatiza este proceso: el agente introduce las características del inmueble en la APP, y el sistema devuelve un rango de precio estimado (mínimo, central y máximo).

El proyecto ha sido desarrollado como iniciativa emprendedora en el mercado de Madrid, con modelos específicos por barrio/distrito para maximizar la precisión de las predicciones.

---

## Arquitectura del Sistema

```
[Agente Inmobiliario]
        |
        v
[Front-end APP]  -- introduce datos del inmueble -->
        |
        v
[API REST - FastAPI en Render]
        |
        ├── Selecciona el modelo según el distrito
        ├── Consulta precios/m2 y diccionarios en PostgreSQL (Supabase)
        └── Ejecuta el modelo ML (.pkl)
        |
        v
[Respuesta: precio mínimo / central / máximo]
```

**Stack tecnológico:**
- **Base de datos:** PostgreSQL (Supabase / AWS eu-west-1)
- **API:** FastAPI desplegada en [Render](https://render.com)
- **Modelos ML:** scikit-learn 1.5.2, serializados con joblib (`.pkl`)
- **Lenguaje:** Python 3

---

## Modelos de Machine Learning

Hay **12 modelos** entrenados, uno por grupo de distritos de Madrid (`modelo_1.pkl` ... `modelo_12.pkl`). El mapeo entre distrito y modelo se gestiona dinámicamente desde la base de datos (`LU_Modelos_Distrito`), lo que permite añadir o reasignar modelos sin tocar el código.

El desarrollo de los modelos se documenta en tres notebooks:

| Notebook | Descripción |
|---|---|
| `1. Notebook_Madrid_Tractament_Dades.ipynb` | Ingesta, limpieza y transformación de datos |
| `2. Notebook_Madrid_EDA.ipynb` | Análisis Exploratorio de Datos (EDA) |
| `3. Notebook_Madrid_Models.ipynb` | Entrenamiento, evaluación y selección de modelos |

---

## API Endpoints

### `POST /predecir`

Recibe los datos de un inmueble y devuelve el precio estimado.

**Body (JSON):**
```json
{
  "ciudad": "Madrid",
  "distrito": "Salamanca",
  "barrio": "Goya",
  "calle": "Calle de Goya",
  "tipo_vivienda": "Piso",
  "exterior_interior": "Exterior",
  "metros_cuadrados": 90.0,
  "habitaciones": 3,
  "banos": 2,
  "planta": 4,
  "estado_obra": "Buen estado",
  "terraza": false,
  "balcon": true,
  "ascensor": true
}
```

**Respuesta:**
```json
{
  "valor_minimo": 450000,
  "valor_central": 500000,
  "valor_maximo": 550000,
  "modelo_usado": "modelo_6.pkl"
}
```

### `POST /refrescar-diccionarios`

Recarga los datos de los diccionarios y precios desde la base de datos sin reiniciar el servidor.

### `GET /`

Health check del servicio. Devuelve el estado del servidor, número de modelos cargados y si los diccionarios están disponibles.

---

## Tablas en Base de Datos (PostgreSQL)

| Tabla | Descripción |
|---|---|
| `core.LU_Tipo_Vivienda` | Codificación de tipos de vivienda |
| `core.LU_Estat_Immoble` | Codificación del estado de obra |
| `core.LU_Preu_m2_Districte_Barri_Carrer` | Precios por m2 a nivel de distrito, barrio y calle |
| `core.LU_Modelos_Distrito` | Mapeo distrito → modelo ML asignado |

---

## Despliegue

El servidor está desplegado en **Render**. En el arranque (`startup`), la aplicación:
1. Carga todos los modelos `.pkl` en memoria.
2. Conecta a PostgreSQL y carga los diccionarios necesarios para la inferencia.

La variable de entorno `DB_PASSWORD` debe estar configurada en Render para que el servicio arranque correctamente.

---

## Instalación local

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Requiere las variables de entorno: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`.
