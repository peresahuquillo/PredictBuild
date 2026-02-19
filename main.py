import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os

app = FastAPI()

# ===== CONFIGURACIÓN DE BASE DE DATOS =====
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-eu-west-1.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.txgjclpciwmivttxvkoy"),
    "password": os.getenv("DB_PASSWORD", "2@Vmhyr_UmC9*NLQKVbUM6u.N.mYfgyn"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# ===== CARGAR MODELOS =====
modelos = {}
# Carga todos los modelos al arrancar
for i in range(1, 13):
    with open(f"modelo_{i}.pkl", "rb") as f:
        modelos[f"modelo_{i}"] = pickle.load(f)

# ===== CARGAR DICCIONARIOS DESDE BD =====
def cargar_diccionarios():
    """Carga todas las tablas diccionario desde Postgres"""
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Diccionario de tipo de vivienda
    tipos_df = pd.read_sql('SELECT tipo_vivienda, index FROM "LU_Tipo_Vivienda"', conn)
    tipos_vivienda = dict(zip(tipos_df['tipo_vivienda'], tipos_df['index']))
    
    # Diccionario de estado de obra
    estados_df = pd.read_sql('SELECT estado, valor FROM "LU_Estat_Immoble"', conn)
    estados_obra = dict(zip(estados_df['estado'], estados_df['valor']))
    
    # Tabla principal con precios
    precios_df = pd.read_sql('''
        SELECT ciudad, distrito, barrio, calle, 
               precio_m2_distrito, precio_m2_barrio, precio_m2_calle
        FROM "LU_Preu_m2_Districte_Barri_Carrer"
    ''', conn)
    
    # Mapeo distrito -> modelo y distrito -> grupo (distrito_valor)
    mapeo_df = pd.read_sql('SELECT grupo, ciudad, distrito, modelo FROM "LU_Modelos_Distrito"', conn)
    mapeo_modelos = dict(zip(mapeo_df['distrito'], mapeo_df['modelo']))
    distrito_valores = dict(zip(mapeo_df['distrito'], mapeo_df['grupo']))
    
    conn.close()
    
    return {
        'distrito_valores': distrito_valores,
        'tipos_vivienda': tipos_vivienda,
        'estados_obra': estados_obra,
        'precios_df': precios_df,
        'mapeo_modelos': mapeo_modelos
    }

# Cargar diccionarios al iniciar el servidor
print("Cargando diccionarios desde BD...")
diccionarios = cargar_diccionarios()
print("Diccionarios cargados correctamente")

# ===== MODELO DE DATOS DE ENTRADA =====
class ViviendaInput(BaseModel):
    ciudad: str
    distrito: str
    barrio: str
    calle: str
    tipo_vivienda: str
    exterior_interior: str
    metros_cuadrados: float
    habitaciones: int
    banos: int
    planta: int
    estado_obra: str
    terraza: bool
    balcon: bool
    ascensor: bool

# ===== ENDPOINT DE PREDICCIÓN =====
@app.post("/predecir")
def predecir_valor(vivienda: ViviendaInput):
    try:
        # 1. Seleccionar el modelo según el distrito
        distrito = vivienda.distrito
        nombre_modelo = diccionarios['mapeo_modelos'].get(distrito)
        
        if not nombre_modelo:
            raise HTTPException(status_code=400, detail=f"No hay modelo disponible para el distrito {distrito}")
        
        modelo = modelos.get(nombre_modelo)
        if not modelo:
            raise HTTPException(status_code=500, detail=f"Modelo {nombre_modelo} no encontrado")
        
        # 2. Mapear valores categóricos a numéricos
        distrito_valor = diccionarios['distrito_valores'].get(distrito, 0)
        tipo_vivienda_valor = diccionarios['tipos_vivienda'].get(vivienda.tipo_vivienda, 0)
        estado_obra_valor = diccionarios['estados_obra'].get(vivienda.estado_obra, 0)
        
        # 3. Obtener precios por m2 buscando en el DataFrame
        precios_df = diccionarios['precios_df']
        
        # Buscar la fila que coincida con ciudad, distrito, barrio y calle
        fila = precios_df[
            (precios_df['ciudad'] == vivienda.ciudad) &
            (precios_df['distrito'] == vivienda.distrito) &
            (precios_df['barrio'] == vivienda.barrio) &
            (precios_df['calle'] == vivienda.calle)
        ]
        
        if fila.empty:
            # Si no encuentra la combinación exacta, busca al menos por distrito
            fila = precios_df[precios_df['distrito'] == vivienda.distrito]
            if fila.empty:
                raise HTTPException(
                    status_code=400, 
                    detail=f"No se encontraron precios para {vivienda.distrito}"
                )
        
        # Tomar la primera fila si hay múltiples coincidencias
        precio_m2_distrito = float(fila.iloc[0]['precio_m2_distrito'])
        precio_m2_barrio = float(fila.iloc[0]['precio_m2_barrio'])
        precio_m2_calle = float(fila.iloc[0]['precio_m2_calle'])
        
        # 4. Preparar los datos en el orden correcto
        entrada = [[
            distrito_valor,
            vivienda.metros_cuadrados,
            precio_m2_distrito,
            precio_m2_barrio,
            precio_m2_calle,
            tipo_vivienda_valor,
            vivienda.habitaciones,
            vivienda.banos,
            vivienda.planta,
            1 if vivienda.terraza else 0,
            1 if vivienda.balcon else 0,
            1 if vivienda.ascensor else 0,
            estado_obra_valor
        ]]
        
        # 5. Hacer la predicción
        prediccion = modelo.predict(entrada)[0]
        
        # 6. Calcular rango (10% de margen)
        valor_central = prediccion
        valor_minimo = int(valor_central * 0.9)
        valor_maximo = int(valor_central * 1.1)
        
        return {
            "valor_minimo": valor_minimo,
            "valor_central": int(valor_central),
            "valor_maximo": valor_maximo,
            "modelo_usado": nombre_modelo
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

# ===== ENDPOINT PARA REFRESCAR DICCIONARIOS =====
@app.post("/refrescar-diccionarios")
def refrescar():
    """Endpoint para recargar los diccionarios sin reiniciar el servidor"""
    global diccionarios
    diccionarios = cargar_diccionarios()
    return {"mensaje": "Diccionarios actualizados correctamente"}

# ===== ENDPOINT DE HEALTH CHECK =====
@app.get("/")
def health_check():
    return {"status": "ok", "modelos_cargados": len(modelos)}
