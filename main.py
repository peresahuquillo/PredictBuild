import joblib
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
    "password": os.getenv("DB_PASSWORD"),  # <- sin default
    "port": int(os.getenv("DB_PORT", 5432))
}

modelos = {}
diccionarios = None
startup_error = None

@app.on_event("startup")
def startup():
    global modelos, diccionarios, startup_error
    try:
        if not DB_CONFIG["password"]:
            raise RuntimeError("DB_PASSWORD no está configurada en Render (Environment Variables).")

        # 1) Cargar modelos
        for i in range(1, 13):
            path = f"modelo_{i}.pkl"  
            try:
                size = os.path.getsize(path)
                print(f"[STARTUP] Loading {path} size={size}")
                modelos[f"modelo_{i}"] = joblib.load(path)
                print(f"[STARTUP] OK {path}")
            except Exception as e:
                raise RuntimeError(f"Fallo cargando {path}: {e}")
        print("[DEBUG] modelo_1 n_features_in_:", modelos["modelo_1"].n_features_in_)
        print("[DEBUG] feature_names_in_:", getattr(modelos["modelo_1"], "feature_names_in_", None))
        
        # 2) Cargar diccionarios desde BD
        print("Cargando diccionarios desde BD...")
        diccionarios = cargar_diccionarios()
        print("Diccionarios cargados correctamente")

    except Exception:
        import traceback
        startup_error = traceback.format_exc()
        print("[STARTUP ERROR]")
        print(startup_error)


# ===== CARGAR DICCIONARIOS DESDE BD =====
def cargar_diccionarios():
    """Carga todas las tablas diccionario desde Postgres"""
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Diccionario de tipo de vivienda
    tipos_df = pd.read_sql('SELECT tipo_vivienda, index FROM core."LU_Tipo_Vivienda"', conn)
    tipos_vivienda = dict(zip(tipos_df['tipo_vivienda'], tipos_df['index']))
    
    # Diccionario de estado de obra
    estados_df = pd.read_sql('SELECT estado, valor FROM core."LU_Estat_Immoble"', conn)
    estados_obra = dict(zip(estados_df['estado'], estados_df['valor']))
    
    # Tabla principal con precios
    precios_df = pd.read_sql('''
        SELECT ciudad, distrito, barrio, calle, 
               precio_m2_distrito, precio_m2_barrio, precio_m2_calle
        FROM core."LU_Preu_m2_Districte_Barri_Carrer"
    ''', conn)
    
    # Mapeo distrito -> modelo y distrito -> grupo (distrito_valor)
    mapeo_df = pd.read_sql('SELECT grupo, ciudad, distrito, modelo FROM core."LU_Modelos_Distrito"', conn)
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
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=f"Startup failed: {startup_error[:300]}")

    if diccionarios is None or not modelos:
        raise HTTPException(status_code=503, detail="Servicio no listo: modelos/diccionarios no cargados todavía")

    try:
        # 1. Seleccionar el modelo según el distrito
        distrito = vivienda.distrito
        nombre_modelo = diccionarios["mapeo_modelos"].get(distrito)
        
        if not nombre_modelo:
            raise HTTPException(
                status_code=400,
                detail=f"No hay modelo disponible para el distrito {distrito}"
            )
        
        # En la BD viene como "modelo_6.pkl" -> lo normalizamos a "modelo_6"
        nombre_modelo_key = os.path.splitext(nombre_modelo)[0]
        
        modelo = modelos.get(nombre_modelo_key)
        if not modelo:
            raise HTTPException(
                status_code=500,
                detail=f"Modelo {nombre_modelo} no encontrado (key={nombre_modelo_key})"
            )
            
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
    global diccionarios, startup_error
    try:
        diccionarios = cargar_diccionarios()
        startup_error = None
        return {"mensaje": "Diccionarios actualizados correctamente"}
    except Exception as e:
        import traceback
        startup_error = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===== ENDPOINT DE HEALTH CHECK =====
@app.get("/")
def health_check():
    return {
        "status": "ok" if startup_error is None else "degraded",
        "modelos_cargados": len(modelos),
        "diccionarios_cargados": diccionarios is not None,
        "startup_error": startup_error if startup_error else None
    }
