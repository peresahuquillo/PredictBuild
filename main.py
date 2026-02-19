import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI()

# ===== CONFIGURACIÓN DE BASE DE DATOS =====
# Cambia estos valores por los de tu BD
DB_CONFIG = {
    "host": "aws-1-eu-west-1.pooler.supabase.com",  # o la URL que tengas
    "database": "postgres",
    "user": "postgres.txgjclpciwmivttxvkoy",
    "password": "2@Vmhyr_UmC9*NLQKVbUM6u.N.mYfgyn",
    "port": 5432
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
    
    # Diccionario de distritos
    distritos_df = pd.read_sql("SELECT distrito, distrito_valor FROM tabla_distritos", conn)
    distritos = dict(zip(distritos_df['distrito'], distritos_df['distrito_valor']))
    
    # Diccionario de tipo de vivienda
    tipos_df = pd.read_sql("SELECT tipo, tipo_valor FROM tabla_tipos_vivienda", conn)
    tipos_vivienda = dict(zip(tipos_df['tipo'], tipos_df['tipo_valor']))
    
    # Diccionario de estado de obra
    estados_df = pd.read_sql("SELECT estado, estado_valor FROM tabla_estados_obra", conn)
    estados_obra = dict(zip(estados_df['estado'], estados_df['estado_valor']))
    
    # Precios por m2 - distrito
    precios_distrito_df = pd.read_sql("SELECT distrito, precio_m2 FROM tabla_precios_distrito", conn)
    precios_distrito = dict(zip(precios_distrito_df['distrito'], precios_distrito_df['precio_m2']))
    
    # Precios por m2 - barrio
    precios_barrio_df = pd.read_sql("SELECT barrio, precio_m2 FROM tabla_precios_barrio", conn)
    precios_barrio = dict(zip(precios_barrio_df['barrio'], precios_barrio_df['precio_m2']))
    
    # Precios por m2 - calle
    precios_calle_df = pd.read_sql("SELECT calle, precio_m2 FROM tabla_precios_calle", conn)
    precios_calle = dict(zip(precios_calle_df['calle'], precios_calle_df['precio_m2']))
    
    # Mapeo distrito -> modelo
    mapeo_df = pd.read_sql("SELECT distrito, modelo FROM tabla_mapeo_modelos", conn)
    mapeo_modelos = dict(zip(mapeo_df['distrito'], mapeo_df['modelo']))
    
    conn.close()
    
    return {
        'distritos': distritos,
        'tipos_vivienda': tipos_vivienda,
        'estados_obra': estados_obra,
        'precios_distrito': precios_distrito,
        'precios_barrio': precios_barrio,
        'precios_calle': precios_calle,
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
        distrito_valor = diccionarios['distritos'].get(distrito, 0)
        tipo_vivienda_valor = diccionarios['tipos_vivienda'].get(vivienda.tipo_vivienda, 0)
        estado_obra_valor = diccionarios['estados_obra'].get(vivienda.estado_obra, 0)
        
        # 3. Obtener precios por m2
        precio_m2_distrito = diccionarios['precios_distrito'].get(distrito, 0)
        precio_m2_barrio = diccionarios['precios_barrio'].get(vivienda.barrio, 0)
        precio_m2_calle = diccionarios['precios_calle'].get(vivienda.calle, 0)
        
        # 4. Preparar los datos en el orden correcto (según la imagen que compartiste)
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
        
        # 6. Calcular rango, Alejandro nos dijo el 10%
        valor_central = prediccion
        valor_minimo = int(valor_central * 0.9) #10% de margen
        valor_maximo = int(valor_central * 1.1) #10% de margen
        
        return {
            "valor_minimo": valor_minimo,
            "valor_central": int(valor_central),
            "valor_maximo": valor_maximo,
            "modelo_usado": nombre_modelo #Validar que estamos ejecutando el modelo correcto. En un futuro va fuera.
        }
        
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
```

---

## Ajustes que tienes que hacer:

1. **Cambiar los nombres de las tablas y columnas** en las queries SQL según cómo las tengas en tu BD (líneas donde dice `SELECT ... FROM ...`)

