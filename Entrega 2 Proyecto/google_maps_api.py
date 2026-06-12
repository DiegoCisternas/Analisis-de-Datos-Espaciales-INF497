import requests
import pandas as pd
import time
from datetime import datetime

def extraer_datos_api(lat, lng, radio, categorias, api_key, nombre_comuna):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    locales_comuna = []
    
    for categoria in categorias:
        print(f"      -> Buscando {categoria}...")
        params = {
            'location': f"{lat},{lng}",
            'radius': radio,
            'keyword': categoria,
            'key': api_key
        }
        
        while True:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'OK':
                for place in data['results']:
                    locales_comuna.append({
                        'Comuna_Origen_Busqueda': nombre_comuna, # Útil para trazabilidad
                        'Categoria': categoria,
                        'Nombre': place.get('name'),
                        'Direccion': place.get('vicinity', 'Sin dirección'),
                        'Latitud': place['geometry']['location']['lat'],
                        'Longitud': place['geometry']['location']['lng']
                    })
            
                
                next_page_token = data.get('next_page_token')
                if next_page_token:
                    time.sleep(2) # Pausa obligatoria de Google
                    params = {'pagetoken': next_page_token, 'key': api_key}
                else:
                    break

            elif data.get('status') == 'ZERO_RESULTS':
                print(f"        [!] No hay {categoria} en este radio.")
                break
            else:
                print(f"        [ERROR GOOGLE]: {data.get('status')} - {data.get('error_message', 'Sin detalles')}")
                break
                
    return pd.DataFrame(locales_comuna)

# --- Configuración del Grid ---
MI_API_KEY = ''
radio_busqueda = 1500 # 1.5 km para escanear en detalle cada nodo
lista_categorias = ['botilleria', 'discoteca', 'estacion de metro', 'comisaria']

# Diccionario modificado: Top 5 comunas para capturar distintas realidades
nodos_comunas = {
    'Santiago': [
        (-33.4370, -70.6500), # P1: Plaza de Armas (Centro Cívico)
        (-33.4410, -70.6680), # P2: Sector Metro Cumming / Barrio Brasil (Oeste)
        (-33.4700, -70.6510), # P3: Sector Metro Franklin / Matadero (Sur)
        (-33.4530, -70.6530), # P4: Sector Parque Almagro / San Diego (Centro-Sur)
        (-33.4400, -70.6780)  # P5: Sector Matucana / Quinta Normal (Extremo Oeste)
    ],
    'Ñuñoa': [
        (-33.4530, -70.5940), # P1: Plaza Ñuñoa (Centro)
        (-33.4510, -70.6150), # P2: Sector Irarrázaval / Barrio Italia Sur (Oeste)
        (-33.4520, -70.5700), # P3: Sector Plaza Egaña (Este)
        (-33.4640, -70.6030), # P4: Sector Estadio Nacional / Av. Grecia (Sur-Oeste)
        (-33.4720, -70.5980)  # P5: Sector Macul con Rodrigo de Araya (Extremo Sur)
    ],
    'Maipú': [
        (-33.5099, -70.7565), # P1: Plaza de Maipú (Centro neurálgico)
        (-33.4815, -70.7523), # P2: Sector Mall Arauco Maipú / El Sol (Norte-Este)
        (-33.4910, -70.7490), # P3: Metro Las Parcelas / Pajaritos (Norte)
        (-33.5135, -70.7765), # P4: Sector Rinconada / Hospital El Carmen (Oeste)
        (-33.5350, -70.7800)  # P5: Sector Tres Poniente / Camino Melipilla (Sur)
    ],
    'Las Condes': [
        (-33.4150, -70.5840), # P1: Escuela Militar / Apoquindo (Oeste)
        (-33.4000, -70.5730), # P2: Parque Araucano / Rosario Norte (Centro-Norte)
        (-33.4070, -70.5400), # P3: Los Dominicos (Este)
        (-33.4210, -70.5480), # P4: Rotonda Atenas / Tomás Moro (Sur)
        (-33.3850, -70.5180)  # P5: Cantagallo / San Carlos de Apoquindo (Extremo Este)
    ],
    'Puente Alto': [
        (-33.6117, -70.5758), # P1: Plaza de Puente Alto (Centro)
        (-33.5960, -70.5800), # P2: Metro Las Mercedes (Norte-Centro)
        (-33.5820, -70.5830), # P3: Metro Protectora de la Infancia (Norte)
        (-33.6080, -70.6050), # P4: Sector Bajos de Mena (Oeste)
        (-33.5680, -70.5550)  # P5: Mall Plaza Tobalaba / Diego Portales (Extremo Norte-Este)
    ]
}

# --- Ejecución del Grid Search ---
dataframes_lista = []

print("Iniciando Grid Search Exhaustivo por Comunas...\n")

for comuna, puntos in nodos_comunas.items():
    print(f"Escaneando: {comuna}")
    # Ahora iteramos sobre los 5 puntos de cada comuna
    for i, (lat, lng) in enumerate(puntos):
        print(f"  -> Punto {i+1} de {len(puntos)} (Lat: {lat}, Lng: {lng})")
        df_temporal = extraer_datos_api(lat, lng, radio_busqueda, lista_categorias, MI_API_KEY, comuna)
        
        if not df_temporal.empty:
            dataframes_lista.append(df_temporal)
            print(f"      [OK] {len(df_temporal)} registros extraídos en este nodo.")
    print("-" * 40)

# --- Concatenación y Limpieza (Pandas) ---
print("\nProcesando y limpiando datos...")

# 1. Juntar todos los DataFrames en uno solo
df_maestro = pd.concat(dataframes_lista, ignore_index=True)
total_bruto = len(df_maestro)

# 2. Dropear duplicados
# Es vital aquí, ya que los radios de 1.5km se cruzarán intencionalmente
df_maestro_limpio = df_maestro.drop_duplicates(subset=['Nombre', 'Latitud', 'Longitud'], keep='first')
total_limpio = len(df_maestro_limpio)

duplicados_eliminados = total_bruto - total_limpio

# 3. Guardar el resultado final
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_archivo = f"dataset_grid_exhaustivo_{timestamp}.csv"
df_maestro_limpio.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')

# --- Resumen Final ---
print("\n--- RESUMEN DE EXTRACCIÓN ---")
print(f"Total registros brutos: {total_bruto}")
print(f"Duplicados eliminados (Zonas superpuestas): {duplicados_eliminados}")
print(f"Total registros únicos (Dataset Final): {total_limpio}")
print(f"Archivo guardado como: {nombre_archivo}")
print("\nDesglose por categoría (Datos limpios):")
print(df_maestro_limpio['Categoria'].value_counts())