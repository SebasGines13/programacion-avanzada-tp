"""Carga, limpieza determinista y separación sin compartir direcciones."""

import logging
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ["superficie", "ambientes", "barrio"]
NUMERICAS = ["superficie", "ambientes"]
CATEGORICAS = ["barrio"]
LOGGER = logging.getLogger(__name__)


def normalizar_texto(valor):
    """Unifica espacios, mayúsculas y tildes, conservando valores faltantes."""
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip().upper().replace("NU?ÆEZ", "NUNEZ")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^A-Z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip() or np.nan


def cargar_y_limpiar(ruta):
    """Conserva el CSV original y devuelve datos preparados más una auditoría."""
    originales = pd.read_csv(ruta)
    requeridas = ["Direccion", "PropiedadS", "Dolares", "Ambientes", "Barrio"]
    faltan = set(requeridas) - set(originales.columns)
    if faltan:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltan)}")
    auditoria = {"filas_originales": len(originales)}
    datos = originales.drop_duplicates().copy()
    auditoria["duplicados_exactos_eliminados"] = len(originales) - len(datos)
    datos.index.name = "fila_fuente"
    datos = datos.reset_index().rename(columns={
        "Direccion": "direccion", "PropiedadS": "superficie",
        "Dolares": "precio_usd", "Ambientes": "ambientes", "Barrio": "barrio",
    })
    datos = datos[["fila_fuente", "direccion", "superficie", "precio_usd",
                   "ambientes", "barrio"]].copy()
    for columna in ["superficie", "precio_usd", "ambientes"]:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
        datos[columna] = datos[columna].replace([np.inf, -np.inf], np.nan)
    precio_valido = datos.precio_usd.notna() & (datos.precio_usd > 0)
    auditoria["precios_invalidos_eliminados"] = int((~precio_valido).sum())
    datos = datos.loc[precio_valido].copy()
    # Regla de plausibilidad explícita: no se reconstruye superficie con el precio.
    for columna, limites in [("superficie", (10, 1000)), ("ambientes", (1, 20))]:
        invalidos = datos[columna].notna() & ~datos[columna].between(*limites)
        auditoria[f"{columna}_invalidos_a_faltante"] = int(invalidos.sum())
        datos.loc[invalidos, columna] = np.nan
    datos["barrio"] = datos.barrio.map(normalizar_texto)
    grupos = datos.direccion.map(normalizar_texto)
    auditoria["direcciones_faltantes_eliminadas"] = int(grupos.isna().sum())
    # Sin dirección no podemos garantizar la separación por edificio/publicación.
    datos = datos.loc[grupos.notna()].copy()
    datos["grupo_direccion"] = grupos.loc[datos.index]
    auditoria["barrios_faltantes"] = int(datos.barrio.isna().sum())
    auditoria["filas_limpias"] = len(datos)
    auditoria["grupos_direccion"] = int(datos.grupo_direccion.nunique())
    LOGGER.info("Limpieza: %s", auditoria)
    if any(auditoria[k] for k in ["superficie_invalidos_a_faltante",
                                 "ambientes_invalidos_a_faltante"]):
        LOGGER.warning("Valores fuera de los límites declarados quedan como faltantes")
    return datos.reset_index(drop=True), auditoria


def separar_y_etiquetar(datos, cuantil=0.75, semilla=42, test_size=0.2):
    """Reserva grupos y calcula el umbral de precio sólo con entrenamiento."""
    if not 0 < cuantil < 1:
        raise ValueError("El cuantil debe estar entre 0 y 1")
    separador = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=semilla)
    train_idx, test_idx = next(separador.split(datos, groups=datos.grupo_direccion))
    train = datos.iloc[train_idx].copy()
    test = datos.iloc[test_idx].copy()
    umbral = float(train.precio_usd.quantile(cuantil))
    for parte in [train, test]:
        parte["segmento_alto"] = (parte.precio_usd > umbral).astype(int)
        if parte.segmento_alto.nunique() != 2:
            raise ValueError("La partición no contiene ambas clases")
    compartidos = set(train.grupo_direccion) & set(test.grupo_direccion)
    if compartidos:
        raise ValueError("Se detectaron direcciones compartidas entre particiones")
    LOGGER.info("Separación: train=%s test=%s umbral_usd=%.2f",
                len(train), len(test), umbral)
    return train, test, umbral
