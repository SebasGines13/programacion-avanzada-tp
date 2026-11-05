"""Orquestación reproducible del proyecto, compartida por CLI y notebook."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import time

import joblib
import pandas as pd

from inmuebles.datos import FEATURES, cargar_y_limpiar, separar_y_etiquetar
from inmuebles.modelos import entrenar, evaluar
from inmuebles.salidas import graficar, guardar_base


@dataclass(frozen=True)
class Configuracion:
    archivo: str = "datasets/departamentos_2020.csv"
    salida: str = "resultados"
    run_id: str = "departamentos_2020"
    cuantil: float = 0.75
    semilla: int = 42
    folds: int = 3


def ejecutar(config=Configuracion()):
    """Genera métricas, predicciones, gráficos, modelo, BD, configuración y log."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config.run_id):
        raise ValueError("run_id sólo admite letras ASCII, números, guion y guion bajo")
    carpeta = Path(config.salida) / config.run_id
    carpeta.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inmuebles")
    logger.setLevel(logging.INFO)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    archivo_log = logging.FileHandler(carpeta / "ejecucion.log", mode="w",
                                      encoding="utf-8")
    consola = logging.StreamHandler()
    for handler in [archivo_log, consola]:
        handler.setFormatter(formato)
        logger.addHandler(handler)
    inicio = time.perf_counter()
    try:
        return _ejecutar(config, carpeta, logger, inicio)
    except Exception:
        logger.exception("La ejecución falló")
        raise
    finally:
        for handler in [archivo_log, consola]:
            logger.removeHandler(handler)
            handler.close()


def _ejecutar(config, carpeta, logger, inicio):
    logger.info("Inicio de corrida %s", config.run_id)
    datos, auditoria = cargar_y_limpiar(config.archivo)
    train, test, umbral = separar_y_etiquetar(datos, config.cuantil, config.semilla)
    busquedas, cv, ganador = entrenar(train, config.semilla, config.folds)
    # La selección ya quedó cerrada por CV antes de mirar el holdout.
    metricas_test, predicciones, matrices = evaluar(busquedas, test)
    comparacion = cv.merge(metricas_test, on="modelo", validate="one_to_one")
    pipeline = busquedas[ganador].best_estimator_
    versiones = {p: importlib.metadata.version(p) for p in [
        "pandas", "numpy", "scikit-learn", "matplotlib", "joblib"]}
    parametros = {
        **asdict(config), "umbral_usd": umbral,
        "regla_clase": "precio_usd > umbral_usd",
        "features": FEATURES, "modelo_elegido": ganador,
        "criterio_seleccion": "Mayor F1 medio en validación cruzada agrupada",
        "parametros_modelo": busquedas[ganador].best_params_,
        "versiones": versiones,
        "sha256_dataset": hashlib.sha256(Path(config.archivo).read_bytes()).hexdigest(),
    }
    resumen = {
        "configuracion": parametros, "limpieza": auditoria,
        "entrenamiento": {"filas": len(train),
                          "grupos": int(train.grupo_direccion.nunique()),
                          "proporcion_alto": float(train.segmento_alto.mean())},
        "prueba": {"filas": len(test), "grupos": int(test.grupo_direccion.nunique()),
                   "proporcion_alto": float(test.segmento_alto.mean())},
        "grupos_compartidos": 0,
        "comparacion": comparacion.to_dict(orient="records"),
        "matrices_confusion": matrices,
    }
    comparacion.to_csv(carpeta / "metricas.csv", index=False)
    predicciones.to_csv(carpeta / "predicciones.csv", index=False)
    for nombre, busqueda in busquedas.items():
        resultados = pd.DataFrame(busqueda.cv_results_)
        resultados.to_csv(carpeta / f"busqueda_{nombre}.csv", index=False)
    joblib.dump({"pipeline": pipeline, "umbral_usd": umbral,
                 "features": FEATURES, "configuracion": parametros},
                carpeta / "modelo.joblib")
    logger.info("Guardando SQLite con datos limpios y transformación ajustada en train")
    guardar_base(carpeta / "proyecto.sqlite", train, test, pipeline,
                 predicciones, comparacion, parametros)
    graficar(train, test, comparacion, matrices, ganador, pipeline,
             carpeta / "graficos")
    resumen["duracion_segundos"] = round(time.perf_counter() - inicio, 2)
    (carpeta / "resumen.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    (carpeta / "configuracion.json").write_text(
        json.dumps(parametros, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finalizó %s en %.1f segundos", ganador, resumen["duracion_segundos"])
    return resumen
