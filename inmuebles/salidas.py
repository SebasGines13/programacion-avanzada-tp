"""Persistencia local y gráficos para el notebook y la defensa."""

import json
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from inmuebles.datos import FEATURES
from inmuebles.modelos import importancia_variables


def guardar_base(ruta, train, test, pipeline, predicciones, metricas, config):
    """Reemplaza tablas dentro de la BD de esta corrida, sin acumular filas."""
    datos = pd.concat([train.assign(particion="train"),
                       test.assign(particion="test")], ignore_index=True)
    transformador = pipeline.named_steps["preprocesamiento"]
    # Sólo transform: el ajuste ya se hizo en entrenamiento, nunca con prueba.
    transformados = pd.DataFrame(
        transformador.transform(datos[FEATURES]),
        columns=transformador.get_feature_names_out())
    transformados.insert(0, "fila_fuente", datos.fila_fuente.to_numpy())
    config_df = pd.DataFrame([
        {"parametro": k, "valor_json": json.dumps(v, ensure_ascii=False)}
        for k, v in config.items()
    ])
    with sqlite3.connect(ruta) as conexion:
        datos.to_sql("datos_limpios", conexion, if_exists="replace", index=False)
        transformados.to_sql("datos_preprocesados", conexion,
                             if_exists="replace", index=False, chunksize=2000)
        predicciones.to_sql("predicciones", conexion, if_exists="replace", index=False)
        metricas.drop(columns="parametros").to_sql(
            "metricas", conexion, if_exists="replace", index=False)
        config_df.to_sql("configuracion", conexion, if_exists="replace", index=False)
        conexion.execute("CREATE UNIQUE INDEX idx_fila ON datos_limpios(fila_fuente)")
        conexion.execute("CREATE UNIQUE INDEX idx_transformada "
                         "ON datos_preprocesados(fila_fuente)")
        conexion.execute("CREATE UNIQUE INDEX idx_prediccion "
                         "ON predicciones(fila_fuente, modelo)")


def graficar(train, test, comparacion, matrices, ganador, pipeline, carpeta):
    """Exploración del entrenamiento y resultados finales del holdout."""
    carpeta.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    archivos = []

    def guardar(fig, nombre):
        fig.tight_layout()
        destino = carpeta / nombre
        fig.savefig(destino, bbox_inches="tight")
        plt.close(fig)
        archivos.append(str(destino))

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    train.precio_usd.plot.hist(bins=60, logx=True, ax=ejes[0], color="#226b8c")
    ejes[0].set(xlabel="Precio publicado (USD, eje logarítmico)",
                ylabel="Publicaciones", title="Precios en entrenamiento")
    distribucion = pd.DataFrame({
        "Entrenamiento": train.segmento_alto.value_counts(normalize=True),
        "Prueba": test.segmento_alto.value_counts(normalize=True),
    }).reindex([0, 1]).rename(index={0: "Estándar", 1: "Alto"})
    distribucion.plot.bar(ax=ejes[1], rot=0, color=["#226b8c", "#e5a146"])
    ejes[1].set(title="Distribución de clases", ylabel="Proporción")
    guardar(fig, "01_distribuciones.png")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    frecuentes = train.barrio.value_counts().head(12).index
    medianas = train.loc[train.barrio.isin(frecuentes)].groupby(
        "barrio").precio_usd.median().sort_values()
    medianas.plot.barh(ax=ejes[0], color="#226b8c")
    ejes[0].set(title="Mediana en los 12 barrios más frecuentes", xlabel="USD",
                ylabel="Barrio / zona de la fuente")
    muestra = train.sample(min(4000, len(train)), random_state=42)
    ejes[1].scatter(muestra.superficie, muestra.precio_usd,
                    c=muestra.segmento_alto, cmap="coolwarm", s=6, alpha=.3)
    ejes[1].set(xscale="log", yscale="log", xlabel="Superficie (m²)",
                ylabel="Precio (USD)", title="Muestra del entrenamiento")
    guardar(fig, "02_exploracion.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    tabla = comparacion.set_index("modelo")
    tabla[["f1_cv_mean", "f1_test"]].rename(columns={
        "f1_cv_mean": "F1 validación cruzada", "f1_test": "F1 prueba",
    }).plot.bar(ax=ax, rot=0, color=["#226b8c", "#e5a146"])
    ax.set(ylim=(0, 1), ylabel="F1 de segmento alto",
           title="Selección por CV; prueba reservada para evaluación final",
           xlabel="Modelo")
    guardar(fig, "03_comparacion.png")

    fig, ejes = plt.subplots(1, len(matrices), figsize=(12, 3.6))
    for ax, (nombre, matriz) in zip(ejes, matrices.items()):
        ConfusionMatrixDisplay(pd.DataFrame(matriz).to_numpy(),
                               display_labels=["Estándar", "Alto"]).plot(
                                   ax=ax, colorbar=False, cmap="Blues")
        ax.set(title=nombre, xlabel="Predicción", ylabel="Clase real")
    guardar(fig, "04_matrices_confusion.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    importancia_variables(pipeline).head(12).sort_values("importancia").plot.barh(
        x="variable", y="importancia", ax=ax, color="#226b8c", legend=False)
    ax.set(title=f"Variables del modelo elegido: {ganador}",
           xlabel="Importancia o |coeficiente|; no implica causalidad", ylabel="")
    guardar(fig, "05_variables.png")
    return archivos
