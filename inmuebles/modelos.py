"""Preprocesamiento dentro del Pipeline, búsqueda y evaluación independiente."""

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from inmuebles.datos import CATEGORICAS, FEATURES, NUMERICAS

LOGGER = logging.getLogger(__name__)


def preprocesador():
    """Encadena las transformaciones vistas en clase sin ajustar aún los datos."""
    # fit aprende mediana y escala; transform reutiliza esos valores en prueba.
    numerico = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputacion", SimpleImputer(strategy="constant", fill_value="SIN_DATO")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False)),
    ])
    # Cada tipo de columna recibe su tratamiento, dentro del mismo Pipeline.
    return ColumnTransformer([
        ("numericas", numerico, NUMERICAS),
        ("categoricas", categorico, CATEGORICAS),
    ], remainder="drop")


def entrenar(train, semilla=42, folds=3):
    """Selecciona por F1 en CV agrupada; nunca recibe el conjunto de prueba."""
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=semilla)
    # Reutilizamos las mismas particiones para comparar los modelos justamente.
    particiones = list(cv.split(train[FEATURES], train.segmento_alto,
                                train.grupo_direccion))
    for ajuste, validacion in particiones:
        assert not (set(train.iloc[ajuste].grupo_direccion)
                    & set(train.iloc[validacion].grupo_direccion))
        if any(train.iloc[idx].segmento_alto.nunique() != 2
               for idx in [ajuste, validacion]):
            raise ValueError("Un fold no tiene ambas clases; reduzca folds")
    candidatos = {
        "base": (DummyClassifier(strategy="most_frequent"), {}),
        "logistica": (
            LogisticRegression(max_iter=1000, random_state=semilla),
            {"modelo__C": [0.1, 1.0, 10.0],
             "modelo__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=120, n_jobs=1,
                                   random_state=semilla),
            {"modelo__max_depth": [12, None],
             "modelo__min_samples_leaf": [5, 20]},
        ),
    }
    # F1 prioriza la clase alta; accuracy sola favorece a la mayoría estándar.
    metricas = {"accuracy": "accuracy", "precision": "precision",
                "recall": "recall", "f1": "f1"}
    busquedas, filas = {}, []
    for nombre, (modelo, rejilla) in candidatos.items():
        LOGGER.info("Entrenamiento %s: %s folds", nombre, folds)
        pipe = Pipeline([("preprocesamiento", clone(preprocesador())),
                         ("modelo", modelo)])
        busqueda = GridSearchCV(pipe, rejilla, scoring=metricas, refit="f1",
                                cv=particiones, n_jobs=2, error_score="raise")
        # GridSearchCV vuelve a ajustar el preprocesador en cada fold.
        busqueda.fit(train[FEATURES], train.segmento_alto)
        busquedas[nombre] = busqueda
        indice = busqueda.best_index_
        fila = {"modelo": nombre, "parametros": busqueda.best_params_}
        for metrica in metricas:
            for estadistico in ["mean", "std"]:
                fila[f"{metrica}_cv_{estadistico}"] = float(
                    busqueda.cv_results_[f"{estadistico}_test_{metrica}"][indice])
        filas.append(fila)
        LOGGER.info("%s F1 CV=%.4f", nombre, busqueda.best_score_)
    comparacion = pd.DataFrame(filas)
    # El desempate conserva el orden declarado: se prefiere el modelo simple.
    elegibles = comparacion.loc[comparacion.modelo != "base"]
    ganador = str(elegibles.loc[elegibles.f1_cv_mean.idxmax(), "modelo"])
    LOGGER.info("Modelo elegido por CV antes de evaluar prueba: %s", ganador)
    return busquedas, comparacion, ganador


def evaluar(busquedas, test):
    """Métricas del holdout una vez fijada la selección de modelo."""
    filas, predicciones, matrices = [], [], {}
    for nombre, busqueda in busquedas.items():
        pred = busqueda.predict(test[FEATURES])
        real = test.segmento_alto
        filas.append({
            "modelo": nombre,
            "accuracy_test": float(accuracy_score(real, pred)),
            "precision_test": float(precision_score(real, pred, zero_division=0)),
            "recall_test": float(recall_score(real, pred, zero_division=0)),
            "f1_test": float(f1_score(real, pred, zero_division=0)),
        })
        salida = test[["fila_fuente", "segmento_alto"]].copy()
        salida["modelo"] = nombre
        salida["prediccion"] = pred
        indice_alto = list(busqueda.classes_).index(1)
        salida["probabilidad_alto"] = busqueda.predict_proba(
            test[FEATURES])[:, indice_alto]
        predicciones.append(salida)
        matrices[nombre] = confusion_matrix(real, pred, labels=[0, 1]).tolist()
    return pd.DataFrame(filas), pd.concat(predicciones, ignore_index=True), matrices


def importancia_variables(pipeline):
    """Magnitud de coeficientes o importancias; no implica causalidad."""
    nombres = pipeline.named_steps["preprocesamiento"].get_feature_names_out()
    modelo = pipeline.named_steps["modelo"]
    valores = (modelo.feature_importances_ if hasattr(modelo, "feature_importances_")
               else np.abs(modelo.coef_[0]))
    return pd.DataFrame({"variable": nombres, "importancia": valores}).sort_values(
        "importancia", ascending=False)
