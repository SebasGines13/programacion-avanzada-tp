"""Controles de fuga de información y persistencia, con datos sintéticos pequeños."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from inmuebles.datos import FEATURES, separar_y_etiquetar
from inmuebles.modelos import preprocesador
from inmuebles.salidas import guardar_base
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


class MetodologiaTest(unittest.TestCase):
    def setUp(self):
        self.datos = pd.DataFrame({
            "fila_fuente": range(200),
            "superficie": [30 + i % 50 for i in range(200)],
            "ambientes": [1 + i % 4 for i in range(200)],
            "barrio": ["ALMAGRO"] * 200,
            "precio_usd": [50000 + i * 1000 for i in range(200)],
            "grupo_direccion": [f"CALLE {i // 2}" for i in range(200)],
        })

    def test_direcciones_separadas_y_umbral_sin_prueba(self):
        train, test, umbral = separar_y_etiquetar(self.datos)
        self.assertFalse(set(train.grupo_direccion) & set(test.grupo_direccion))
        alterados = self.datos.copy()
        alterados.loc[test.precio_usd.idxmax(), "precio_usd"] *= 10
        train2, _, umbral2 = separar_y_etiquetar(alterados)
        self.assertEqual(umbral, umbral2)
        pd.testing.assert_series_equal(train.segmento_alto, train2.segmento_alto)
        self.assertEqual(umbral, train.precio_usd.quantile(0.75))

    def test_transformacion_admite_barrio_nuevo_y_no_aprende_del_test(self):
        train, test, _ = separar_y_etiquetar(self.datos)
        train.loc[train.index[0], "superficie"] = float("nan")
        prep = preprocesador().fit(train[FEATURES])
        media = prep.named_transformers_["numericas"].named_steps["escalado"].mean_.copy()
        test["barrio"] = "NUEVO"
        test["superficie"] = 999999
        transformado = prep.transform(test[FEATURES])
        self.assertEqual(len(test), len(transformado))
        self.assertTrue((media == prep.named_transformers_[
            "numericas"].named_steps["escalado"].mean_).all())
        self.assertFalse(any("precio" in n.lower() or "dolar" in n.lower()
                             for n in prep.get_feature_names_out()))

    def test_sqlite_repetible_y_vinculos(self):
        train, test, _ = separar_y_etiquetar(self.datos)
        pipe = Pipeline([("preprocesamiento", preprocesador()),
                         ("modelo", DummyClassifier())]).fit(
                             train[FEATURES], train.segmento_alto)
        pred = test[["fila_fuente"]].assign(modelo="base", prediccion=0)
        met = pd.DataFrame([{"modelo": "base", "f1_test": 0., "parametros": {}}])
        with tempfile.TemporaryDirectory() as temp:
            ruta = Path(temp) / "prueba.sqlite"
            for _ in range(2):
                guardar_base(ruta, train, test, pipe, pred, met, {"semilla": 42})
            with sqlite3.connect(ruta) as conexion:
                self.assertEqual(conexion.execute(
                    "SELECT count(*) FROM datos_preprocesados").fetchone()[0], 200)
                huerfanas = conexion.execute(
                    "SELECT count(*) FROM predicciones p LEFT JOIN datos_limpios d "
                    "ON p.fila_fuente=d.fila_fuente WHERE d.fila_fuente IS NULL"
                ).fetchone()[0]
                self.assertEqual(huerfanas, 0)


if __name__ == "__main__":
    unittest.main()
