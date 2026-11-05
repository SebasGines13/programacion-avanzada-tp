"""Punto de entrada: python ejecutar.py [--run-id nombre] [--cuantil 0.75]."""

import argparse

from inmuebles.flujo import Configuracion, ejecutar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", default="datasets/departamentos_2020.csv")
    parser.add_argument("--salida", default="resultados")
    parser.add_argument("--run-id", default="departamentos_2020")
    parser.add_argument("--cuantil", type=float, default=0.75)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    resumen = ejecutar(Configuracion(**vars(args)))
    print(f"Modelo elegido por CV: {resumen['configuracion']['modelo_elegido']}")
    print(f"Umbral del segmento alto: USD {resumen['configuracion']['umbral_usd']:,.0f}")


if __name__ == "__main__":
    main()
