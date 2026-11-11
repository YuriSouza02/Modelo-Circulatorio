import time
import numpy as np
from meuModelo import MeuModeloCardiovascular


def main():
    # Executa a simulação do modelo cardiovascular 100 vezes para medir a média, desvio padrão e tempo total de execução.
    num_execucoes = 100
    tempos = []

    print(f"Iniciando benchmark (MeuModeloCardiovascular): executando {num_execucoes} vezes...")

    for i in range(num_execucoes):
        t0 = time.time()
        modelo = MeuModeloCardiovascular(verbose=False)
        _ = modelo.solve(num_beats=10, dt=1e-4, method="forward_euler")
        tf = time.time()

        duracao = tf - t0
        tempos.append(duracao)

        print(f"Execução {i + 1}/{num_execucoes}: {duracao:.4f} s")

    tempo_medio = np.mean(tempos)
    desvio_padrao = np.std(tempos)
    tempo_total_acumulado = np.sum(tempos)

    print("\n--- RESULTADOS DO BENCHMARK (MEU MODELO) ---")
    print(f"⏱️ Tempo médio por execução: {tempo_medio:.4f} s")
    print(f"📊 Desvio padrão: ±{desvio_padrao:.4f} s")
    print(f"⏳ Tempo total acumulado: {tempo_total_acumulado:.2f} s")


if __name__ == "__main__":
    main()