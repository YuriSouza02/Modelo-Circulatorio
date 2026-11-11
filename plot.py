import time
import matplotlib.pyplot as plt
from pathlib import Path

from meuModelo import MeuModeloCardiovascular


def main():
    # Executa a simulação completa do modelo cardiovascular, cronometra o tempo de processamento e gera os gráficos de loops PV e sinais no tempo.
    tempo_inicio = time.time()
    
    pasta_graficos = Path("graficos")
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    print("Iniciando a simulação do modelo cardiovascular...")
    modelo = MeuModeloCardiovascular(
        outdir=Path("resultados_meu_modelo"), verbose=False
    )

    history = modelo.solve(num_beats=10, dt=1e-4, method="forward_euler")

    time_array = history["time"]
    V_lv, P_lv = history["V_LV"], history["p_LV"]
    V_la, P_la = history["V_LA"], history["p_LA"]
    P_ar_sys = history["p_AR_SYS"]

    passos_por_batimento = int((1.0 / modelo.HR) / 1e-4)

    print("Gerando Figura 1: Loops Pressão-Volume...")
    fig1, ax1 = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4.5))

    ax1[0].plot(V_lv, P_lv, color="blue")
    ax1[0].set_title("PV loop (Todos os Batimentos)")
    ax1[0].set_xlabel("Volume (mL)")
    ax1[0].set_ylabel("Pressão (mmHg)")
    ax1[0].grid(True, linestyle=":", alpha=0.7)

    ax1[1].plot(V_lv[-passos_por_batimento:], P_lv[-passos_por_batimento:])
    ax1[1].set_title("PV loop (Último Batimento)")
    ax1[1].set_xlabel("Volume (mL)")
    ax1[1].grid(True, linestyle=":", alpha=0.7)

    fig1.tight_layout()
    caminho_fig1 = pasta_graficos / "figura1_pv_loops.png"
    fig1.savefig(caminho_fig1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    print("Gerando Figura 2: Sinais de Pressão e Volume no Tempo...")
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(12, 9))

    ax2[0].plot(time_array, P_lv, label="p_LV", color="blue")
    ax2[0].plot(time_array, P_la, label="p_LA", color="purple")
    ax2[0].plot(time_array, P_ar_sys, label="p_AR_SYS", color="red")
    ax2[0].set_title("Pressões ao Longo do Tempo")
    ax2[0].set_ylabel("Pressão (mmHg)")
    ax2[0].legend(loc="upper right")
    ax2[0].grid(True, linestyle=":", alpha=0.7)

    ax2[1].plot(time_array, V_lv, label="V_LV", color="blue")
    ax2[1].plot(time_array, V_la, label="V_LA", color="purple")
    ax2[1].set_title("Volumes ao Longo do Tempo")
    ax2[1].set_xlabel("Tempo (s)")
    ax2[1].set_ylabel("Volume (mL)")
    ax2[1].legend(loc="upper right")
    ax2[1].grid(True, linestyle=":", alpha=0.7)

    fig2.tight_layout()
    caminho_fig2 = pasta_graficos / "figura2_sinais_tempo.png"
    fig2.savefig(caminho_fig2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    tempo_fim = time.time()
    tempo_total = tempo_fim - tempo_inicio
    print(f"Sucesso! Gráficos gerados e salvos em: {pasta_graficos.absolute()}")
    print(f"⏱️ Tempo total de processamento: {tempo_total:.2f} segundos.")


if __name__ == "__main__":
    main()