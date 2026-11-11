import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from circulation.log import setup_logging
from circulation.regazzoni2020 import Regazzoni2020
from meuModelo import MeuModeloCardiovascular


def main():
    # Executa ambas as simulações e gera gráficos sobrepostos (linhas contínuas para o seu modelo, tracejadas para Regazzoni).
    tempo_inicio = time.time()
    
    pasta_graficos = Path("graficos_comparacao")
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    setup_logging()

    print("Iniciando a simulação do modelo Regazzoni (Original)...")
    regazzoni = Regazzoni2020()
    hist_r = regazzoni.solve(num_beats=10)
    
    print("Iniciando a simulação do Meu Modelo...")
    modelo = MeuModeloCardiovascular(verbose=False)
    hist_m = modelo.solve(num_beats=10, dt=1e-4, method="forward_euler")

    # Extração de dados - Regazzoni
    t_r = hist_r["time"]
    V_lv_r, P_lv_r = hist_r["V_LV"], hist_r["p_LV"]
    V_la_r, P_la_r = hist_r["V_LA"], hist_r["p_LA"]
    P_ar_r = hist_r["p_AR_SYS"]
    mask_r = t_r >= (t_r[-1] - (1.0 / regazzoni.HR)) # Máscara para o último batimento

    # Extração de dados - Meu Modelo
    t_m = hist_m["time"]
    V_lv_m, P_lv_m = hist_m["V_LV"], hist_m["p_LV"]
    V_la_m, P_la_m = hist_m["V_LA"], hist_m["p_LA"]
    P_ar_m = hist_m["p_AR_SYS"]
    mask_m = t_m >= (t_m[-1] - (1.0 / modelo.HR)) # Máscara para o último batimento

    print("Gerando Figura 1: Loops Pressão-Volume (Comparativo)...")
    fig1, ax1 = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4.5))

    # Painel 0: Todos os Batimentos
    ax1[0].plot(V_lv_r, P_lv_r, color="gray", linestyle="--", alpha=0.7, label="Regazzoni")
    ax1[0].plot(V_lv_m, P_lv_m, color="blue", linestyle="-", label="Meu Modelo")
    ax1[0].set_title("PV loop (Todos os Batimentos)")
    ax1[0].set_xlabel("Volume (mL)")
    ax1[0].set_ylabel("Pressão (mmHg)")
    ax1[0].legend()
    ax1[0].grid(True, linestyle=":", alpha=0.7)

    # Painel 1: Último Batimento
    ax1[1].plot(V_lv_r[mask_r], P_lv_r[mask_r], color="gray", linestyle="--", alpha=0.7, label="Regazzoni")
    ax1[1].plot(V_lv_m[mask_m], P_lv_m[mask_m], color="blue", linestyle="-", label="Meu Modelo")
    ax1[1].set_title("PV loop (Último Batimento)")
    ax1[1].set_xlabel("Volume (mL)")
    ax1[1].legend()
    ax1[1].grid(True, linestyle=":", alpha=0.7)

    fig1.tight_layout()
    fig1.savefig(pasta_graficos / "figura1_pv_loops_comparativo.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    print("Gerando Figura 2: Sinais de Pressão e Volume no Tempo (Comparativo)...")
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(12, 9))

    # Painel 0: Pressões
    # Regazzoni (Tracejados claros)
    ax2[0].plot(t_r, P_lv_r, color="blue", linestyle="--", alpha=0.5, label="Regazzoni p_LV")
    ax2[0].plot(t_r, P_la_r, color="purple", linestyle="--", alpha=0.5, label="Regazzoni p_LA")
    ax2[0].plot(t_r, P_ar_r, color="red", linestyle="--", alpha=0.5, label="Regazzoni p_AR_SYS")
    # Meu Modelo (Contínuos escuros)
    ax2[0].plot(t_m, P_lv_m, color="blue", linestyle="-", label="Meu p_LV")
    ax2[0].plot(t_m, P_la_m, color="purple", linestyle="-", label="Meu p_LA")
    ax2[0].plot(t_m, P_ar_m, color="red", linestyle="-", label="Meu p_AR_SYS")
    
    ax2[0].set_title("Pressões ao Longo do Tempo")
    ax2[0].set_ylabel("Pressão (mmHg)")
    ax2[0].legend(loc="upper right", ncol=2)
    ax2[0].grid(True, linestyle=":", alpha=0.7)

    # Painel 1: Volumes
    # Regazzoni (Tracejados claros)
    ax2[1].plot(t_r, V_lv_r, color="blue", linestyle="--", alpha=0.5, label="Regazzoni V_LV")
    ax2[1].plot(t_r, V_la_r, color="purple", linestyle="--", alpha=0.5, label="Regazzoni V_LA")
    # Meu Modelo (Contínuos escuros)
    ax2[1].plot(t_m, V_lv_m, color="blue", linestyle="-", label="Meu V_LV")
    ax2[1].plot(t_m, V_la_m, color="purple", linestyle="-", label="Meu V_LA")
    
    ax2[1].set_title("Volumes ao Longo do Tempo")
    ax2[1].set_xlabel("Tempo (s)")
    ax2[1].set_ylabel("Volume (mL)")
    ax2[1].legend(loc="upper right", ncol=2)
    ax2[1].grid(True, linestyle=":", alpha=0.7)

    fig2.tight_layout()
    fig2.savefig(pasta_graficos / "figura2_sinais_tempo_comparativo.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    tempo_total = time.time() - tempo_inicio
    print(f"Sucesso! Gráficos comparativos salvos em: {pasta_graficos.absolute()}")
    print(f"⏱️ Tempo total de execução do script: {tempo_total:.2f} segundos.")


if __name__ == "__main__":
    main()