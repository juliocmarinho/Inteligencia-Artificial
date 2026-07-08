"""
Módulo 4 — Experimentos (seção 5 do enunciado).

Compara as três estratégias de ordenação (FIFO, Gulosa, A*) em dois
cenários (pequeno: 5-8 pacientes; médio: 20-30 pacientes) e mede o risco
acumulado total ao final do atendimento de todos os pacientes.

Gera:
  - resultados/tabela_resultados.csv
  - resultados/comparacao_estrategias.png
"""
import os
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from integracao import simular_base_de_pacientes
from astar import busca_astar, estrategia_fifo, estrategia_gulosa


def rodar_cenario(nome, n_pacientes, seed, max_expansoes=60_000):
    pacientes = simular_base_de_pacientes(n_pacientes, seed=seed)

    t0 = time.time()
    ordem_astar, custo_astar, info = busca_astar(pacientes, max_expansoes=max_expansoes)
    tempo_astar = time.time() - t0

    _, custo_fifo, _ = estrategia_fifo(pacientes)
    _, custo_gulosa, _ = estrategia_gulosa(pacientes)

    return {
        "cenario": nome,
        "n_pacientes": n_pacientes,
        "custo_fifo": custo_fifo,
        "custo_gulosa": custo_gulosa,
        "custo_astar": custo_astar,
        "astar_otimo_garantido": info["otimo"],
        "astar_estados_expandidos": info["estados_expandidos"],
        "astar_tempo_segundos": round(tempo_astar, 3),
    }


def main():
    os.makedirs("resultados", exist_ok=True)

    resultados = []

    print("Rodando cenário pequeno (8 pacientes)...")
    resultados.append(rodar_cenario("Pequeno (8 pacientes)", 8, seed=10))

    print("Rodando cenário médio (25 pacientes)...")
    resultados.append(rodar_cenario("Médio (25 pacientes)", 25, seed=20))

    # Salva tabela CSV
    with open("resultados/tabela_resultados.csv", "w", newline="", encoding="utf-8") as f:
        campos = list(resultados[0].keys())
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    # Imprime tabela no console
    print("\n=== Resultados ===")
    for r in resultados:
        print(f"\n{r['cenario']}")
        print(f"  FIFO   : {r['custo_fifo']:.1f}")
        print(f"  Gulosa : {r['custo_gulosa']:.1f}")
        print(f"  A*     : {r['custo_astar']:.1f}  "
              f"(ótimo garantido: {r['astar_otimo_garantido']}, "
              f"{r['astar_estados_expandidos']} estados, "
              f"{r['astar_tempo_segundos']}s)")

    # Gráfico de barras comparando as estratégias nos dois cenários
    fig, ax = plt.subplots(figsize=(8, 5))
    cenarios = [r["cenario"] for r in resultados]
    largura = 0.25
    x = range(len(cenarios))

    ax.bar([i - largura for i in x], [r["custo_fifo"] for r in resultados],
           width=largura, label="FIFO", color="#94a3b8")
    ax.bar(x, [r["custo_gulosa"] for r in resultados],
           width=largura, label="Gulosa", color="#f59e0b")
    ax.bar([i + largura for i in x], [r["custo_astar"] for r in resultados],
           width=largura, label="A*", color="#16a34a")

    ax.set_xticks(list(x))
    ax.set_xticklabels(cenarios)
    ax.set_ylabel("Risco acumulado total")
    ax.set_title("Comparação de estratégias de triagem")
    ax.legend()
    fig.tight_layout()
    fig.savefig("resultados/comparacao_estrategias.png", dpi=150)
    print("\nGráfico salvo em resultados/comparacao_estrategias.png")
    print("Tabela salva em resultados/tabela_resultados.csv")


if __name__ == "__main__":
    main()
