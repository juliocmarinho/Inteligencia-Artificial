"""
Módulo 3 — Integração entre a Rede Bayesiana (Módulo 1) e o A* (Módulo 2).

Fluxo (conforme seção 4 do enunciado):

  Paciente chega
       |
       v
  [Coleta de sintomas e sinais vitais]
       |
       v
  [Rede Bayesiana] -> produz P(gravidade baixa / média / alta)
       |
       v
  [Fila de pacientes atualizada com novos riscos]
       |
       v
  [Algoritmo A*] -> calcula a ordem ótima de atendimento
       |
       v
  [Saída: ordem de chamada dos pacientes]

A ponte entre os dois módulos é a função `gerar_paciente_astar`, que recebe
os sintomas de um paciente, roda a inferência bayesiana e devolve um
`Paciente` (objeto do módulo A*) já com o `p_alta` = P(Gravidade=alta)
vindo diretamente da rede.
"""

import random

from rede_bayesiana import construir_rede, inferir_gravidade
from astar import Paciente


def gerar_paciente_astar(modelo_bayesiano, id_paciente, sintomas, espera_inicial,
                          tempo_atendimento=8.0, modelo_risco="linear", tau=15.0):
    """
    sintomas: dicionário com evidências para a rede bayesiana, por exemplo:
        {"Febre": "alta", "Saturacao": "critica", "Dor": "intensa"}
    (variáveis não incluídas são tratadas como ausentes/desconhecidas)
    """
    resultado = inferir_gravidade(modelo_bayesiano, sintomas)
    p_alta = resultado["alta"]
    return Paciente(
        id=id_paciente,
        p_alta=p_alta,
        espera_inicial=espera_inicial,
        tempo_atendimento=tempo_atendimento,
        modelo_risco=modelo_risco,
        tau=tau,
    )


def simular_base_de_pacientes(n_pacientes, seed=42, modelo_risco="linear"):
    """
    Gera uma base sintética de pacientes com sintomas aleatórios (mas
    plausíveis) e já converte cada um em um `Paciente` do módulo A*,
    passando pela rede bayesiana -- demonstrando a integração completa
    entre os dois módulos.
    """
    random.seed(seed)
    modelo = construir_rede()

    opcoes = {
        "Febre": ["ausente", "moderada", "alta"],
        "Saturacao": ["normal", "reduzida", "critica"],
        "PressaoArterial": ["normal", "baixa", "muito_baixa"],
        "FrequenciaCardiaca": ["normal", "taquicardia", "grave"],
        "Dor": ["leve", "moderada", "intensa"],
        "IdadeDoencaCronica": ["baixo_risco", "alto_risco"],
    }
    # Pesos para deixar a base mais realista: a maioria dos pacientes tem
    # quadros leves/moderados, poucos são graves (distribuição de um PS real)
    pesos = {
        "Febre": [0.55, 0.30, 0.15],
        "Saturacao": [0.60, 0.25, 0.15],
        "PressaoArterial": [0.55, 0.30, 0.15],
        "FrequenciaCardiaca": [0.55, 0.30, 0.15],
        "Dor": [0.40, 0.35, 0.25],
        "IdadeDoencaCronica": [0.65, 0.35],
    }

    pacientes = []
    for i in range(n_pacientes):
        sintomas = {}
        for variavel, valores in opcoes.items():
            # 15% de chance do sinal não ter sido coletado ainda (dado
            # faltante), para demonstrar que a rede lida bem com isso
            if random.random() < 0.15:
                continue
            sintomas[variavel] = random.choices(valores, weights=pesos[variavel])[0]

        espera_inicial = random.randint(0, 45)
        tempo_atendimento = random.randint(4, 15)

        paciente = gerar_paciente_astar(
            modelo, f"Paciente_{i+1:03d}", sintomas, espera_inicial,
            tempo_atendimento, modelo_risco=modelo_risco,
        )
        pacientes.append(paciente)

    return pacientes


if __name__ == "__main__":
    print("=== Demonstração da integração (5 pacientes) ===\n")
    pacientes = simular_base_de_pacientes(5, seed=1)
    for p in pacientes:
        print(f"{p.id}: P(alta)={p.p_alta:.2f} | espera_inicial={p.espera_inicial}min "
              f"| tempo_atendimento={p.tempo_atendimento}min")

    from astar import busca_astar, estrategia_fifo, estrategia_gulosa

    print("\n=== Ordem de atendimento decidida pelo A* ===")
    ordem, custo, info = busca_astar(pacientes)
    print(f"Ordem: {ordem}")
    print(f"Custo total (risco acumulado): {custo:.2f}")
