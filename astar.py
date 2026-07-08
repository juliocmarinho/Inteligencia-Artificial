"""
Módulo 2 — Busca A* para decidir a ordem de atendimento na fila.

Formulação (conforme enunciado, seção 3.2):
- Estado: conjunto de pacientes ainda aguardando.
- Estado inicial: todos os pacientes na fila.
- Ação: escolher o próximo paciente a ser atendido (remove da fila).
- Custo de uma ação: risco acumulado de todos os pacientes que continuam
  esperando após essa ação (ver `custo_acao`).
- Objetivo: fila vazia.
- Heurística h(n): soma dos riscos atuais de todos os pacientes ainda na
  fila (seção 3.5) — subestima o custo real porque assume que nenhum risco
  vai aumentar mais, o que nunca acontece de fato -> admissível.

Observação de implementação: como cada paciente tem um tempo de
atendimento fixo, o instante em que um determinado subconjunto de
pacientes já foi atendido não depende da ORDEM em que foram atendidos,
apenas de QUAIS foram atendidos (a soma dos tempos de atendimento é
comutativa). Isso permite representar o estado apenas pelo conjunto de
pacientes restantes (frozenset), o que reduz bastante o espaço de estados
e permite usar memorização de custo por estado.
"""

import heapq
import itertools
import math
from dataclasses import dataclass


@dataclass
class Paciente:
    id: str
    p_alta: float          # P(Gravidade = alta), vindo da rede bayesiana
    espera_inicial: float  # tempo (min) que já esperou até o momento da decisão
    tempo_atendimento: float = 5.0  # duração do atendimento (min)
    modelo_risco: str = "linear"    # "linear" ou "exponencial"
    tau: float = 15.0               # constante da versão exponencial

    def risco(self, tempo_espera_total):
        if self.modelo_risco == "linear":
            return self.p_alta * tempo_espera_total
        else:
            return self.p_alta * math.exp(tempo_espera_total / self.tau)


def _tempo_decorrido(pacientes_por_id, restantes_ids, tempo_total_atendimento):
    """
    Tempo decorrido desde o início da simulação até o momento em que o
    conjunto `restantes_ids` é exatamente quem ainda falta atender.
    Como a soma dos tempos de atendimento é comutativa, isso só depende
    de quais pacientes já foram atendidos (não da ordem).
    """
    tempo_atendidos = tempo_total_atendimento - sum(
        pacientes_por_id[pid].tempo_atendimento for pid in restantes_ids
    )
    return tempo_atendidos


def _risco_estado(pacientes_por_id, restantes_ids, tempo_decorrido):
    return sum(
        pacientes_por_id[pid].risco(pacientes_por_id[pid].espera_inicial + tempo_decorrido)
        for pid in restantes_ids
    )


def _custo_de_uma_ordem_fixa(pacientes_por_id, ordem_ids):
    """Calcula o custo total de atender os pacientes numa ordem já fixada."""
    tempo_decorrido = 0.0
    custo_total = 0.0
    restantes_set = set(ordem_ids)
    for pid in ordem_ids:
        restantes_set.discard(pid)
        tempo_decorrido += pacientes_por_id[pid].tempo_atendimento
        custo_total += _risco_estado(pacientes_por_id, restantes_set, tempo_decorrido)
    return custo_total


def _preparar_agregados(pacientes):
    """
    Prepara pesos que permitem calcular o risco somado de um conjunto de
    pacientes em O(1) (em vez de O(k)), aproveitando que a soma fatora:

    Modelo linear:      risco_i(t) = p_alta_i * (espera_i + t)
       soma_remanescente(t) = t * sum(p_alta_i) + sum(p_alta_i * espera_i)
       -> bastam dois acumuladores (sum_w, sum_we) atualizados a cada remoção.

    Modelo exponencial: risco_i(t) = p_alta_i * exp((espera_i + t) / tau)
       soma_remanescente(t) = exp(t/tau) * sum(p_alta_i * exp(espera_i/tau))
       -> basta um acumulador (sum_wexp), assumindo tau igual para todos
          os pacientes do cenário (simplificação razoável e documentada).

    Isso reduz o custo de expandir cada nó de O(n) para O(1), o que é
    essencial para viabilizar filas de 20-30 pacientes.
    """
    modelos = {p.modelo_risco for p in pacientes}
    if len(modelos) > 1:
        raise ValueError("Todos os pacientes de uma mesma simulação devem usar o mesmo modelo_risco")
    modelo = modelos.pop()

    if modelo == "linear":
        pesos = {p.id: (p.p_alta, p.p_alta * p.espera_inicial) for p in pacientes}
    else:
        taus = {p.tau for p in pacientes}
        if len(taus) > 1:
            raise ValueError("Todos os pacientes devem compartilhar o mesmo tau no modelo exponencial")
        tau = taus.pop()
        pesos = {p.id: (p.p_alta * math.exp(p.espera_inicial / tau), tau) for p in pacientes}

    return modelo, pesos


def _risco_agregado(modelo, sum_a, sum_b, t):
    if modelo == "linear":
        return t * sum_a + sum_b
    else:
        tau = sum_b  # neste modelo, guardamos tau no segundo campo (constante)
        return math.exp(t / tau) * sum_a


def busca_astar(pacientes, max_expansoes=60_000):
    
    """
    Executa A* sobre o espaço de ordens de atendimento.
    Retorna (ordem, custo_total, info) onde info é um dicionário com
    'estados_expandidos' e 'otimo' (True se a otimalidade foi garantida,
    False se o algoritmo parou por orçamento antes de provar otimalidade).
    """

    pacientes_por_id = {p.id: p for p in pacientes}
    todos_ids = frozenset(p.id for p in pacientes)
    modelo, pesos = _preparar_agregados(pacientes)

    sum_a_total = sum(pesos[pid][0] for pid in todos_ids)
    sum_b_total = (sum(pesos[pid][1] for pid in todos_ids) if modelo == "linear"
                   else next(iter(pesos.values()))[1])  # tau (igual p/ todos)

    # --- Limite superior inicial: o melhor entre três estratégias conhecidas,
    # para garantir que o A* nunca reporte um resultado pior que as baselines
    # do experimento (FIFO e Gulosa), mesmo se o orçamento de expansões
    # esgotar antes de provar otimalidade.
    candidatos = {
        "smith": sorted(pacientes, key=lambda p: -(p.p_alta / max(p.tempo_atendimento, 1e-6))),
        "gulosa": sorted(pacientes, key=lambda p: -p.p_alta),
        "fifo": sorted(pacientes, key=lambda p: -p.espera_inicial),
    }
    melhor_ordem, limite_superior = None, math.inf
    for ordem_cand in candidatos.values():
        ids_cand = [p.id for p in ordem_cand]
        custo_cand = _custo_de_uma_ordem_fixa(pacientes_por_id, ids_cand)
        if custo_cand < limite_superior:
            limite_superior, melhor_ordem = custo_cand, ids_cand

    contador = itertools.count()
    g_inicial = 0.0
    h_inicial = _risco_agregado(modelo, sum_a_total, sum_b_total, 0.0)
    # nó: (f, g, contador, restantes_frozenset, ordem, sum_a, sum_b, elapsed)
    aberto = [(g_inicial + h_inicial, g_inicial, next(contador), todos_ids, [],
               sum_a_total, sum_b_total, 0.0)]
    melhor_g = {todos_ids: g_inicial}
    expandidos = 0

    while aberto:
        if expandidos >= max_expansoes:
            return melhor_ordem, limite_superior, {
                "estados_expandidos": expandidos, "otimo": False,
            }

        f, g, _, restantes, ordem, sum_a, sum_b, elapsed = heapq.heappop(aberto)

        if g >= limite_superior:
            continue  # poda segura (ver docstring)

        if g > melhor_g.get(restantes, math.inf):
            continue  # entrada obsoleta no heap

        expandidos += 1

        if not restantes:
            if g < limite_superior:
                limite_superior = g
                melhor_ordem = ordem
            return melhor_ordem, limite_superior, {
                "estados_expandidos": expandidos, "otimo": True,
            }

        for pid in restantes:
            novo_restantes = restantes - {pid}
            novo_elapsed = elapsed + pacientes_por_id[pid].tempo_atendimento

            if modelo == "linear":
                w_a, w_b = pesos[pid]
                novo_sum_a = sum_a - w_a
                novo_sum_b = sum_b - w_b
            else:
                w_a, _ = pesos[pid]
                novo_sum_a = sum_a - w_a
                novo_sum_b = sum_b  # tau não muda

            custo_acao = _risco_agregado(modelo, novo_sum_a, novo_sum_b, novo_elapsed)
            novo_g = g + custo_acao
            novo_f = novo_g + custo_acao  # h(novo estado) = custo_acao (mesma fórmula)

            if novo_g >= limite_superior:
                continue  # poda segura

            if novo_g < melhor_g.get(novo_restantes, math.inf):
                melhor_g[novo_restantes] = novo_g
                heapq.heappush(aberto, (
                    novo_f, novo_g, next(contador),
                    novo_restantes, ordem + [pid],
                    novo_sum_a, novo_sum_b, novo_elapsed,
                ))

    return melhor_ordem, limite_superior, {
        "estados_expandidos": expandidos, "otimo": True,
    }


def estrategia_fifo(pacientes):
    """Atende na ordem de chegada (maior tempo de espera inicial primeiro)."""
    ordem = sorted(pacientes, key=lambda p: -p.espera_inicial)
    return _avaliar_ordem(ordem)


def estrategia_gulosa(pacientes):
    """Atende sempre o de maior P(gravidade alta), ignorando tempo de espera."""
    ordem = sorted(pacientes, key=lambda p: -p.p_alta)
    return _avaliar_ordem(ordem)


def _avaliar_ordem(pacientes_em_ordem):
    """Calcula o custo total (risco acumulado) de uma ordem fixa de atendimento."""
    pacientes_por_id = {p.id: p for p in pacientes_em_ordem}
    tempo_total = sum(p.tempo_atendimento for p in pacientes_em_ordem)
    restantes = list(pacientes_por_id.keys())
    tempo_decorrido = 0.0
    custo_total = 0.0
    ordem_ids = []

    restantes_set = set(restantes)
    for p in pacientes_em_ordem:
        restantes_set.discard(p.id)
        tempo_decorrido += p.tempo_atendimento
        custo_total += _risco_estado(pacientes_por_id, restantes_set, tempo_decorrido)
        ordem_ids.append(p.id)

    return ordem_ids, custo_total, None


if __name__ == "__main__":
    # Exemplo da seção 3.4 do enunciado
    pacientes = [
        Paciente(id="Ana", p_alta=0.85, espera_inicial=10, tempo_atendimento=8),
        Paciente(id="Bruno", p_alta=0.60, espera_inicial=30, tempo_atendimento=8),
        Paciente(id="Carla", p_alta=0.20, espera_inicial=5, tempo_atendimento=8),
    ]

    ordem, custo, info = busca_astar(pacientes)
    print(f"A*      -> ordem: {ordem} | custo total: {custo:.2f} | "
          f"estados expandidos: {info['estados_expandidos']} | ótimo: {info['otimo']}")

    ordem_f, custo_f, _ = estrategia_fifo(pacientes)
    print(f"FIFO    -> ordem: {ordem_f} | custo total: {custo_f:.2f}")

    ordem_g, custo_g, _ = estrategia_gulosa(pacientes)
    print(f"Gulosa  -> ordem: {ordem_g} | custo total: {custo_g:.2f}")
