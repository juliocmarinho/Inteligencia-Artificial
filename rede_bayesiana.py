"""
Módulo 1 — Rede Bayesiana para estimativa de gravidade de pacientes.

Estrutura (conforme enunciado):

Febre ──────────────────────────────┐
Saturação de O2 ────────────────────┤
Pressão Arterial ───────────────────┼──► Gravidade ──► Risco de Deterioração
Frequência Cardíaca ────────────────┤
Nível de Dor ───────────────────────┤
Idade + Doença Crônica ─────────────┘

A variável "Gravidade" tem 3 estados: 0=baixa, 1=média, 2=alta.
"Risco de Deterioração" é calculado no módulo 2 (astar.py), a partir de
P(Gravidade=alta) e do tempo de espera — por isso ele não entra como nó
determinístico aqui, e sim como saída consumida pelo A*.

Os valores das CPTs foram estimados de forma plausível, inspirados no
Protocolo de Manchester (classificação de risco por sinais vitais), pois
o grupo não usou uma base de dados real do protocolo.
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def construir_rede():
    """Constrói e retorna a rede bayesiana de triagem já com as CPTs."""

    modelo = DiscreteBayesianNetwork([
        ("Febre", "Gravidade"),
        ("Saturacao", "Gravidade"),
        ("PressaoArterial", "Gravidade"),
        ("FrequenciaCardiaca", "Gravidade"),
        ("Dor", "Gravidade"),
        ("IdadeDoencaCronica", "Gravidade"),
    ])

    # --- CPTs das variáveis raiz (evidências) ---
    # Cada uma tem uma distribuição a priori simples (podem ser ajustadas
    # com dados reais de admissão hospitalar, se disponíveis).

    cpd_febre = TabularCPD(
        variable="Febre", variable_card=3,
        values=[[0.55], [0.30], [0.15]],
        state_names={"Febre": ["ausente", "moderada", "alta"]},
    )

    cpd_saturacao = TabularCPD(
        variable="Saturacao", variable_card=3,
        values=[[0.60], [0.25], [0.15]],
        state_names={"Saturacao": ["normal", "reduzida", "critica"]},
    )

    cpd_pressao = TabularCPD(
        variable="PressaoArterial", variable_card=3,
        values=[[0.55], [0.30], [0.15]],
        state_names={"PressaoArterial": ["normal", "baixa", "muito_baixa"]},
    )

    cpd_freq_cardiaca = TabularCPD(
        variable="FrequenciaCardiaca", variable_card=3,
        values=[[0.55], [0.30], [0.15]],
        state_names={"FrequenciaCardiaca": ["normal", "taquicardia", "grave"]},
    )

    cpd_dor = TabularCPD(
        variable="Dor", variable_card=3,
        values=[[0.40], [0.35], [0.25]],
        state_names={"Dor": ["leve", "moderada", "intensa"]},
    )

    cpd_idade = TabularCPD(
        variable="IdadeDoencaCronica", variable_card=2,
        values=[[0.65], [0.35]],
        state_names={"IdadeDoencaCronica": ["baixo_risco", "alto_risco"]},
    )

    # --- CPT da Gravidade | pais ---
    # 3 (Febre) x 3 (Saturacao) x 3 (Pressao) x 3 (FC) x 3 (Dor) x 2 (Idade)
    # = 486 combinações. Construímos programaticamente com uma função de
    # score aditivo (cada fator de risco contribui com um peso), depois
    # convertida em probabilidades por nível de gravidade. Isso evita ter
    # que escrever 486 linhas manualmente, mas o "espírito" dos valores
    # segue a lógica de tabelas como a do enunciado (ex.: Saturação crítica
    # -> maior peso para gravidade alta).

    estados_febre = ["ausente", "moderada", "alta"]
    estados_sat = ["normal", "reduzida", "critica"]
    estados_pa = ["normal", "baixa", "muito_baixa"]
    estados_fc = ["normal", "taquicardia", "grave"]
    estados_dor = ["leve", "moderada", "intensa"]
    estados_idade = ["baixo_risco", "alto_risco"]

    peso_febre = {"ausente": 0.0, "moderada": 0.4, "alta": 0.9}
    peso_sat = {"normal": 0.0, "reduzida": 0.6, "critica": 1.3}
    peso_pa = {"normal": 0.0, "baixa": 0.5, "muito_baixa": 1.1}
    peso_fc = {"normal": 0.0, "taquicardia": 0.4, "grave": 0.9}
    peso_dor = {"leve": 0.0, "moderada": 0.3, "intensa": 0.6}
    peso_idade = {"baixo_risco": 0.0, "alto_risco": 0.5}

    colunas = []
    ordem_pais = []
    for f in estados_febre:
        for s in estados_sat:
            for p in estados_pa:
                for fc in estados_fc:
                    for d in estados_dor:
                        for idd in estados_idade:
                            score = (
                                peso_febre[f] + peso_sat[s] + peso_pa[p]
                                + peso_fc[fc] + peso_dor[d] + peso_idade[idd]
                            )
                            # score varia de 0.0 (tudo normal) a ~5.3 (tudo crítico)
                            # Convertendo score em P(alta) via função logística,
                            # depois dividindo o restante entre média/baixa.
                            import math
                            p_alta = 1 / (1 + math.exp(-(score - 2.2) * 1.6))
                            p_alta = min(max(p_alta, 0.02), 0.95)
                            p_media = (1 - p_alta) * min(0.3 + score * 0.12, 0.75)
                            p_baixa = 1 - p_alta - p_media
                            colunas.append([p_baixa, p_media, p_alta])
                            ordem_pais.append((f, s, p, fc, d, idd))

    valores_gravidade = list(map(list, zip(*colunas)))  # transpõe: 3 x 486

    cpd_gravidade = TabularCPD(
        variable="Gravidade", variable_card=3,
        values=valores_gravidade,
        evidence=["Febre", "Saturacao", "PressaoArterial",
                  "FrequenciaCardiaca", "Dor", "IdadeDoencaCronica"],
        evidence_card=[3, 3, 3, 3, 3, 2],
        state_names={
            "Gravidade": ["baixa", "media", "alta"],
            "Febre": estados_febre,
            "Saturacao": estados_sat,
            "PressaoArterial": estados_pa,
            "FrequenciaCardiaca": estados_fc,
            "Dor": estados_dor,
            "IdadeDoencaCronica": estados_idade,
        },
    )

    modelo.add_cpds(cpd_febre, cpd_saturacao, cpd_pressao, cpd_freq_cardiaca,
                     cpd_dor, cpd_idade, cpd_gravidade)

    assert modelo.check_model(), "CPTs inconsistentes na rede bayesiana"
    return modelo


def inferir_gravidade(modelo, evidencias):
    """
    Retorna um dicionário {'baixa': p, 'media': p, 'alta': p} dado um
    dicionário de evidências, por exemplo:

    evidencias = {
        "Febre": "alta",
        "Saturacao": "critica",
        "PressaoArterial": "baixa",
        "Dor": "intensa",
        "IdadeDoencaCronica": "alto_risco",
    }

    Evidências ausentes são tratadas naturalmente pela inferência
    (a rede lida com dados faltantes por design).
    """
    infer = VariableElimination(modelo)
    resultado = infer.query(variables=["Gravidade"], evidence=evidencias, show_progress=False)
    estados = resultado.state_names["Gravidade"]
    return {estado: float(resultado.values[i]) for i, estado in enumerate(estados)}


if __name__ == "__main__":
    modelo = construir_rede()

    print("=== Exemplo de inferência (paciente do enunciado, seção 2.4) ===")
    evidencias_exemplo = {
        "Febre": "alta",
        "Saturacao": "critica",
        "PressaoArterial": "baixa",
        "Dor": "intensa",
        "IdadeDoencaCronica": "alto_risco",
        # FrequenciaCardiaca não informada -> rede lida com a ausência
    }
    resultado = inferir_gravidade(modelo, evidencias_exemplo)
    for k, v in resultado.items():
        print(f"P(Gravidade = {k}) = {v:.3f}")
