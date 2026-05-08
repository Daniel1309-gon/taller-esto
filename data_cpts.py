import pandas as pd
from models import BayesianNode, BayesianNetwork

def create_telecom_network():
    bn = BayesianNetwork()

    # Define Nodes
    clima = BayesianNode("Clima_Severo")
    mantenimiento = BayesianNode("Mantenimiento_Programado")
    corte = BayesianNode("Corte_Electrico")
    dns = BayesianNode("Caida_DNS")
    router = BayesianNode("Fallo_Router")
    latencia = BayesianNode("Latencia_Alta")
    internet = BayesianNode("Sin_Internet")
    pagina = BayesianNode("Pagina_No_Carga")

    # Add Nodes to Network
    for node in [clima, mantenimiento, corte, dns, router, latencia, internet, pagina]:
        bn.add_node(node)

    # Define Edges (Topology)
    bn.add_edge("Clima_Severo", "Corte_Electrico")
    bn.add_edge("Corte_Electrico", "Caida_DNS")
    bn.add_edge("Corte_Electrico", "Fallo_Router")
    bn.add_edge("Mantenimiento_Programado", "Fallo_Router")
    bn.add_edge("Fallo_Router", "Latencia_Alta")
    bn.add_edge("Fallo_Router", "Sin_Internet")
    bn.add_edge("Caida_DNS", "Sin_Internet")
    bn.add_edge("Caida_DNS", "Pagina_No_Carga")
    bn.add_edge("Latencia_Alta", "Pagina_No_Carga")

    # Define CPTs (Using 1 for True, 0 for False)
    
    # Root: Clima_Severo
    clima.set_cpt(pd.DataFrame({
        'prob': [0.05] # P(C=T)
    }))

    # Root: Mantenimiento_Programado
    mantenimiento.set_cpt(pd.DataFrame({
        'prob': [0.1] # P(M=T)
    }))

    # Corte_Electrico | Clima_Severo
    corte.set_cpt(pd.DataFrame({
        'Clima_Severo': [1, 0],
        'prob': [0.8, 0.01] # P(CE=T | C)
    }))

    # Caida_DNS | Corte_Electrico
    dns.set_cpt(pd.DataFrame({
        'Corte_Electrico': [1, 0],
        'prob': [0.3, 0.02] # P(D=T | CE)
    }))

    # Fallo_Router | Corte_Electrico, Mantenimiento_Programado
    router.set_cpt(pd.DataFrame({
        'Corte_Electrico': [1, 1, 0, 0],
        'Mantenimiento_Programado': [1, 0, 1, 0],
        'prob': [0.95, 0.5, 0.8, 0.01] # P(R=T | CE, M)
    }))

    # Latencia_Alta | Fallo_Router
    latencia.set_cpt(pd.DataFrame({
        'Fallo_Router': [1, 0],
        'prob': [0.9, 0.05] # P(L=T | R)
    }))

    # Sin_Internet | Fallo_Router, Caida_DNS
    internet.set_cpt(pd.DataFrame({
        'Fallo_Router': [1, 1, 0, 0],
        'Caida_DNS': [1, 0, 1, 0],
        'prob': [0.99, 0.9, 0.7, 0.001] # P(I=T | R, D)
    }))

    # Pagina_No_Carga | Caida_DNS, Latencia_Alta
    pagina.set_cpt(pd.DataFrame({
        'Caida_DNS': [1, 1, 0, 0],
        'Latencia_Alta': [1, 0, 1, 0],
        'prob': [0.99, 0.8, 0.6, 0.01] # P(P=T | D, L)
    }))

    return bn

if __name__ == "__main__":
    bn = create_telecom_network()
    order = bn.get_topological_order()
    print("Orden Topológico:", order)
    for name in order:
        node = bn.get_node(name)
        print(f"\nCPT for {name}:")
        print(node.cpt)
