# TSP_Classical_Solutions

Creación de algoritmos clásicos para la resolución del TSP 


Pasos para la preparación del entorno: 

1. pip install -r requirements.txt



Para Guardar las bases de datos creadas: 


import torch

# --- Una vez generados tus TSPs:
# TSPs = [...]
torch.save(TSPs, "tsps.pt")


import torch

TSPs = torch.load("tsps.pt")
