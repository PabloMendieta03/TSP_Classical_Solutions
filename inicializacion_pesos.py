
"""
Este archivo define funciones para inicializar los pesos de modelos de redes neuronales usando distintas estrategias:

- init_xavier: inicialización Xavier (Glorot) uniforme
- init_he: inicialización Kaiming (He) uniforme
- init_orthogonal: inicialización ortogonal

La función apply_init permite aplicar cualquiera de estas estrategias a todo un modelo de PyTorch.
"""

# importanción de librerías
import torch
import torch.nn as nn


def init_xavier(module: nn.Module):
    """
    Aplica inicialización uniforme de Xavier (Glorot) a todos los pesos del módulo que sean matrices.

    Parámetros:
        module: Módulo de PyTorch cuyos pesos se inicializarán.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        try:
            if module.weight.dim() >= 2:
                nn.init.xavier_uniform_(module.weight)
        except Exception:
            # Ignorar tensores no compatibles
            pass
    if hasattr(module, 'bias') and module.bias is not None:
        try:
            nn.init.zeros_(module.bias)
        except Exception:
            pass


def init_he(module: nn.Module):
    """
    Aplica inicialización uniforme de Kaiming (He) a todos los pesos del módulo que sean matrices.
    Funciona mejor con activaciones ReLU.

    Parámetros:
        module: Módulo de PyTorch cuyos pesos se inicializarán.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        try:
            if module.weight.dim() >= 2:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        except Exception:
            pass
    if hasattr(module, 'bias') and module.bias is not None:
        try:
            nn.init.zeros_(module.bias)
        except Exception:
            pass


def init_orthogonal(module: nn.Module):
    """
    Aplica inicialización ortogonal a todos los pesos del módulo que sean matrices.

    Parámetros:
        module: Módulo de PyTorch cuyos pesos se inicializarán.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        try:
            if module.weight.dim() >= 2:
                nn.init.orthogonal_(module.weight)
        except Exception:
            pass
    if hasattr(module, 'bias') and module.bias is not None:
        try:
            nn.init.zeros_(module.bias)
        except Exception:
            pass


def apply_init(model: nn.Module, strategy: str):
    """
    Aplica una estrategia de inicialización al modelo completo.

    Args:
        model: Instancia de nn.Module (por ejemplo, tu modelo GNN) a inicializar.
        strategy: Cadena con la estrategia a usar. Opciones: ['xavier', 'he', 'orthogonal'].

    Raises:
        ValueError: Si la estrategia no está entre las permitidas.
    """
    strat = strategy.lower()
    if strat == 'xavier':
        func = init_xavier
    elif strat == 'he':
        func = init_he
    elif strat == 'orthogonal':
        func = init_orthogonal
    else:
        raise ValueError(f"Estrategia de inicialización desconocida: '{strategy}'")

    # Aplicar recursivamente, ignorando errores de inicialización específicos
    for module in model.modules():
        func(module)



