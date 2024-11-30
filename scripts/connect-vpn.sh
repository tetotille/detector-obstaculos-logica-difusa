#!/bin/bash

# Nombre de la interfaz de WireGuard
WG_INTERFACE="wg0"

# Verificar si la interfaz está activa
if ip a show "$WG_INTERFACE" up &>/dev/null; then
    echo "La interfaz $WG_INTERFACE está activa."
    
    # Verificar conexión a la VPN
    WG_PEERS=$(wg show "$WG_INTERFACE" peers)
    if [ -n "$WG_PEERS" ]; then
        echo "Conectado a la VPN de WireGuard."
    else
        echo "La interfaz $WG_INTERFACE está activa, pero no hay peers conectados."
    fi
else
    echo "No estás conectado a la VPN de WireGuard."
fi