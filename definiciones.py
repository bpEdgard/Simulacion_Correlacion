# ─── Paletas de color para el área de gráficos (Plotly) ──────────────────────
# Mantienen la misma estructura que el Simulador de Muestreo y agregan claves
# específicas para el simulador de correlación:
#   "code"     → patrón de sincronismo (referencia limpia)
#   "peak"     → marcador del pico de correlación detectado
#   "noise"    → trazo del ruido aditivo
#   "received" → señal recibida con ruido
#   "real"     → marca del retardo verdadero (ground truth)

DARK = {
    "bg":            "#0a0e17",
    "paper":         "#0f172a",
    "grid":          "#1e2d4a",
    "font":          "#e2e8f0",
    "legend_border": "#1e293b",
    "signal":        "#22d3ee",        # señal A / original (cyan)
    "sampled":       "#f59e0b",        # muestras (amber) — heredado
    "rec":           "#a78bfa",        # señal reconstruida (violeta) — heredado
    "alias":         "#ef4444",
    "nyquist":       "#ef4444",
    "filtered":      "#10b981",        # resultado de correlación (verde)
    "replica":       "#475569",
    "zone_base":     "rgba(34,211,238,0.08)",
    "zone_mirror":   "rgba(239,68,68,0.07)",
    "zone_rep1":     "rgba(34,211,238,0.04)",
    "zone_rep2":     "rgba(239,68,68,0.04)",
    "zone_rep3":     "rgba(34,211,238,0.02)",
    "vline_mid":     "#334155",
    "ann_mid":       "#475569",
    "zone_label":    "#64748b",
    # ─── Específicos de correlación ─────────────────────────────────────────
    "code":          "#fbbf24",        # patrón de sincronismo (gold)
    "peak":          "#f43f5e",        # pico detectado (rosé)
    "real":          "#22d3ee",        # retardo real (cyan, igual que signal)
    "noise":         "#94a3b8",        # ruido (gris)
    "received":      "#a78bfa",        # señal recibida (violeta, igual que rec)
    "info_bg":       "#111827",
    "info_border":   "#1e293b",
}

LIGHT = {
    "bg":            "#ffffff",
    "paper":         "#f8fafc",
    "grid":          "#e2e8f0",
    "font":          "#1e293b",
    "legend_border": "#cbd5e1",
    "signal":        "#0891b2",
    "sampled":       "#d97706",
    "rec":           "#7c3aed",
    "alias":         "#dc2626",
    "nyquist":       "#dc2626",
    "filtered":      "#059669",
    "replica":       "#94a3b8",
    "zone_base":     "rgba(8,145,178,0.10)",
    "zone_mirror":   "rgba(220,38,38,0.08)",
    "zone_rep1":     "rgba(8,145,178,0.05)",
    "zone_rep2":     "rgba(220,38,38,0.05)",
    "zone_rep3":     "rgba(8,145,178,0.02)",
    "vline_mid":     "#94a3b8",
    "ann_mid":       "#64748b",
    "zone_label":    "#475569",
    # ─── Específicos de correlación ─────────────────────────────────────────
    "code":          "#b45309",
    "peak":          "#be123c",
    "real":          "#0891b2",
    "noise":         "#64748b",
    "received":      "#7c3aed",
    "info_bg":       "#f1f5f9",
    "info_border":   "#cbd5e1",
}
