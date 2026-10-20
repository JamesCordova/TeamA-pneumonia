#!/usr/bin/env python
"""
Renderiza a imagen las ecuaciones de las secciones de tesis.

Las ecuaciones se insertan en el documento como imágenes, de modo que deben
poder regenerarse. Este script vive en el repositorio precisamente por eso: en
una ocasión anterior los generadores quedaron en un directorio temporal y se
perdieron al limpiarlo.

Salida:
    pruebasg/results/tesis/ecuaciones/*.png
    pruebasg/results/tesis/ecuaciones/manifiesto.json   (ruta y tamaño de cada una)

Uso:
    python pruebasg/render_ecuaciones.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PRUEBAS_DIR = Path(__file__).resolve().parent
ROOT = PRUEBAS_DIR.parent
sys.path.insert(0, str(PRUEBAS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from _common import RESULTS_DIR

OUT = RESULTS_DIR / "tesis" / "ecuaciones"
DPI = 220

ECUACIONES = {
    # --- perfilador (3.6.4) ---
    "t_perfil": r"$v(y) = \left(n,\; \bar{y},\; ID,\; F_T,\; F_S,\; H,\; \rho_1,\; \rho_{52},\; p_0,\; \dots\right)$",
    "t_ft": r"$F_T = \max\left(0,\; 1 - \frac{Var(R_t)}{Var(T_t + R_t)}\right)$",
    "t_fs": r"$F_S = \max\left(0,\; 1 - \frac{Var(R_t)}{Var(S_t + R_t)}\right)$",
    "t_disp": r"$ID = \frac{s^2}{\bar{y}}$",
    "t_ent": r"$H = -\frac{1}{\ln m}\sum_{i=1}^{m} p_i\,\ln p_i, \qquad p_i = \frac{f(\omega_i)}{\sum_{j=1}^{m} f(\omega_j)}$",
    "t_p0": r"$p_0 = \frac{1}{n}\sum_{t=1}^{n}\mathbf{1}\left[y_t = 0\right]$",

    # --- conjunto equivalente (3.8.3) ---
    "t_dif": r"$d_k = \left|e_k^{(A)}\right| - \left|e_k^{(B)}\right|, \qquad k = 1,\dots,K$",
    "t_t": r"$t = \frac{\bar{d}}{s_d / \sqrt{K}} \;\sim\; t_{K-1}$",
    "t_empate": r"$\mathcal{E}_i = \left\{\, m \in \mathcal{M} \;:\; p_{m} \geq \alpha \,\right\} \cup \left\{m_i^{*}\right\}$",

    # --- criterio (3.10) ---
    "t_selector": r"$s: \; v(y) \;\longmapsto\; m \in \mathcal{M}$",
    "t_descarte": r"$n_{sec} = m_{tr} - L \;<\; N_{par} \;\Longrightarrow\; \mathcal{M} \leftarrow \mathcal{M}\setminus\{\mathrm{LSTM}\}$",
    "t_poisson": r"$CV_{Poisson} = \frac{\sqrt{\lambda}}{\lambda} = \frac{1}{\sqrt{\lambda}}$",
    "t_umbral": r"$\lambda = 10 \;\Rightarrow\; CV = 32\,\%, \qquad \lambda = 40 \;\Rightarrow\; CV = 16\,\%$",
    "t_cobertura": r"$C = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left[\,s(v_i) \in \mathcal{E}_i\,\right]$",
    "t_regret": r"$r_i = \frac{MAE_{s(v_i)} - MAE_{m_i^{*}}}{MAE_{m_i^{*}}}, \qquad \bar{r} = \frac{1}{N}\sum_{i=1}^{N} r_i$",
    "t_ahorro": r"$A = 1 - \frac{\sum_{i=1}^{N} c\left(s(v_i)\right)}{\sum_{i=1}^{N}\sum_{m \in \mathcal{M}} c(m)}$",

    # --- explicabilidad (3.11) ---
    "t_shapley": (r"$\phi_j = \sum_{S \subseteq F\setminus\{j\}} "
                  r"\frac{|S|!\,\left(|F| - |S| - 1\right)!}{|F|!}"
                  r"\left[\,f_{S\cup\{j\}}(x) - f_S(x)\,\right]$"),
    "t_shap_efic": r"$f(x) = \phi_0 + \sum_{j=1}^{|F|}\phi_j$",
    "t_shap_global": r"$\Phi_j = \frac{1}{n}\sum_{i=1}^{n}\left|\phi_{ij}\right|$",
    "t_shap_familia": r"$\Phi_{\mathcal{G}} = \frac{\sum_{j \in \mathcal{G}}\Phi_j}{\sum_{j=1}^{|F|}\Phi_j}$",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifiesto, fallos = {}, []
    for clave, tex in ECUACIONES.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        try:
            fig.text(0, 0, tex, fontsize=15, color="black")
            ruta = OUT / f"{clave}.png"
            fig.savefig(ruta, dpi=DPI, bbox_inches="tight", pad_inches=0.06,
                        facecolor="white", transparent=False)
            plt.close(fig)
            with Image.open(ruta) as im:
                w, h = im.size
            # Pixeles a 96 ppp, que es la unidad que espera docx-js
            manifiesto[clave] = {"file": str(ruta).replace("\\", "/"),
                                 "w": round(w * 96 / DPI), "h": round(h * 96 / DPI)}
        except Exception as exc:  # noqa: BLE001
            plt.close(fig)
            fallos.append((clave, str(exc)[:160]))

    (OUT / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=1), encoding="utf-8")
    print(f"renderizadas: {len(manifiesto)}   fallidas: {len(fallos)}")
    for k, e in fallos:
        print("  FALLO", k, "->", e)
    if manifiesto:
        ancho_max = max(v["w"] for v in manifiesto.values())
        print(f"ancho máximo: {ancho_max} px (el área útil de una página A4 son ~560)")
        excede = [k for k, v in manifiesto.items() if v["w"] > 560]
        if excede:
            print("  exceden el ancho:", ", ".join(excede))
    print(f"Manifiesto: {(OUT / 'manifiesto.json').relative_to(ROOT)}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
