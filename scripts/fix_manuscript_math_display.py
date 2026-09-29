"""
Format mathematical notation in MANUSCRIPT_FreeSolvE_Preprint.md to include
native Unicode superscripts and subscripts (r⁻¹², 10⁸, 10⁶, ℝ³, 𝕋ᵏ, C¹, etc.)
so that the text displays correctly in raw text/markdown editors without requiring MathJax.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("MANUSCRIPT_FreeSolvE_Preprint.md", "r", encoding="utf-8") as f:
    text = f.read()

replacements = [
    # Powers and superscripts
    (r"$10^6$", "10⁶ ($10^6$)"),
    (r"$r^{-12}$", "r⁻¹² ($r^{-12}$)"),
    (r"10^8\text{ kcal/(mol}\cdot\text{\AA)}", "10⁸ kcal/(mol·Å) ($10^8$)"),
    (r"$C^1$", "C¹ ($C^1$)"),
    (r"$C^1$-continuous", "C¹-continuous ($C^1$)"),
    (r"**$C^1$ Continuity**", "**C¹ Continuity ($C^1$ Continuity)**"),
    (r"\mathbb{R}^3", "ℝ³"),
    (r"\mathbb{T}^k", "𝕋ᵏ"),
    (r"\|\mathbf{k}\|^2", "‖k‖² ($|\\mathbf{k}|^2$)"),
    (r"$\epsilon_r \approx 78.4$", "εᵣ ≈ 78.4 ($\epsilon_r \\approx 78.4$)"),
    (r"$\Delta G_{\text{cavity}} \propto \text{SASA}$", "ΔG_cavity ∝ SASA ($\\Delta G_{\\text{cavity}} \\propto \\text{SASA}$)"),
    (r"$r_{ij}$", "r_ij ($r_{ij}$)"),
    (r"$E_0(r_{ij}) \le E_{\text{cap}}$", "E_0(r_ij) ≤ E_cap ($E_0(r_{ij}) \\le E_{\\text{cap}}$)"),
    (r"(where $E_{\text{cap}} = 25.0\text{ kcal/mol}$)", "(where E_cap = 25.0 kcal/mol, $E_{\\text{cap}} = 25.0\\text{ kcal/mol}$)"),
    (r"($E_0(r_{ij}) > E_{\text{cap}}$", "(E_0(r_ij) > E_cap, $E_0(r_{ij}) > E_{\\text{cap}}$"),
]

for target, repl in replacements:
    if target in text:
        text = text.replace(target, repl)
        print(f"Replaced: {repr(target)[:25]}... -> {repr(repl)[:25]}...")
    else:
        print(f"NOT FOUND: {repr(target)[:25]}...")

with open("MANUSCRIPT_FreeSolvE_Preprint.md", "w", encoding="utf-8") as f:
    f.write(text)

print("Manuscript updated successfully!")
