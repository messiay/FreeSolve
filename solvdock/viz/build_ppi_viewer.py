"""Builds standalone interactive 3D WebGL viewer for Induced-Fit PPI Docking (2PTC)."""

import os

def build_viewer():
    pdb_path = "data/docking/2ptc_ppi_docked_result.pdb"
    with open(pdb_path, "r") as f:
        pdb_data = f.read()

    # Escape backticks and backslashes for JS template literal
    clean_pdb = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SolvDock Induced-Fit Protein-Protein Docking (2PTC)</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        :root {{
            --bg-main: #090d16;
            --card-bg: rgba(15, 23, 42, 0.90);
            --border-color: rgba(255, 255, 255, 0.12);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --green: #10b981;
            --red: #ef4444;
            --cyan: #38bdf8;
            --gold: #f59e0b;
            --purple: #a855f7;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        #viewport {{
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            z-index: 1;
        }}
        #sidebar {{
            position: absolute;
            top: 16px; left: 16px; width: 420px;
            background: var(--card-bg);
            backdrop-filter: blur(18px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            z-index: 10;
            box-shadow: 0 24px 48px rgba(0, 0, 0, 0.7);
            max-height: calc(100vh - 32px);
            overflow-y: auto;
        }}
        .header h1 {{
            font-size: 19px;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 3px;
        }}
        .concept-alert {{
            margin-top: 12px;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 11.5px;
            line-height: 1.5;
            color: #bae6fd;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 14px;
        }}
        .stat-box {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 12px;
        }}
        .stat-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }}
        .stat-value {{
            font-size: 17px;
            font-weight: 700;
            margin-top: 3px;
        }}
        .badge {{
            display: inline-block;
            font-size: 9px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            margin-top: 4px;
        }}
        .badge-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .badge-cyan {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; }}
        .section-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        .btn-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .action-btn {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 9px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        .action-btn:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.25);
        }}
        .action-btn.active {{
            background: rgba(16, 185, 129, 0.15);
            border-color: var(--green);
            color: #34d399;
        }}
        .action-btn.active.blue {{
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--cyan);
            color: #38bdf8;
        }}
        .action-btn.active.purple {{
            background: rgba(168, 85, 247, 0.15);
            border-color: var(--purple);
            color: #c084fc;
        }}
        .action-btn.active.red {{
            background: rgba(239, 68, 68, 0.15);
            border-color: var(--red);
            color: #f87171;
        }}
        .action-btn.active.gold {{
            background: rgba(245, 158, 11, 0.15);
            border-color: var(--gold);
            color: #fbbf24;
        }}
        .legend {{
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
            font-size: 11px;
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        .dot {{
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }}
    </style>
</head>
<body>
    <div id="viewport"></div>

    <div id="sidebar">
        <div class="header">
            <h1>SolvDock Induced-Fit PPI</h1>
            <div class="subtitle">Bovine Trypsin (Chain E) + BPTI Inhibitor (2PTC)</div>
        </div>

        <div class="concept-alert">
            <strong>Why did ribbons look like regular docking?</strong><br>
            Cartoon ribbons <em>only</em> trace the backbone alpha-carbons (which are held stable). The induced-fit flexibility occurs in the <strong>169 interfacial sidechains (rotamers)</strong> twisting to avoid clashes. Use the modes below to see the sidechains pack!
        </div>

        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Clash Relief</div>
                <div class="stat-value" style="color: #34d399;">11 &rarr; 1</div>
                <span class="badge badge-green">90.9% Relief</span>
            </div>
            <div class="stat-box">
                <div class="stat-label">Pose RMSD</div>
                <div class="stat-value" style="color: #38bdf8;">1.349 &Aring;</div>
                <span class="badge badge-cyan">Backbone: 1.205 &Aring;</span>
            </div>
            <div class="stat-box">
                <div class="stat-label">Sidechain Max Disp</div>
                <div class="stat-value" style="color: #c084fc;">4.30 &Aring;</div>
                <span class="badge badge-cyan">169 Flexible Torsions</span>
            </div>
            <div class="stat-box">
                <div class="stat-label">Refinement Time</div>
                <div class="stat-value" style="color: #f59e0b;">4.40 s</div>
                <span class="badge badge-green">30 Adam Steps</span>
            </div>
        </div>

        <div class="section-title">Induced-Fit Inspection Modes</div>
        <div class="btn-group">
            <button class="action-btn active purple" id="btn-mode-sidechains" onclick="setMode('sidechains')">
                <span><span>&#128065;</span> <strong>Full Interface Sidechain Packing</strong></span>
                <span style="font-size: 10px; color: #c084fc;">All 45 Contact Residues</span>
            </button>
            <button class="action-btn" id="btn-mode-ribbon" onclick="setMode('ribbon')">
                <span><span>&#129525;</span> <strong>Ribbon Architecture Only</strong></span>
                <span style="font-size: 10px;">Backbone Tube</span>
            </button>
            <button class="action-btn" id="btn-mode-clash-inspect" onclick="setMode('compare')">
                <span><span>&#9888;</span> <strong>Clash Relief Comparison (Red vs Green)</strong></span>
                <span style="font-size: 10px; color: #f87171;">Before vs After</span>
            </button>
        </div>

        <div class="section-title">Inhibitor Pose Selection</div>
        <div class="btn-group">
            <button class="action-btn active" id="btn-docked" onclick="togglePose('docked')">
                <span><span class="dot" style="background: #10b981;"></span>SolvDock Docked BPTI</span>
                <span style="font-size: 10px; color: #34d399;">1.349 &Aring; (Active)</span>
            </button>
            <button class="action-btn red" id="btn-clash" onclick="togglePose('clash')">
                <span><span class="dot" style="background: #ef4444;"></span>Perturbed Clashing BPTI</span>
                <span style="font-size: 10px; color: #f87171;">11 Severe Clashes</span>
            </button>
            <button class="action-btn gold" id="btn-crystal" onclick="togglePose('crystal')">
                <span><span class="dot" style="background: #f59e0b;"></span>Native Crystal BPTI</span>
                <span style="font-size: 10px; color: #fbbf24;">X-ray Reference</span>
            </button>
        </div>

        <div class="section-title">Camera & Surface Controls</div>
        <div class="btn-group">
            <button class="action-btn" onclick="focusInterface()">
                <span>Zoom into Catalytic Interface</span>
                <span style="font-size: 10px; color: #38bdf8;">Lys15 &rarr; Asp189</span>
            </button>
            <button class="action-btn" id="btn-surf" onclick="toggleSurface()">
                <span>Trypsin Pocket Surface</span>
                <span id="surf-status" style="font-size: 10px;">Off</span>
            </button>
            <button class="action-btn" onclick="resetZoom()">
                <span>Reset Full Complex View</span>
                <span style="font-size: 10px;">Global</span>
            </button>
        </div>

        <div class="legend">
            <div><span class="dot" style="background: #38bdf8;"></span><strong>Trypsin (Chain E)</strong>: Sky Blue Ribbon & Sticks</div>
            <div><span class="dot" style="background: #10b981;"></span><strong>Docked BPTI (Chain I)</strong>: Emerald Green Ribbon & Sticks</div>
            <div><span class="dot" style="background: #ef4444;"></span><strong>Clashing Pose (Chain P / U)</strong>: Coral Red Sticks (Steric overlap)</div>
            <div style="margin-top: 6px; font-size: 10.5px; color: #94a3b8;">
                In <strong>Full Interface Sidechain Packing</strong>, all 31 Trypsin interface residues and 14 BPTI interface residues are shown as atomic sticks, revealing how sidechain rotamers interlock without penetrating.
            </div>
        </div>
    </div>

    <script>
        const PDB_DATA = `{clean_pdb}`;
        let viewer = null;
        let surfaceObj = null;
        let currentMode = 'sidechains';
        let showDocked = true;
        let showClash = false;
        let showCrystal = false;

        const TRYP_IFACE = [39, 40, 41, 42, 57, 58, 60, 94, 96, 97, 98, 99, 102, 151, 175, 189, 190, 191, 192, 193, 194, 195, 213, 214, 215, 216, 219, 220, 226, 227, 228];
        const BPTI_IFACE = [11, 12, 13, 14, 15, 16, 17, 18, 19, 34, 36, 37, 38, 39];

        window.addEventListener('DOMContentLoaded', () => {{
            const element = document.getElementById('viewport');
            const config = {{ backgroundColor: '#090d16' }};
            viewer = $3Dmol.createViewer(element, config);

            viewer.addModel(PDB_DATA, "pdb");
            applyStyles();
            focusInterface();
        }});

        function setMode(mode) {{
            currentMode = mode;
            document.getElementById('btn-mode-sidechains').classList.toggle('active', mode === 'sidechains');
            document.getElementById('btn-mode-ribbon').classList.toggle('active', mode === 'ribbon');
            document.getElementById('btn-mode-clash-inspect').classList.toggle('active', mode === 'compare');

            if (mode === 'compare') {{
                showDocked = true;
                showClash = true;
            }}
            applyStyles();
            viewer.render();
        }}

        function applyStyles() {{
            // 1. Trypsin Receptor (Chain E)
            viewer.setStyle({{ chain: 'E' }}, {{
                cartoon: {{ color: '#38bdf8', opacity: 0.85, thickness: 0.3 }}
            }});

            if (currentMode === 'sidechains') {{
                // Show all 31 interface contact residue sidechains as sticks
                viewer.addStyle({{ chain: 'E', resi: TRYP_IFACE }}, {{
                    stick: {{ radius: 0.16, colorscheme: 'cyanCarbon' }}
                }});
                // Key catalytic triad & specificity
                viewer.addStyle({{ chain: 'E', resi: [57, 102, 189, 195] }}, {{
                    stick: {{ radius: 0.24, colorscheme: 'cyanCarbon' }}
                }});
            }} else if (currentMode === 'ribbon') {{
                // Ribbon only, minimal sticks
                viewer.addStyle({{ chain: 'E', resi: [189, 195] }}, {{
                    stick: {{ radius: 0.18, colorscheme: 'cyanCarbon' }}
                }});
            }} else if (currentMode === 'compare') {{
                // Compare mode: relaxed Trypsin sticks in Cyan
                viewer.addStyle({{ chain: 'E', resi: TRYP_IFACE }}, {{
                    stick: {{ radius: 0.16, colorscheme: 'cyanCarbon' }}
                }});
            }}

            // 2. SolvDock Docked BPTI (Chain I)
            if (showDocked) {{
                viewer.setStyle({{ chain: 'I' }}, {{
                    cartoon: {{ color: '#10b981', opacity: 0.90, thickness: 0.35 }}
                }});
                if (currentMode === 'sidechains' || currentMode === 'compare') {{
                    // Show all 14 BPTI interface sidechains
                    viewer.addStyle({{ chain: 'I', resi: BPTI_IFACE }}, {{
                        stick: {{ radius: 0.18, colorscheme: 'greenCarbon' }}
                    }});
                    // Anchor Lys15
                    viewer.addStyle({{ chain: 'I', resi: [15] }}, {{
                        stick: {{ radius: 0.26, colorscheme: 'greenCarbon' }}
                    }});
                }} else {{
                    viewer.addStyle({{ chain: 'I', resi: [15] }}, {{
                        stick: {{ radius: 0.20, colorscheme: 'greenCarbon' }}
                    }});
                }}
            }} else {{
                viewer.setStyle({{ chain: 'I' }}, {{ hidden: true }});
            }}

            // 3. Clashing Perturbed BPTI (Chain P)
            if (showClash) {{
                viewer.setStyle({{ chain: 'P' }}, {{
                    cartoon: {{ color: '#ef4444', opacity: 0.60, thickness: 0.25 }}
                }});
                viewer.addStyle({{ chain: 'P', resi: BPTI_IFACE }}, {{
                    stick: {{ radius: 0.20, colorscheme: 'redCarbon' }}
                }});
            }} else {{
                viewer.setStyle({{ chain: 'P' }}, {{ hidden: true }});
            }}

            // 4. Native Crystal BPTI (Chain X)
            if (showCrystal) {{
                viewer.setStyle({{ chain: 'X' }}, {{
                    cartoon: {{ color: '#f59e0b', opacity: 0.70, thickness: 0.3 }}
                }});
                viewer.addStyle({{ chain: 'X', resi: BPTI_IFACE }}, {{
                    stick: {{ radius: 0.16, colorscheme: 'yellowCarbon' }}
                }});
            }} else {{
                viewer.setStyle({{ chain: 'X' }}, {{ hidden: true }});
            }}
        }}

        function togglePose(type) {{
            if (type === 'docked') {{
                showDocked = !showDocked;
                document.getElementById('btn-docked').classList.toggle('active', showDocked);
            }} else if (type === 'clash') {{
                showClash = !showClash;
                document.getElementById('btn-clash').classList.toggle('active', showClash);
            }} else if (type === 'crystal') {{
                showCrystal = !showCrystal;
                document.getElementById('btn-crystal').classList.toggle('active', showCrystal);
            }}
            applyStyles();
            viewer.render();
        }}

        function toggleSurface() {{
            const btn = document.getElementById('btn-surf');
            const status = document.getElementById('surf-status');
            if (surfaceObj) {{
                viewer.removeSurface(surfaceObj);
                surfaceObj = null;
                btn.classList.remove('active');
                status.innerText = 'Off';
            }} else {{
                surfaceObj = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.30,
                    color: '#38bdf8'
                }}, {{ chain: 'E', resi: TRYP_IFACE }});
                btn.classList.add('active');
                status.innerText = 'On';
            }}
            viewer.render();
        }}

        function focusInterface() {{
            viewer.zoomTo({{ chain: ['E', 'I'], resi: [15, 189, 195, 57, 102] }});
            viewer.render();
        }}

        function resetZoom() {{
            viewer.zoomTo();
            viewer.render();
        }}
    </script>
</body>
</html>
"""

    out_file = "data/viz/ppi_docking_viewer.html"
    os.makedirs("data/viz", exist_ok=True)
    with open(out_file, "w") as f:
        f.write(html_content)
    print(f"Generated standalone 3D PPI docking viewer: {out_file}")

if __name__ == "__main__":
    build_viewer()
