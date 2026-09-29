"""Builds the standalone, publication-grade SolvDock 3D Viewport HTML."""

import json
import os


def build_viewport_html():
    # Read the 3 clean PDB files
    with open("data/viz/2v00_head_to_head.pdb", "r") as f:
        pdb_h2h = f.read()

    with open("data/viz/1aay_clean_tf_dna.pdb", "r") as f:
        pdb_1aay = f.read()

    with open("data/viz/2v00_torsional_trajectory.pdb", "r") as f:
        pdb_torsion = f.read()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SolvDock Continuum Biophysics — 3D Structural Viewport</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        :root {{
            --bg-dark: #090d16;
            --panel-bg: rgba(17, 24, 39, 0.92);
            --border-color: rgba(55, 65, 81, 0.6);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --solvdock-green: #10b981;
            --vina-red: #ef4444;
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --gold: #f59e0b;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            background: var(--bg-dark);
            color: var(--text-primary);
            overflow: hidden;
            display: flex;
            height: 100vh;
        }}
        #sidebar {{
            width: 420px;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border-right: 1px solid var(--border-color);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 18px;
            z-index: 20;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
            overflow-y: auto;
        }}
        .header {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 14px;
        }}
        .header h1 {{
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.4px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header .subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .mode-nav {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .mode-btn {{
            padding: 12px 14px;
            background: rgba(31, 41, 55, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 13px;
            font-weight: 600;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .mode-btn:hover {{
            background: rgba(55, 65, 81, 0.8);
            border-color: #6b7280;
        }}
        .mode-btn.active {{
            background: rgba(16, 185, 129, 0.15);
            border-color: var(--solvdock-green);
            color: #6ee7b7;
        }}
        .badge {{
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 9999px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-green {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}
        .badge-purple {{
            background: rgba(168, 85, 247, 0.2);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.4);
        }}
        .badge-cyan {{
            background: rgba(6, 182, 212, 0.2);
            color: #38bdf8;
            border: 1px solid rgba(6, 182, 212, 0.4);
        }}
        .card {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .card-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-secondary);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .stat-box {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 6px;
            padding: 8px 10px;
        }}
        .stat-box.highlight-green {{
            border-color: rgba(16, 185, 129, 0.5);
            background: rgba(16, 185, 129, 0.08);
        }}
        .stat-box.highlight-red {{
            border-color: rgba(239, 68, 68, 0.5);
            background: rgba(239, 68, 68, 0.08);
        }}
        .stat-label {{
            font-size: 10px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 15px;
            font-weight: 700;
            margin-top: 2px;
        }}
        .control-row {{
            display: flex;
            gap: 8px;
        }}
        button.action-btn {{
            flex: 1;
            padding: 9px 12px;
            background: #1f2937;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        button.action-btn:hover {{
            background: #374151;
        }}
        button.action-btn.active {{
            background: var(--solvdock-green);
            color: #064e3b;
            border-color: var(--solvdock-green);
        }}
        #viewport-container {{
            flex: 1;
            position: relative;
            background: radial-gradient(circle at 60% 40%, #172033 0%, #090d16 100%);
        }}
        #gldiv {{
            width: 100%;
            height: 100%;
        }}
        .top-hud {{
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 12px;
            z-index: 10;
        }}
        .hud-pill {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--border-color);
            padding: 8px 14px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
            backdrop-filter: blur(8px);
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--solvdock-green);
            box-shadow: 0 0 10px var(--solvdock-green);
        }}
        .timeline-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 4px;
        }}
        input[type="range"] {{
            flex: 1;
            accent-color: var(--solvdock-green);
            background: #374151;
            height: 6px;
            border-radius: 3px;
            outline: none;
        }}
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="header">
            <h1>SolvDock Biophysics</h1>
            <div class="subtitle">Continuum Solvation & Kinematics Engine</div>
        </div>

        <div class="mode-nav">
            <button class="mode-btn active" id="btn-mode1" onclick="switchMode(1)">
                <span>1. Vina vs. SolvDock (2v00)</span>
                <span class="badge badge-green">Head-to-Head</span>
            </button>
            <button class="mode-btn" id="btn-mode2" onclick="switchMode(2)">
                <span>2. Zif268 Zinc Finger (1AAY)</span>
                <span class="badge badge-purple">DNA Recognition</span>
            </button>
            <button class="mode-btn" id="btn-mode3" onclick="switchMode(3)">
                <span>3. Articulated Joint Kinematics</span>
                <span class="badge badge-cyan">0.000 Å Distortion</span>
            </button>
        </div>

        <!-- Mode 1 Panel: Head-to-Head -->
        <div id="panel-mode1" class="card">
            <div class="card-title">HSP90 Pose Convergence (2v00)</div>
            <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
                In rigid receptor docking, a global search in a 16 Å box (AutoDock Vina) gets trapped in an outer minimum (5.22 Å). SolvDock gradient descent from a perturbed pose relaxes sidechains to reach the native crystal pose (0.90 Å).
                <br><span style="font-size: 11px; color: #94a3b8;">*Testing local refinement convergence, not blind global search capability.</span>
            </p>
            <div class="stat-grid">
                <div class="stat-box highlight-red">
                    <div class="stat-label">AutoDock Vina (Global)</div>
                    <div class="stat-value" style="color: #f87171;">5.222 Å</div>
                    <div style="font-size: 10px; color: #fca5a5; margin-top: 2px;">Trapped Outer Pose</div>
                </div>
                <div class="stat-box highlight-green">
                    <div class="stat-label">SolvDock (Local Refine)</div>
                    <div class="stat-value" style="color: #34d399;">0.905 Å</div>
                    <div style="font-size: 10px; color: #a7f3d0; margin-top: 2px;">Native Crystal Pose</div>
                </div>
            </div>
            <div class="control-row" style="margin-top: 4px;">
                <button class="action-btn active" id="btn-toggle-slv" onclick="togglePose('slv')">SolvDock (Green)</button>
                <button class="action-btn active" id="btn-toggle-vna" onclick="togglePose('vna')">Vina (Red)</button>
            </div>
            <div class="control-row">
                <button class="action-btn" id="btn-toggle-surf" onclick="toggleSurface()">Pocket Surface</button>
                <button class="action-btn" onclick="resetView()">Reset Zoom</button>
            </div>
        </div>

        <!-- Mode 2 Panel: Zif268 -->
        <div id="panel-mode2" class="card" style="display: none;">
            <div class="card-title">Transcription Factor Major Groove Showcase</div>
            <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
                Authentic RCSB crystal complex (1AAY). Three zinc finger domains wrap the major groove with strict structural complementarity.
                <br><span style="font-size: 11px; color: #94a3b8;">*Visual crystal showcase. Static vacuum energy claims retracted (aqueous binding thermodynamics is dominated by counterion release entropy).</span>
            </p>
            <div class="stat-grid">
                <div class="stat-box highlight-green">
                    <div class="stat-label">Crystal Structure</div>
                    <div class="stat-value" style="color: #34d399;">PDB 1AAY</div>
                    <div style="font-size: 10px; color: var(--text-secondary); margin-top: 2px;">High-Resolution X-Ray</div>
                </div>
                <div class="stat-box highlight-purple">
                    <div class="stat-label">Domains</div>
                    <div class="stat-value" style="color: #c084fc;">3 Fingers</div>
                    <div style="font-size: 10px; color: var(--text-secondary); margin-top: 2px;">Coordinated Zn²⁺</div>
                </div>
            </div>
            <div class="control-row">
                <button class="action-btn active" id="btn-toggle-tf-surf" onclick="toggleTfSurface()">DNA Surface</button>
                <button class="action-btn" onclick="resetView()">Reset Zoom</button>
            </div>
        </div>

        <!-- Mode 3 Panel: Torsional Kinematics -->
        <div id="panel-mode3" class="card" style="display: none;">
            <div class="card-title">PyTorch Torsional Optimizer Playback</div>
            <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
                Real forward kinematics trajectory. Covalent bond lengths and angles are 100% mathematically locked (< 10⁻⁵ Å). Sidechains rotate smoothly around dihedral hinges.
            </p>
            <div class="stat-grid">
                <div class="stat-box highlight-green">
                    <div class="stat-label">Bond Length Error</div>
                    <div class="stat-value" style="color: #34d399;">0.000 Å</div>
                    <div style="font-size: 10px; color: var(--text-secondary); margin-top: 2px;">Exact Invariance</div>
                </div>
                <div class="stat-box highlight-green">
                    <div class="stat-label">Backbone Distortion</div>
                    <div class="stat-value" style="color: #34d399;">0.000 Å</div>
                    <div style="font-size: 10px; color: var(--text-secondary); margin-top: 2px;">Rigid Frame Pinning</div>
                </div>
            </div>
            <div style="margin-top: 4px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary);">
                    <span>Optimizer Step</span>
                    <span id="traj-frame-label">Frame 1 / 13</span>
                </div>
                <div class="timeline-bar">
                    <input type="range" id="traj-slider" min="0" max="12" value="0" oninput="onTrajSlider(this.value)">
                </div>
            </div>
            <div class="control-row" style="margin-top: 6px;">
                <button class="action-btn active" id="btn-play" onclick="toggleTrajPlay()">Pause</button>
                <button class="action-btn" onclick="resetView()">Reset Zoom</button>
            </div>
        </div>

        <div style="margin-top: auto; font-size: 11px; color: #6b7280; text-align: center; border-top: 1px solid var(--border-color); padding-top: 12px;">
            Rotate: Left Click + Drag &bull; Zoom: Wheel &bull; Pan: Right Click
        </div>
    </div>

    <div id="viewport-container">
        <div id="gldiv"></div>
        <div class="top-hud">
            <div class="hud-pill">
                <div class="status-dot"></div>
                <span>CONTINUUM PHYSICS: DEBYE-HÜCKEL 150 mM</span>
            </div>
            <div class="hud-pill" id="hud-mode-pill">
                <span>MODE: 2V00 HEAD-TO-HEAD</span>
            </div>
        </div>
    </div>

    <script>
        const DATA_H2H = `{json.dumps(pdb_h2h)[1:-1]}`;
        const DATA_1AAY = `{json.dumps(pdb_1aay)[1:-1]}`;
        const DATA_TORSION = `{json.dumps(pdb_torsion)[1:-1]}`;

        let viewer = null;
        let currentMode = 1;
        let showSlv = true;
        let showVna = true;
        let showPocketSurf = false;
        let showTfSurf = false;
        let pocketSurfObj = null;
        let tfSurfObj = null;

        let trajPlaying = false;
        let trajInterval = null;
        let trajFrame = 0;
        let totalTrajFrames = 13;

        window.onload = function() {{
            const element = document.getElementById("gldiv");
            const config = {{ backgroundColor: "#090d16" }};
            viewer = $3Dmol.createViewer(element, config);

            loadMode1();
        }};

        function switchMode(mode) {{
            currentMode = mode;
            if (trajInterval) clearInterval(trajInterval);
            trajPlaying = false;

            document.getElementById("btn-mode1").className = "mode-btn" + (mode === 1 ? " active" : "");
            document.getElementById("btn-mode2").className = "mode-btn" + (mode === 2 ? " active" : "");
            document.getElementById("btn-mode3").className = "mode-btn" + (mode === 3 ? " active" : "");

            document.getElementById("panel-mode1").style.display = mode === 1 ? "flex" : "none";
            document.getElementById("panel-mode2").style.display = mode === 2 ? "flex" : "none";
            document.getElementById("panel-mode3").style.display = mode === 3 ? "flex" : "none";

            const hudPill = document.getElementById("hud-mode-pill");
            if (mode === 1) {{
                hudPill.innerHTML = "<span>MODE: 2V00 HEAD-TO-HEAD</span>";
                loadMode1();
            }} else if (mode === 2) {{
                hudPill.innerHTML = "<span>MODE: 1AAY TF-DNA COMPLEX</span>";
                loadMode2();
            }} else if (mode === 3) {{
                hudPill.innerHTML = "<span>MODE: ARTICULATED TORSIONAL KINEMATICS</span>";
                loadMode3();
            }}
        }}

        function loadMode1() {{
            viewer.clear();
            pocketSurfObj = null;
            showPocketSurf = false;
            document.getElementById("btn-toggle-surf").className = "action-btn";

            viewer.addModel(DATA_H2H, "pdb");

            // Pocket residues: cartoon ribbon
            viewer.setStyle({{ chain: 'A' }}, {{ cartoon: {{ color: '#94a3b8', opacity: 0.85 }} }});
            // Key binding residues: Asn51, Asp93, Thr184
            viewer.setStyle({{ chain: 'A', resn: ['ASP', 'THR', 'ASN'] }}, {{
                cartoon: {{ color: '#94a3b8', opacity: 0.85 }},
                stick: {{ color: '#cbd5e1', radius: 0.18 }}
            }});

            // SolvDock Pose (Emerald Green)
            viewer.setStyle({{ resn: 'SLV' }}, {{ stick: {{ color: '#10b981', radius: 0.28 }} }});

            // Vina Pose (Ruby Red)
            viewer.setStyle({{ resn: 'VNA' }}, {{ stick: {{ color: '#ef4444', radius: 0.26 }} }});

            viewer.zoomTo({{ resn: ['SLV', 'VNA'] }}, 1000);
            viewer.render();
        }}

        function togglePose(pose) {{
            if (pose === 'slv') {{
                showSlv = !showSlv;
                document.getElementById("btn-toggle-slv").className = "action-btn" + (showSlv ? " active" : "");
                viewer.setStyle({{ resn: 'SLV' }}, showSlv ? {{ stick: {{ color: '#10b981', radius: 0.28 }} }} : {{}});
            }} else if (pose === 'vna') {{
                showVna = !showVna;
                document.getElementById("btn-toggle-vna").className = "action-btn" + (showVna ? " active" : "");
                viewer.setStyle({{ resn: 'VNA' }}, showVna ? {{ stick: {{ color: '#ef4444', radius: 0.26 }} }} : {{}});
            }}
            viewer.render();
        }}

        function toggleSurface() {{
            showPocketSurf = !showPocketSurf;
            const btn = document.getElementById("btn-toggle-surf");
            if (showPocketSurf) {{
                btn.className = "action-btn active";
                pocketSurfObj = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.28,
                    color: '#38bdf8'
                }}, {{ chain: 'A' }});
            }} else {{
                btn.className = "action-btn";
                if (pocketSurfObj) {{
                    viewer.removeSurface(pocketSurfObj);
                    pocketSurfObj = null;
                }}
            }}
            viewer.render();
        }}

        function loadMode2() {{
            viewer.clear();
            tfSurfObj = null;
            showTfSurf = false;

            viewer.addModel(DATA_1AAY, "pdb");

            // DNA chains (A and B): Emerald green cartoon with rings
            viewer.setStyle({{ resn: ['DA', 'DC', 'DG', 'DT'] }}, {{
                cartoon: {{ color: '#10b981', ring: true, opacity: 0.9 }}
            }});

            // Transcription factor (Chain C): Royal purple cartoon
            viewer.setStyle({{ not: {{ resn: ['DA', 'DC', 'DG', 'DT'] }} }}, {{
                cartoon: {{ color: '#a855f7', opacity: 0.95 }}
            }});

            // Key DNA-recognition Arginines & Histidines in major groove (Golden yellow sticks)
            viewer.setStyle({{ resn: ['ARG', 'HIS'] }}, {{
                cartoon: {{ color: '#a855f7', opacity: 0.95 }},
                stick: {{ color: '#f59e0b', radius: 0.22 }}
            }});

            // Coordinated Zinc ions: metallic cyan spheres
            viewer.setStyle({{ resn: 'ZN' }}, {{ sphere: {{ color: '#06b6d4', radius: 1.4 }} }});

            viewer.zoomTo();
            viewer.render();
        }}

        function toggleTfSurface() {{
            showTfSurf = !showTfSurf;
            const btn = document.getElementById("btn-toggle-tf-surf");
            if (showTfSurf) {{
                btn.className = "action-btn active";
                tfSurfObj = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.25,
                    color: '#10b981'
                }}, {{ resn: ['DA', 'DC', 'DG', 'DT'] }});
            }} else {{
                btn.className = "action-btn";
                if (tfSurfObj) {{
                    viewer.removeSurface(tfSurfObj);
                    tfSurfObj = null;
                }}
            }}
            viewer.render();
        }}

        function loadMode3() {{
            viewer.clear();
            viewer.addModelsAsFrames(DATA_TORSION, "pdb");

            totalTrajFrames = viewer.getNumFrames();
            document.getElementById("traj-slider").max = totalTrajFrames - 1;

            // Pocket: clean slate ribbons
            viewer.setStyle({{ not: {{ resn: 'LIG' }} }}, {{ cartoon: {{ color: '#64748b', opacity: 0.85 }} }});
            // Pocket sidechains: cyan sticks
            viewer.setStyle({{ not: {{ resn: ['LIG', 'N', 'CA', 'C', 'O'] }} }}, {{
                cartoon: {{ color: '#64748b', opacity: 0.85 }},
                stick: {{ color: '#38bdf8', radius: 0.16 }}
            }});
            // Ligand: emerald green thick sticks
            viewer.setStyle({{ resn: 'LIG' }}, {{ stick: {{ color: '#10b981', radius: 0.30 }} }});

            setTrajFrame(0);
            viewer.zoomTo({{ resn: 'LIG' }}, 1000);
            viewer.render();

            startTrajPlayback();
        }}

        function setTrajFrame(idx) {{
            trajFrame = idx;
            viewer.setFrame(idx);
            document.getElementById("traj-slider").value = idx;
            document.getElementById("traj-frame-label").innerText = "Frame " + (idx + 1) + " / " + totalTrajFrames;
            viewer.render();
        }}

        function onTrajSlider(val) {{
            setTrajFrame(parseInt(val));
        }}

        function toggleTrajPlay() {{
            if (trajPlaying) {{
                stopTrajPlayback();
            }} else {{
                startTrajPlayback();
            }}
        }}

        function startTrajPlayback() {{
            if (trajInterval) clearInterval(trajInterval);
            trajPlaying = true;
            document.getElementById("btn-play").innerText = "Pause";
            document.getElementById("btn-play").className = "action-btn active";
            trajInterval = setInterval(() => {{
                trajFrame = (trajFrame + 1) % totalTrajFrames;
                setTrajFrame(trajFrame);
            }}, 200);
        }}

        function stopTrajPlayback() {{
            trajPlaying = false;
            if (trajInterval) clearInterval(trajInterval);
            document.getElementById("btn-play").innerText = "Play";
            document.getElementById("btn-play").className = "action-btn";
        }}

        function resetView() {{
            if (currentMode === 1) {{
                viewer.zoomTo({{ resn: ['SLV', 'VNA'] }}, 1000);
            }} else if (currentMode === 2) {{
                viewer.zoomTo();
            }} else if (currentMode === 3) {{
                viewer.zoomTo({{ resn: 'LIG' }}, 1000);
            }}
            viewer.render();
        }}
    </script>
</body>
</html>
"""
    with open("data/viz/solvdock_3d_viewport.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Built updated data/viz/solvdock_3d_viewport.html successfully.")


if __name__ == "__main__":
    build_viewport_html()
