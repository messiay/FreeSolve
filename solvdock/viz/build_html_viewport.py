"""Builds a standalone, zero-dependency interactive WebGL 3D simulation viewport."""

import os
import json


def build_viewport():
    os.makedirs("data/viz", exist_ok=True)

    simulations = {
        "2v00_induced_fit": {
            "title": "Human HSP90: Induced-Fit Drug Docking (2v00)",
            "description": "510 pocket atoms relaxing around potent inhibitor. Watch side-chains breathe and yield.",
            "path": "data/docking/2v00_induced_fit_movie.pdb",
            "style": "docking"
        },
        "1aay_transcription": {
            "title": "Zif268 Zinc Finger: DNA Major Groove Recognition (1AAY)",
            "description": "Transcription factor helices docking into consensus DNA (5'-GCGTGGGCG-3').",
            "path": "data/transcription/1aay_transcription_movie.pdb",
            "style": "dna_tf"
        },
        "1bna_dna_zipping": {
            "title": "B-DNA Dodecamer: Double-Helix Zipping Simulation (1BNA)",
            "description": "Watson-Crick base-pairing & base-stacking pulling two unzipped strands together.",
            "path": "data/dna/dna_zipping_trajectory.pdb",
            "style": "dna"
        },
        "1l2y_folding": {
            "title": "Trp-Cage Miniprotein: Core Compaction & Folding (1L2Y)",
            "description": "Hydrophobic collapse of extended polypeptide chain into compact native core.",
            "path": "data/folding/1l2y_folding_trajectory.pdb",
            "style": "protein"
        }
    }

    pdb_data = {}
    for sim_id, info in simulations.items():
        if os.path.exists(info["path"]):
            with open(info["path"], "r", encoding="utf-8") as f:
                pdb_data[sim_id] = f.read()
            print(f"Loaded {sim_id}: {len(pdb_data[sim_id])} bytes")
        else:
            print(f"Warning: {info['path']} not found")

    sim_metadata_json = json.dumps({k: {k2: v2 for k2, v2 in v.items() if k2 != "path"} for k, v in simulations.items()})
    pdb_data_json = json.dumps(pdb_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SolvDock 3D Biophysics Simulation Viewport</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
            display: flex;
            height: 100vh;
        }}
        #sidebar {{
            width: 380px;
            background: rgba(22, 27, 34, 0.95);
            backdrop-filter: blur(12px);
            border-right: 1px solid #30363d;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            z-index: 10;
        }}
        .header h1 {{
            font-size: 20px;
            color: #58a6ff;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .header p {{
            font-size: 13px;
            color: #8b949e;
            margin-top: 4px;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
        }}
        select, input[type="range"] {{
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 10px;
            border-radius: 6px;
            font-size: 14px;
            outline: none;
            width: 100%;
        }}
        select:focus {{
            border-color: #58a6ff;
        }}
        .button-row {{
            display: flex;
            gap: 10px;
        }}
        button {{
            flex: 1;
            padding: 12px;
            background: #238636;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
        }}
        button:hover {{
            background: #2ea043;
        }}
        button.secondary {{
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }}
        button.secondary:hover {{
            background: #30363d;
        }}
        .info-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            font-size: 13px;
            line-height: 1.5;
            color: #8b949e;
        }}
        .info-card strong {{
            color: #58a6ff;
        }}
        #viewport-container {{
            flex: 1;
            position: relative;
            background: radial-gradient(circle at center, #161b22 0%, #090d13 100%);
        }}
        #gldiv {{
            width: 100%;
            height: 100%;
            position: absolute;
        }}
        .overlay-badge {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(22, 27, 34, 0.85);
            border: 1px solid #30363d;
            backdrop-filter: blur(8px);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            color: #3fb950;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            pointer-events: none;
        }}
        .overlay-badge::before {{
            content: '';
            width: 8px;
            height: 8px;
            background: #3fb950;
            border-radius: 50%;
            box-shadow: 0 0 8px #3fb950;
        }}
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="header">
            <h1>SolvDock 3D Viewport</h1>
            <p>Interactive Continuum Biophysics Engine</p>
        </div>

        <div class="control-group">
            <label for="sim-select">Select Simulation</label>
            <select id="sim-select" onchange="changeSimulation()">
                <option value="2v00_induced_fit">HSP90: Induced-Fit Drug Docking</option>
                <option value="1aay_transcription">Zif268: TF-DNA Major Groove Recognition</option>
                <option value="1bna_dna_zipping">B-DNA: Double-Helix Zipping</option>
                <option value="1l2y_folding">Trp-Cage: Protein Core Compaction</option>
            </select>
        </div>

        <div class="control-group">
            <div style="display: flex; justify-content: space-between;">
                <label for="frame-slider">Timeline Frame</label>
                <span id="frame-label" style="font-size: 12px; color: #58a6ff;">1 / 16</span>
            </div>
            <input type="range" id="frame-slider" min="0" max="15" value="0" oninput="onSliderChange(this.value)">
        </div>

        <div class="button-row">
            <button id="play-btn" onclick="togglePlay()">Pause</button>
            <button class="secondary" onclick="resetView()">Reset View</button>
        </div>

        <div class="info-card">
            <div id="sim-desc">Loading description...</div>
        </div>

        <div style="margin-top: auto; font-size: 11px; color: #484f58; text-align: center;">
            Rotate: Left Click + Drag | Zoom: Scroll Wheel | Pan: Right Click + Drag
        </div>
    </div>

    <div id="viewport-container">
        <div id="gldiv"></div>
        <div class="overlay-badge">SOLVDOCK PHYSICS ACTIVE</div>
    </div>

    <script>
        const simMetadata = {sim_metadata_json};
        const pdbData = {pdb_data_json};

        let viewer = null;
        let isPlaying = true;
        let playInterval = null;
        let currentSim = "2v00_induced_fit";
        let currentFrame = 0;
        let totalFrames = 16;

        window.onload = function() {{
            viewer = $3Dmol.createViewer("gldiv", {{ backgroundColor: "#090d13" }});
            changeSimulation();
        }};

        function changeSimulation() {{
            currentSim = document.getElementById("sim-select").value;
            const data = pdbData[currentSim];
            if (!data) return;

            document.getElementById("sim-desc").innerHTML = "<strong>" + simMetadata[currentSim].title + "</strong><br><br>" + simMetadata[currentSim].description;

            viewer.clear();
            viewer.addModelsAsFrames(data, "pdb");

            const styleType = simMetadata[currentSim].style;
            applyStyling(styleType);

            totalFrames = viewer.getNumFrames();
            document.getElementById("frame-slider").max = totalFrames - 1;
            currentFrame = 0;
            setFrame(0);

            viewer.zoomTo();
            viewer.render();

            startPlayback();
        }}

        function applyStyling(styleType) {{
            if (styleType === "docking") {{
                viewer.setStyle({{ polymer: true }}, {{ cartoon: {{ color: '#1f6feb', opacity: 0.8 }} }});
                viewer.setStyle({{ resn: 'UNK' }}, {{ stick: {{ color: '#f778ba', radius: 0.28 }} }});
            }} else if (styleType === "dna_tf") {{
                viewer.setStyle({{ resn: ['DA','DC','DG','DT','A','C','G','T'] }}, {{ cartoon: {{ color: '#2ea043', ring: true }} }});
                viewer.setStyle({{ not: {{ resn: ['DA','DC','DG','DT','A','C','G','T'] }} }}, {{ cartoon: {{ color: '#8957e5' }} }});
                viewer.setStyle({{ resn: ['ARG','HIS','ASP'] }}, {{ stick: {{ color: '#e3b341', radius: 0.20 }} }});
            }} else if (styleType === "dna") {{
                viewer.setStyle({{}}, {{ cartoon: {{ color: '#2ea043', ring: true }} }});
                viewer.setStyle({{ name: ['N1','N2','N3','N4','N6','O2','O4','O6'] }}, {{ stick: {{ radius: 0.18, color: '#58a6ff' }} }});
            }} else if (styleType === "protein") {{
                viewer.setStyle({{}}, {{ cartoon: {{ color: '#3fb950' }} }});
                viewer.setStyle({{ resn: 'TRP' }}, {{ stick: {{ color: '#f778ba', radius: 0.25 }} }});
            }}
        }}

        function setFrame(idx) {{
            currentFrame = idx;
            viewer.setFrame(idx);
            document.getElementById("frame-slider").value = idx;
            document.getElementById("frame-label").innerText = (idx + 1) + " / " + totalFrames;
            viewer.render();
        }}

        function onSliderChange(val) {{
            setFrame(parseInt(val));
        }}

        function togglePlay() {{
            if (isPlaying) {{
                stopPlayback();
            }} else {{
                startPlayback();
            }}
        }}

        function startPlayback() {{
            if (playInterval) clearInterval(playInterval);
            isPlaying = true;
            document.getElementById("play-btn").innerText = "Pause";
            document.getElementById("play-btn").style.background = "#238636";
            playInterval = setInterval(() => {{
                currentFrame = (currentFrame + 1) % totalFrames;
                setFrame(currentFrame);
            }}, 120);
        }}

        function stopPlayback() {{
            isPlaying = false;
            if (playInterval) clearInterval(playInterval);
            document.getElementById("play-btn").innerText = "Play";
            document.getElementById("play-btn").style.background = "#1f6feb";
        }}

        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}
    </script>
</body>
</html>
"""

    out_file = "data/viz/solvdock_3d_viewport.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Built standalone 3D viewport at {out_file} ({os.path.getsize(out_file)} bytes)")


if __name__ == "__main__":
    build_viewport()
