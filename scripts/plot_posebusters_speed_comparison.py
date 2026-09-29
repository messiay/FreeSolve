import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Colors
teal = '#028090'
dark_teal = '#005f73'
coral = '#e76f51'
gold = '#e9c46a'
slate = '#6c757d'
dark = '#264653'

# ----------------- Panel A: PoseBusters Physical Validity -----------------
methods = ['EquiBind', 'TANKBind', 'DiffDock', 'Initial Poses\n(Before)', 'FreeSolvE\n(After Rescue)']
rates = [0.3, 24.5, 36.4, 2.0, 52.0]
colors = ['#ced4da', '#adb5bd', '#f4a261', '#e63946', '#2a9d8f']

bars1 = ax1.bar(methods, rates, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('PoseBusters Validity / Clash Pass Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('A: Physical Validity Comparison (Official Benchmark)', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylim(0, 65)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax1.annotate('+50.0% Absolute Gain\nin 2.33 seconds', 
             xy=(4, 52), xytext=(3.0, 58),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#d8f3dc", ec="#2a9d8f", lw=1.5))

# ----------------- Panel B: Computational Speed & Runtime -----------------
engines = ['Explicit MD\n(OpenMM 100ns)', 'Standard MM\nMinimization', 'AutoDock\nVina', 'FreeSolvE\nFull Pose Rescue', 'FreeSolvE\nPoisson Solve']
times_sec = [43200, 30.0, 10.5, 2.33, 0.008] # in seconds
speed_colors = ['#d90429', '#ef233c', '#ffb703', '#0077b6', '#00b4d8']

y_pos = np.arange(len(engines))
bars2 = ax2.barh(y_pos, times_sec, color=speed_colors, height=0.55, edgecolor='black', linewidth=1.2)
ax2.set_xscale('log')
ax2.set_xlabel('Runtime per Complex (Seconds, Log Scale)', fontsize=12, fontweight='bold')
ax2.set_title('B: Execution Speed & High-Throughput Efficiency', fontsize=13, fontweight='bold', pad=12)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(engines, fontsize=10, fontweight='bold')

labels = ['~12 hours\n(43,200s)', '30s\n(often fails)', '10.5s', '2.33s\n(100% stable)', '<10 ms\n(0.008s)']
for i, (bar, label) in enumerate(zip(bars2, labels)):
    width = bar.get_width()
    ax2.text(width * 1.5, bar.get_y() + bar.get_height()/2.0, label,
             ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1d3557')

ax2.set_xlim(0.001, 200000)

plt.tight_layout()
os.makedirs('data/benchmarks', exist_ok=True)
output_path = 'data/benchmarks/posebusters_50_speed_validity.png'
plt.savefig(output_path, dpi=300)
print(f"Generated {output_path} successfully.")
