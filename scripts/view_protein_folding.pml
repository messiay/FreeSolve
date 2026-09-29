# SolvDock PyMOL Script: Trp-Cage Protein Folding / Core Compaction (1L2Y)
reinitialize
load data/folding/1l2y_folding_trajectory.pdb, movie

bg_color white
set ray_shadows, 0
set antialias, 2

hide everything
show cartoon, movie
color teal, movie

# Highlight central Trp6 hydrophobic core
select trp_core, movie and resn TRP
show sticks, trp_core
color warmpink, trp_core
set stick_radius, 0.25, trp_core

mset 1 -11
mplay

zoom movie, 8.0
orient
