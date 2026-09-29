# SolvDock PyMOL Script: B-DNA Double-Helix Zipping Simulation (1BNA)
reinitialize
load data/dna/dna_zipping_trajectory.pdb, movie

bg_color white
set ray_shadows, 0
set antialias, 2

hide everything
show cartoon, movie
set cartoon_ring_mode, 3
color forest, movie and (resn DG,DC,G,C)
color orange, movie and (resn DA,DT,A,T)

show sticks, movie and name n1,n2,n3,n4,n6,o2,o4,o6
set stick_radius, 0.20

mset 1 -6
mplay

zoom movie, 10.0
orient
