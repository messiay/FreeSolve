# SolvDock PyMOL Script: Zif268 Zinc Finger - DNA Major Groove Docking (1AAY)
reinitialize
load data/transcription/1aay_transcription_movie.pdb, movie

bg_color white
set ray_shadows, 0
set antialias, 2

hide everything

# DNA duplex representation: cartoon ladder + colored bases
select dna, movie and (resn DA,DC,DG,DT,A,C,G,T)
show cartoon, dna
set cartoon_ring_mode, 3
color forest, dna and (resn DG,DC,G,C)
color orange, dna and (resn DA,DT,A,T)

# Protein representation: cartoon helices
select protein, movie and not dna
show cartoon, protein
color slate, protein

# Highlight key recognition arginines and histidines
select recognition_res, protein and (resn ARG,HIS,ASP) within 4.0 of dna
show sticks, recognition_res
color yellow, recognition_res
set stick_radius, 0.22, recognition_res

# Movie animation
mset 1 -16
mplay

zoom dna, 14.0
orient
