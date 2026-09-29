# SolvDock PyMOL Script: Human HSP90 Induced-Fit Docking (2v00)
reinitialize
load data/docking/2v00_induced_fit_movie.pdb, movie

bg_color white
set ray_shadows, 0
set antialias, 2

hide everything
show cartoon, movie and polymer
color marine, movie and polymer

# Show flexible pocket side chains
select sidechains, movie and polymer and not (name n,ca,c,o)
show sticks, sidechains within 4.5 of (movie and not polymer)
color lightblue, sidechains

# Show drug molecule in vibrant magenta
select drug, movie and not polymer
show sticks, drug
color magenta, drug
set stick_radius, 0.28, drug

# Movie animation
mset 1 -16
mplay

zoom drug, 8.0
orient
