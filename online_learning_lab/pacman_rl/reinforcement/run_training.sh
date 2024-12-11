#!/bin/bash
EPISODES=5000 # number of training episodes
MAP=capsuleClassic #small map to ensure fast execution, you find different maps to test in the layouts folder
                # you should execute the code on the originalClassic map if you can
# Esegui il comando Python con i parametri
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x $EPISODES -n $EPISODES -l $MAP -g DirectionalGhost
