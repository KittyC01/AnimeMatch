#!/bin/bash

python3 anime_recommender.py \
    --action 9 \
    --comedy 4 \
    --romance 3 \
    --fantasy 8 \
    --scifi 2 \
    --number 5 \
    --input anime_data.csv \
    --output output/recommendations.csv
