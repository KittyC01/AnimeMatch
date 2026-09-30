# AnimeMatch - GPU Accelerated Anime Recommendation System

## Project Overview

AnimeMatch is a GPU-accelerated anime recommendation system.

The program asks the user to provide preferences for action, comedy,
romance, fantasy, and science fiction. It then compares those preferences
against an anime dataset and produces a ranked list of recommendations.

The project demonstrates how GPU acceleration can be used for vector
similarity calculations using PyTorch and CUDA.

## Why GPU?

The recommendation system represents each anime as a numerical feature
vector.

The user's preferences are represented as another vector.

Cosine similarity is calculated between the user vector and all anime
vectors using GPU tensor operations.

The main GPU operation is:

    torch.matmul()

PyTorch moves the data to the CUDA GPU when CUDA is available.

## Requirements

- Python 3
- PyTorch
- NVIDIA CUDA-compatible GPU
- Google Colab with GPU runtime

## Running in Google Colab

Select:

Runtime -> Change runtime type -> T4 GPU

Install dependencies:

    pip install -r requirements.txt

Run the program:

    ./run.sh

## Command Line Arguments

Example:

    python3 anime_recommender.py \
        --action 9 \
        --comedy 4 \
        --romance 3 \
        --fantasy 8 \
        --scifi 2 \
        --number 5

Available arguments:

- --action
- --comedy
- --romance
- --fantasy
- --scifi
- --number
- --input
- --output

Each preference is rated from 0 to 10.

## Example

A user who likes action and fantasy can run:

    python3 anime_recommender.py \
        --action 10 \
        --comedy 2 \
        --romance 2 \
        --fantasy 9 \
        --scifi 1 \
        --number 5

The program produces a ranked list of anime.

## Algorithm

Each anime is represented using five features:

1. Action
2. Comedy
3. Romance
4. Fantasy
5. Sci-Fi

The user's preferences form another five-dimensional vector.

The vectors are normalized and cosine similarity is calculated.

The final score combines:

- 80% similarity to user preferences
- 20% anime rating

The highest scoring anime are returned.

## GPU Implementation

PyTorch tensors are created using:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

The similarity calculation uses:

    torch.matmul(anime_norm, user_norm)

This allows the vector calculation to execute on the NVIDIA GPU.

## Project Structure

    AnimeMatch/
    ├── anime_recommender.py
    ├── anime_data.csv
    ├── README.md
    ├── requirements.txt
    ├── run.sh
    ├── output/
    └── screenshots/

## Lessons Learned

This project demonstrates how a recommendation problem can be represented
as numerical vectors and accelerated using GPU tensor operations.

One important lesson was that using a GPU does not automatically make every
program faster. GPU acceleration becomes more useful as the amount of
parallel data increases.

A future version could use a much larger anime dataset and compare CPU
and GPU execution times.
