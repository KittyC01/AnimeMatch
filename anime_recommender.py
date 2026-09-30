
import argparse
import csv
import math
import os

import torch


def load_anime(filename):
    """Load anime information from CSV."""
    anime = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            for key in [
                "action",
                "comedy",
                "romance",
                "fantasy",
                "scifi",
                "rating",
            ]:
                row[key] = float(row[key])

            anime.append(row)

    return anime


def gpu_recommend(anime, preferences, number):
    """Calculate anime similarity using the CUDA GPU."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nComputation device: {device}")

    anime_features = torch.tensor(
        [
            [
                item["action"],
                item["comedy"],
                item["romance"],
                item["fantasy"],
                item["scifi"],
            ]
            for item in anime
        ],
        dtype=torch.float32,
        device=device,
    )

    user_preferences = torch.tensor(
        preferences,
        dtype=torch.float32,
        device=device,
    )

    # Normalize the vectors.
    anime_norm = torch.nn.functional.normalize(
        anime_features,
        dim=1,
    )

    user_norm = torch.nn.functional.normalize(
        user_preferences,
        dim=0,
    )

    # GPU matrix-vector multiplication.
    similarity = torch.matmul(
        anime_norm,
        user_norm,
    )

    ratings = torch.tensor(
        [item["rating"] for item in anime],
        dtype=torch.float32,
        device=device,
    )

    # Combine similarity and rating.
    scores = (
        similarity * 0.8
        + (ratings / 10.0) * 0.2
    )

    top_k = min(number, len(anime))

    values, indices = torch.topk(
        scores,
        k=top_k,
    )

    results = []

    for value, index in zip(values, indices):
        index = index.item()

        results.append(
            {
                "title": anime[index]["title"],
                "genre": anime[index]["genre"],
                "rating": anime[index]["rating"],
                "similarity": similarity[index].item(),
                "score": value.item(),
            }
        )

    return results


def save_results(results, filename):
    """Save recommendations to CSV."""
    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                "title",
                "genre",
                "rating",
                "similarity",
                "score",
            ]
        )

        for rank, result in enumerate(results, start=1):
            writer.writerow(
                [
                    rank,
                    result["title"],
                    result["genre"],
                    f"{result['rating']:.2f}",
                    f"{result['similarity']:.4f}",
                    f"{result['score']:.4f}",
                ]
            )


def main():

    parser = argparse.ArgumentParser(
        description="GPU-accelerated Anime Recommendation System"
    )

    parser.add_argument("--action", type=float, default=5)
    parser.add_argument("--comedy", type=float, default=5)
    parser.add_argument("--romance", type=float, default=5)
    parser.add_argument("--fantasy", type=float, default=5)
    parser.add_argument("--scifi", type=float, default=5)

    parser.add_argument("--number", type=int, default=5)

    parser.add_argument(
        "--input",
        default="anime_data.csv",
    )

    parser.add_argument(
        "--output",
        default="output/recommendations.csv",
    )

    args = parser.parse_args()

    preferences = [
        args.action,
        args.comedy,
        args.romance,
        args.fantasy,
        args.scifi,
    ]

    anime = load_anime(args.input)

    results = gpu_recommend(
        anime,
        preferences,
        args.number,
    )

    print("\n======================================")
    print("       ANIME RECOMMENDATION SYSTEM")
    print("======================================")

    print("\nYour preferences:")

    print(f"Action:   {args.action}")
    print(f"Comedy:   {args.comedy}")
    print(f"Romance:  {args.romance}")
    print(f"Fantasy:  {args.fantasy}")
    print(f"Sci-Fi:   {args.scifi}")

    print("\nRecommended Anime:\n")

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. {result['title']} "
            f"| Rating: {result['rating']:.1f} "
            f"| Match: {result['score']:.3f}"
        )

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True,
    )

    save_results(
        results,
        args.output,
    )

    print(
        f"\nResults saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
