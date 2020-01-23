import gzip
import json
import random


def generate_mini_r2r_split(size, split="val_seen"):
    r""" Generates a new R2R sub-dataset from an existing split. Chooses `size`
    number of scenes at random and every episode in each of those scenes.
    """
    with gzip.open(
        f"data/datasets/vln/mp3d/r2r/v1/preprocessed/{split}/{split}.json.gz",
        "rt",
    ) as f:
        d = json.load(f)

    assert size > 0 and size < len(d["episodes"]), "Size out of limits"

    new_split = {"instruction_vocab": d["instruction_vocab"], "episodes": []}
    scenes = random.sample(
        list(set([e["scene_id"] for e in d["episodes"]])), k=size
    )

    for e in d["episodes"]:
        if e["scene_id"] in scenes:
            new_split["episodes"].append(e)

    print(f"Number of scenes in new split: {size}")
    print(f"Number of episodes in new split: {len(new_split['episodes'])}")
    with open(f"{split}_{size}.json", "w") as f:
        json.dump(new_split, f)
    print(f"Saved new split to: ./{split}_{size}.json")


if __name__ == "__main__":
    generate_mini_r2r_split(20)
