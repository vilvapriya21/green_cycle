import csv
import random
from pathlib import Path
from collections import Counter

# Ensure reproducibility
random.seed(42)

DATA_PATH = Path("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_PATH / "waste_data.csv"


# Category-Specific Modifiers

RECYCLABLE_MODIFIERS = [
    "empty", "crushed", "rinsed", "used", "flattened", "clean"
]

COMPOST_MODIFIERS = [
    "spoiled", "leftover", "rotten", "stale", "wilted", "overripe"
]

HAZARDOUS_MODIFIERS = [
    "leaking", "expired", "broken", "damaged", "partially used", "old"
]


# Base Items

RECYCLABLE_ITEMS = [
    "plastic water bottle", "glass beer bottle", "aluminum soda can",
    "cardboard shipping box", "newspaper stack", "office printer paper",
    "cereal box carton", "milk carton", "plastic food container",
    "steel tin can", "shampoo bottle", "detergent bottle",
    "glass jar", "paper envelope", "magazine bundle",
    "plastic takeaway container", "juice carton",
    "plastic packaging wrap", "metal food can", "plastic yogurt cup"
]

COMPOST_ITEMS = [
    "banana peel", "apple core", "coffee grounds", "tea leaves",
    "vegetable scraps", "lettuce leaves", "orange peel",
    "potato peels", "egg shells", "bread slices",
    "leftover rice", "fruit pulp", "garden leaves",
    "grass clippings", "spinach leaves", "onion skins",
    "tomato scraps", "cucumber peels", "avocado skin",
    "watermelon rind"
]

HAZARDOUS_ITEMS = [
    "used batteries", "car battery", "fluorescent light tube",
    "CFL bulb", "paint thinner can", "pesticide container",
    "medicine tablets", "aerosol spray can",
    "nail polish remover bottle", "acid cleaning bottle",
    "motor oil container", "chemical solvent bottle",
    "insecticide spray", "mercury thermometer",
    "gasoline container", "bleach bottle",
    "printer ink cartridge", "lithium battery pack",
    "medical syringes", "glue thinner can"
]


# Dataset Generation Logic

def generate_variations(items, modifiers, label):
    """
    Generate 2 variations per base item:
    1. Original phrase
    2. Modified phrase using category-specific modifier
    """
    samples = []

    for item in items:
        # Original phrase
        samples.append((item, label))

        # Modifier variation
        modifier = random.choice(modifiers)
        samples.append((f"{modifier} {item}", label))

    return samples


def main():
    dataset = []

    dataset.extend(generate_variations(RECYCLABLE_ITEMS, RECYCLABLE_MODIFIERS, "Recyclable"))
    dataset.extend(generate_variations(COMPOST_ITEMS, COMPOST_MODIFIERS, "Compost"))
    dataset.extend(generate_variations(HAZARDOUS_ITEMS, HAZARDOUS_MODIFIERS, "Hazardous"))

    # Remove accidental duplicates
    dataset = list(set(dataset))

    # Shuffle dataset
    random.shuffle(dataset)

    # Write to CSV
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)

    print(f"\nDataset generated with {len(dataset)} samples.")
    print("Class distribution:")
    counts = Counter(label for _, label in dataset)
    print(counts)


if __name__ == "__main__":
    main()
