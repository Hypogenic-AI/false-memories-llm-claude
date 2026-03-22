"""
Probe dataset for false memory experiments.
Contains real items, plausible-nonexistent items, and implausible-nonexistent items
across multiple categories with varying density.
"""

import json

# Category definitions with real members, plausible nonexistent, and implausible nonexistent
CATEGORIES = {
    "marine_animal_emojis": {
        "description": "Marine/ocean animal emojis",
        "density": "high",  # Many real members
        "real": [
            ("whale", "🐳"),
            ("dolphin", "🐬"),
            ("fish", "🐟"),
            ("tropical fish", "🐠"),
            ("blowfish", "🐡"),
            ("shark", "🦈"),
            ("octopus", "🐙"),
            ("crab", "🦀"),
            ("lobster", "🦞"),
            ("shrimp", "🦐"),
            ("squid", "🦑"),
            ("seal", "🦭"),
            ("jellyfish", "🪼"),
            ("coral", "🪸"),
            ("turtle", "🐢"),
        ],
        "plausible_nonexistent": [
            "seahorse",
            "starfish",
            "sea urchin",
            "manta ray",
            "eel",
            "clam",
            "sea lion",
            "narwhal",
            "manatee",
            "stingray",
        ],
        "implausible_nonexistent": [
            "anglerfish",
            "sea cucumber",
            "barnacle",
            "plankton",
            "sea slug",
        ],
    },
    "land_animal_emojis": {
        "description": "Land animal emojis",
        "density": "high",
        "real": [
            ("dog", "🐕"),
            ("cat", "🐈"),
            ("horse", "🐎"),
            ("cow", "🐄"),
            ("pig", "🐖"),
            ("sheep", "🐑"),
            ("goat", "🐐"),
            ("elephant", "🐘"),
            ("lion", "🦁"),
            ("tiger", "🐅"),
            ("bear", "🐻"),
            ("rabbit", "🐇"),
            ("fox", "🦊"),
            ("deer", "🦌"),
            ("giraffe", "🦒"),
        ],
        "plausible_nonexistent": [
            "platypus",
            "armadillo",
            "capybara",
            "alpaca",
            "wombat",
            "cheetah",
            "gazelle",
            "pangolin",
            "tapir",
            "opossum",
        ],
        "implausible_nonexistent": [
            "aardvark",
            "okapi",
            "binturong",
            "quokka",
            "numbat",
        ],
    },
    "bird_emojis": {
        "description": "Bird emojis",
        "density": "medium",
        "real": [
            ("eagle", "🦅"),
            ("owl", "🦉"),
            ("parrot", "🦜"),
            ("swan", "🦢"),
            ("flamingo", "🦩"),
            ("peacock", "🦚"),
            ("penguin", "🐧"),
            ("rooster", "🐓"),
            ("dove", "🕊️"),
            ("duck", "🦆"),
        ],
        "plausible_nonexistent": [
            "seagull",
            "hummingbird",
            "pelican",
            "toucan",
            "crow",
            "sparrow",
            "hawk",
            "robin",
        ],
        "implausible_nonexistent": [
            "kiwi bird",
            "cassowary",
            "albatross",
            "ibis",
        ],
    },
    "insect_emojis": {
        "description": "Insect/bug emojis",
        "density": "medium",
        "real": [
            ("butterfly", "🦋"),
            ("bug", "🐛"),
            ("ant", "🐜"),
            ("honeybee", "🐝"),
            ("ladybug", "🐞"),
            ("cricket", "🦗"),
            ("cockroach", "🪳"),
            ("beetle", "🪲"),
        ],
        "plausible_nonexistent": [
            "dragonfly",
            "firefly",
            "grasshopper",
            "moth",
            "wasp",
            "centipede",
        ],
        "implausible_nonexistent": [
            "silverfish",
            "earwig",
            "mite",
            "weevil",
        ],
    },
    "fruit_emojis": {
        "description": "Fruit emojis",
        "density": "high",
        "real": [
            ("apple", "🍎"),
            ("banana", "🍌"),
            ("grapes", "🍇"),
            ("strawberry", "🍓"),
            ("watermelon", "🍉"),
            ("cherry", "🍒"),
            ("peach", "🍑"),
            ("pineapple", "🍍"),
            ("mango", "🥭"),
            ("lemon", "🍋"),
            ("orange", "🍊"),
            ("kiwi", "🥝"),
            ("coconut", "🥥"),
            ("blueberries", "🫐"),
        ],
        "plausible_nonexistent": [
            "papaya",
            "passion fruit",
            "guava",
            "pomegranate",
            "fig",
            "dragonfruit",
            "lychee",
            "raspberry",
        ],
        "implausible_nonexistent": [
            "durian",
            "jackfruit",
            "starfruit",
            "rambutan",
        ],
    },
    # Non-emoji categories for cross-domain generalization
    "country_capitals": {
        "description": "Country capital cities",
        "density": "high",
        "real": [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Japan", "Tokyo"),
            ("Australia", "Canberra"),
            ("Brazil", "Brasília"),
            ("Canada", "Ottawa"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("India", "New Delhi"),
            ("Egypt", "Cairo"),
        ],
        "plausible_nonexistent": [
            # Cities that sound like they could be capitals but aren't
            "Sydney is the capital of Australia",
            "Mumbai is the capital of India",
            "Shanghai is the capital of China",
            "Istanbul is the capital of Turkey",
            "Rio de Janeiro is the capital of Brazil",
        ],
        "implausible_nonexistent": [
            "Springfield is the capital of France",
            "Atlantis is the capital of Greece",
            "Gondor is the capital of New Zealand",
        ],
    },
    "chemical_elements": {
        "description": "Chemical elements from the periodic table",
        "density": "high",
        "real": [
            ("Hydrogen", "H"),
            ("Helium", "He"),
            ("Carbon", "C"),
            ("Oxygen", "O"),
            ("Iron", "Fe"),
            ("Gold", "Au"),
            ("Silver", "Ag"),
            ("Uranium", "U"),
            ("Platinum", "Pt"),
            ("Titanium", "Ti"),
        ],
        "plausible_nonexistent": [
            "Viridium",
            "Celestium",
            "Terranium",
            "Aquorium",
            "Luminite",
        ],
        "implausible_nonexistent": [
            "Unobtainium",
            "Kryptonite",
            "Vibranium",
        ],
    },
}

# Prompt templates for different framing conditions
PROMPT_TEMPLATES = {
    "direct_emoji": {
        "system": "You are a knowledgeable assistant. Answer questions accurately and concisely. If you are unsure, say so.",
        "real": "Does a {item} emoji exist in Unicode? Answer with just 'Yes' or 'No', then briefly explain.",
        "nonexistent": "Does a {item} emoji exist in Unicode? Answer with just 'Yes' or 'No', then briefly explain.",
    },
    "indirect_emoji": {
        "system": "You are a helpful assistant.",
        "real": "Can you show me the {item} emoji? If it exists, paste it. If not, say it doesn't exist.",
        "nonexistent": "Can you show me the {item} emoji? If it exists, paste it. If not, say it doesn't exist.",
    },
    "category_listing": {
        "system": "You are a knowledgeable assistant. Be accurate and complete.",
        "prompt": "List all {category_description} that exist in Unicode. Be complete but only include emojis that actually exist.",
    },
    "direct_factual": {
        "system": "You are a knowledgeable assistant. Answer questions accurately and concisely.",
        "real": "Is this statement true: '{item}'? Answer with just 'Yes' or 'No', then briefly explain.",
        "nonexistent": "Is this statement true: '{item}'? Answer with just 'Yes' or 'No', then briefly explain.",
    },
    "confidence_probe": {
        "system": "You are a knowledgeable assistant. For each question, provide your answer and rate your confidence from 0-100.",
        "real": "Does a {item} emoji exist in Unicode? Answer Yes/No, then rate your confidence (0-100).",
        "nonexistent": "Does a {item} emoji exist in Unicode? Answer Yes/No, then rate your confidence (0-100).",
    },
}


def build_probe_set():
    """Build the complete set of probes for all experiments."""
    probes = []

    for cat_name, cat_data in CATEGORIES.items():
        is_emoji_category = "emoji" in cat_name

        # Real items
        for item in cat_data["real"]:
            if is_emoji_category:
                name, emoji_char = item
                probes.append({
                    "category": cat_name,
                    "item": name,
                    "ground_truth": True,
                    "plausibility": "real",
                    "density": cat_data["density"],
                    "emoji_char": emoji_char,
                    "domain": "emoji" if is_emoji_category else "factual",
                })
            else:
                name, detail = item
                probes.append({
                    "category": cat_name,
                    "item": name,
                    "ground_truth": True,
                    "plausibility": "real",
                    "density": cat_data["density"],
                    "detail": detail,
                    "domain": "factual",
                })

        # Plausible nonexistent
        for item in cat_data["plausible_nonexistent"]:
            probes.append({
                "category": cat_name,
                "item": item,
                "ground_truth": False,
                "plausibility": "plausible",
                "density": cat_data["density"],
                "domain": "emoji" if is_emoji_category else "factual",
            })

        # Implausible nonexistent
        for item in cat_data["implausible_nonexistent"]:
            probes.append({
                "category": cat_name,
                "item": item,
                "ground_truth": False,
                "plausibility": "implausible",
                "density": cat_data["density"],
                "domain": "emoji" if is_emoji_category else "factual",
            })

    return probes


def save_probe_set(filepath="results/probe_dataset.json"):
    """Save the probe set to a JSON file."""
    probes = build_probe_set()
    with open(filepath, "w") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)

    # Print summary
    from collections import Counter
    cats = Counter(p["category"] for p in probes)
    plaus = Counter(p["plausibility"] for p in probes)
    print(f"Total probes: {len(probes)}")
    print(f"By plausibility: {dict(plaus)}")
    print(f"By category: {dict(cats)}")
    return probes


if __name__ == "__main__":
    save_probe_set()
