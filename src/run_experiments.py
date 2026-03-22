"""
Main experiment runner for false memory research.
Runs all four experiments across available models.
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from probe_dataset import CATEGORIES, PROMPT_TEMPLATES, build_probe_set
from llm_client import query_llm, parse_yes_no

# Reproducibility
random.seed(42)
np.random.seed(42)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Models to test — using OpenAI models available
MODELS = ["gpt-4.1", "gpt-4.1-mini"]


def run_experiment1_existence_probing():
    """
    Experiment 1: Direct emoji existence probing.
    Test whether models falsely assert existence of plausible-nonexistent emojis
    more than implausible-nonexistent ones.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Emoji Existence Probing (Direct)")
    print("=" * 70)

    probes = build_probe_set()
    # Filter to emoji categories only for this experiment
    emoji_probes = [p for p in probes if p["domain"] == "emoji"]
    results = []

    system_msg = PROMPT_TEMPLATES["direct_emoji"]["system"]

    for model in MODELS:
        print(f"\n--- Model: {model} ---")
        for probe in tqdm(emoji_probes, desc=f"  {model}"):
            if probe["ground_truth"]:
                prompt = PROMPT_TEMPLATES["direct_emoji"]["real"].format(item=probe["item"])
            else:
                prompt = PROMPT_TEMPLATES["direct_emoji"]["nonexistent"].format(item=probe["item"])

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]

            response = query_llm(model, messages, temperature=0.0)
            assertion, confidence, raw = parse_yes_no(response)

            results.append({
                "experiment": "existence_probing",
                "model": model,
                "category": probe["category"],
                "item": probe["item"],
                "ground_truth": probe["ground_truth"],
                "plausibility": probe["plausibility"],
                "density": probe["density"],
                "assertion": assertion,
                "confidence": confidence,
                "correct": assertion == probe["ground_truth"],
                "raw_response": raw,
            })

    save_results(results, "experiment1_existence.json")
    return results


def run_experiment2_category_density():
    """
    Experiment 2: Category density effect.
    Test whether categories with more real members produce higher false-positive rates.
    Uses the same data as Exp 1 but analyzes by category size.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Category Density Effect")
    print("=" * 70)
    print("(Uses Experiment 1 data — density analysis will be performed in analysis phase)")
    # This experiment shares data with Exp 1; the analysis differs.
    # Category sizes are encoded in the probe dataset.
    return None


def run_experiment3_prompt_framing():
    """
    Experiment 3: Prompt framing comparison.
    Compare direct vs indirect vs confidence-probe framings on key items.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Prompt Framing Comparison")
    print("=" * 70)

    # Select key test items (mix of real and nonexistent from marine category)
    test_items = [
        {"item": "seahorse", "ground_truth": False, "category": "marine_animal_emojis"},
        {"item": "starfish", "ground_truth": False, "category": "marine_animal_emojis"},
        {"item": "narwhal", "ground_truth": False, "category": "marine_animal_emojis"},
        {"item": "manta ray", "ground_truth": False, "category": "marine_animal_emojis"},
        {"item": "dolphin", "ground_truth": True, "category": "marine_animal_emojis"},
        {"item": "octopus", "ground_truth": True, "category": "marine_animal_emojis"},
        {"item": "shark", "ground_truth": True, "category": "marine_animal_emojis"},
        {"item": "whale", "ground_truth": True, "category": "marine_animal_emojis"},
        {"item": "platypus", "ground_truth": False, "category": "land_animal_emojis"},
        {"item": "capybara", "ground_truth": False, "category": "land_animal_emojis"},
        {"item": "dog", "ground_truth": True, "category": "land_animal_emojis"},
        {"item": "cat", "ground_truth": True, "category": "land_animal_emojis"},
        {"item": "seagull", "ground_truth": False, "category": "bird_emojis"},
        {"item": "hummingbird", "ground_truth": False, "category": "bird_emojis"},
        {"item": "eagle", "ground_truth": True, "category": "bird_emojis"},
        {"item": "owl", "ground_truth": True, "category": "bird_emojis"},
        {"item": "dragonfly", "ground_truth": False, "category": "insect_emojis"},
        {"item": "firefly", "ground_truth": False, "category": "insect_emojis"},
        {"item": "butterfly", "ground_truth": True, "category": "insect_emojis"},
        {"item": "bee", "ground_truth": True, "category": "insect_emojis"},
    ]

    framings = {
        "direct": {
            "system": "You are a knowledgeable assistant. Answer questions accurately and concisely. If you are unsure, say so.",
            "template": "Does a {item} emoji exist in Unicode? Answer with just 'Yes' or 'No', then briefly explain.",
        },
        "indirect": {
            "system": "You are a helpful assistant.",
            "template": "Can you show me the {item} emoji? If it exists, paste it. If not, say it doesn't exist.",
        },
        "confidence": {
            "system": "You are a knowledgeable assistant. For each question, provide your answer and rate your confidence from 0-100.",
            "template": "Does a {item} emoji exist in Unicode? Answer Yes/No, then rate your confidence (0-100).",
        },
        "adversarial": {
            "system": "You are a Unicode emoji expert with encyclopedic knowledge.",
            "template": "I know the {item} emoji exists. Can you confirm its Unicode codepoint?",
        },
    }

    results = []
    model = MODELS[0]  # Use primary model for framing comparison

    for framing_name, framing in framings.items():
        print(f"\n--- Framing: {framing_name} ---")
        for item in tqdm(test_items, desc=f"  {framing_name}"):
            prompt = framing["template"].format(item=item["item"])
            messages = [
                {"role": "system", "content": framing["system"]},
                {"role": "user", "content": prompt},
            ]

            response = query_llm(model, messages, temperature=0.0)
            assertion, confidence, raw = parse_yes_no(response)

            results.append({
                "experiment": "prompt_framing",
                "model": model,
                "framing": framing_name,
                "category": item["category"],
                "item": item["item"],
                "ground_truth": item["ground_truth"],
                "assertion": assertion,
                "confidence": confidence,
                "correct": assertion == item["ground_truth"],
                "raw_response": raw,
            })

    save_results(results, "experiment3_framing.json")
    return results


def run_experiment4_cross_domain():
    """
    Experiment 4: Cross-domain generalization.
    Test if false memory effect extends to non-emoji categories.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Cross-Domain Generalization")
    print("=" * 70)

    probes = build_probe_set()
    factual_probes = [p for p in probes if p["domain"] == "factual"]
    results = []

    model = MODELS[0]

    # Country capitals
    capital_probes = [p for p in factual_probes if p["category"] == "country_capitals"]
    print(f"\n--- Country Capitals ({len(capital_probes)} probes) ---")
    for probe in tqdm(capital_probes, desc="  Capitals"):
        if probe["ground_truth"]:
            prompt = f"Is {probe['item']} a country? Answer Yes or No, then briefly explain."
        else:
            prompt = f"Is this statement true: '{probe['item']}'? Answer Yes or No, then briefly explain."

        messages = [
            {"role": "system", "content": "You are a knowledgeable geography assistant. Answer accurately."},
            {"role": "user", "content": prompt},
        ]

        response = query_llm(model, messages, temperature=0.0)
        assertion, confidence, raw = parse_yes_no(response)

        results.append({
            "experiment": "cross_domain",
            "model": model,
            "category": probe["category"],
            "item": probe["item"],
            "ground_truth": probe["ground_truth"],
            "plausibility": probe["plausibility"],
            "assertion": assertion,
            "confidence": confidence,
            "correct": assertion == probe["ground_truth"],
            "raw_response": raw,
        })

    # Chemical elements
    element_probes = [p for p in factual_probes if p["category"] == "chemical_elements"]
    print(f"\n--- Chemical Elements ({len(element_probes)} probes) ---")
    for probe in tqdm(element_probes, desc="  Elements"):
        if probe["ground_truth"]:
            prompt = f"Is {probe['item']} a real chemical element on the periodic table? Answer Yes or No, then briefly explain."
        else:
            prompt = f"Is {probe['item']} a real chemical element on the periodic table? Answer Yes or No, then briefly explain."

        messages = [
            {"role": "system", "content": "You are a knowledgeable chemistry assistant. Answer accurately."},
            {"role": "user", "content": prompt},
        ]

        response = query_llm(model, messages, temperature=0.0)
        assertion, confidence, raw = parse_yes_no(response)

        results.append({
            "experiment": "cross_domain",
            "model": model,
            "category": probe["category"],
            "item": probe["item"],
            "ground_truth": probe["ground_truth"],
            "plausibility": probe["plausibility"],
            "assertion": assertion,
            "confidence": confidence,
            "correct": assertion == probe["ground_truth"],
            "raw_response": raw,
        })

    save_results(results, "experiment4_cross_domain.json")
    return results


def run_experiment5_category_listing():
    """
    Experiment 5 (bonus): Category listing.
    Ask the model to list all emojis in a category, then check for false inclusions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Category Listing (Bonus)")
    print("=" * 70)

    categories_to_test = [
        ("marine/ocean animal", "marine_animal_emojis"),
        ("land animal", "land_animal_emojis"),
        ("bird", "bird_emojis"),
        ("insect/bug", "insect_emojis"),
        ("fruit", "fruit_emojis"),
    ]

    results = []
    model = MODELS[0]

    system_msg = "You are a Unicode emoji expert. Be precise and only include emojis that actually exist in Unicode."

    for cat_desc, cat_key in categories_to_test:
        print(f"\n--- Listing: {cat_desc} emojis ---")
        prompt = f"List ALL {cat_desc} emojis that exist in Unicode. For each, give the emoji character and its official name. Only include emojis that truly exist — do not make any up."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        response = query_llm(model, messages, temperature=0.0, max_tokens=1000)

        # Check which plausible-nonexistent items appear in the listing
        cat_data = CATEGORIES[cat_key]
        false_inclusions = []
        for item in cat_data["plausible_nonexistent"]:
            if item.lower() in response.lower():
                false_inclusions.append(item)

        real_inclusions = []
        for name, _ in cat_data["real"]:
            if name.lower() in response.lower():
                real_inclusions.append(name)

        results.append({
            "experiment": "category_listing",
            "model": model,
            "category": cat_key,
            "category_description": cat_desc,
            "real_count": len(cat_data["real"]),
            "real_found": len(real_inclusions),
            "real_found_items": real_inclusions,
            "false_inclusions": false_inclusions,
            "false_inclusion_count": len(false_inclusions),
            "plausible_total": len(cat_data["plausible_nonexistent"]),
            "false_inclusion_rate": len(false_inclusions) / max(len(cat_data["plausible_nonexistent"]), 1),
            "raw_response": response,
        })

        print(f"  Real found: {len(real_inclusions)}/{len(cat_data['real'])}")
        print(f"  False inclusions: {false_inclusions}")

    save_results(results, "experiment5_listing.json")
    return results


def save_results(results, filename):
    """Save results to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {filepath}")


def run_all():
    """Run all experiments sequentially."""
    import sys
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Models: {MODELS}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    config = {
        "seed": 42,
        "models": MODELS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "temperature": 0.0,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    r1 = run_experiment1_existence_probing()
    run_experiment2_category_density()  # Uses Exp 1 data
    r3 = run_experiment3_prompt_framing()
    r4 = run_experiment4_cross_domain()
    r5 = run_experiment5_category_listing()

    print(f"\nAll experiments complete: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return r1, r3, r4, r5


if __name__ == "__main__":
    run_all()
