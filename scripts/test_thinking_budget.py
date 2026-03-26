"""
Quick diagnostic: Gemini 2.5 Flash with vs without thinking tokens.
Tests on local S2 images to compare quality, cost, and latency.

Usage:
    python scripts/test_thinking_budget.py
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from PIL import Image

api_key = os.environ.get("GOOGLE_AI_API_KEY", "")
if not api_key:
    print("ERROR: Set GOOGLE_AI_API_KEY in .env")
    sys.exit(1)
genai.configure(api_key=api_key)

PROMPT = """Analyze this satellite image and estimate the percentage of land cover for each class.
The image shows a ~1-5km area around a potential solar energy site in South Asia.

Return a JSON object with these exact keys and percentage values (must sum to 100):
{
  "water": <float>,
  "trees": <float>,
  "grass": <float>,
  "flooded_vegetation": <float>,
  "crops": <float>,
  "shrub_and_scrub": <float>,
  "built": <float>,
  "bare": <float>,
  "snow_and_ice": <float>,
  "solar_panels": <float>
}

Also include:
  "solar_visible": true/false (are solar panels clearly visible?)
  "description": "<brief 1-sentence description of the landscape>"

Return ONLY the JSON object, no other text."""

# Test images: mix of pre/post, different sites and sizes
TEST_IMAGES = [
    ("BA_0001 (pre-2020)", "data/s2_images/BA_0001_2020.png"),
    ("BA_0001 (post-2024)", "data/s2_images/BA_0001_2024.png"),
    ("Teesta 200MW (post)", "data/case_study_s2_images/teesta_2024_s2.png"),
    ("Feni 75MW (post)", "data/case_study_s2_images/feni_2025_s2.png"),
    ("Manikganj 35MW (pre)", "data/case_study_s2_images/manikganj_2018_s2.png"),
    ("Manikganj 35MW (post)", "data/case_study_s2_images/manikganj_2023_s2.png"),
    ("Moulvibazar 10MW (pre)", "data/case_study_s2_images/moulvibazar_2022_s2.png"),
    ("Teesta 200MW (pre)", "data/case_study_s2_images/teesta_2020_s2.png"),
]

# Gemini pricing (per 1M tokens, standard tier)
# https://ai.google.dev/pricing
PRICE_INPUT_1M = 0.15    # $0.15 per 1M input tokens
PRICE_OUTPUT_1M = 0.60   # $0.60 per 1M output tokens
PRICE_THINKING_1M = 3.50 # $3.50 per 1M thinking tokens


def run_single(img_path, label, thinking_budget, model_name="gemini-2.5-flash"):
    """Run one classification, return result + metadata."""
    img = Image.open(img_path)

    model = genai.GenerativeModel(model_name)

    gen_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.1,
    }

    # Try setting thinking_config if SDK supports it
    if thinking_budget is not None:
        try:
            gen_config = genai.GenerationConfig(
                thinking_config={"thinking_budget": thinking_budget},
                **gen_kwargs,
            )
        except TypeError:
            # SDK too old for thinking_config
            print(f"  WARNING: SDK doesn't support thinking_config, using default")
            gen_config = genai.GenerationConfig(**gen_kwargs)
    else:
        gen_config = genai.GenerationConfig(**gen_kwargs)

    t0 = time.time()
    try:
        response = model.generate_content(
            [PROMPT, img],
            generation_config=gen_config,
        )
        elapsed = time.time() - t0

        # Extract token counts from usage_metadata
        meta = response.usage_metadata
        input_tokens = getattr(meta, "prompt_token_count", 0)
        output_tokens = getattr(meta, "candidates_token_count", 0)
        # thinking tokens show up as part of candidates but are tracked separately
        thinking_tokens = getattr(meta, "thoughts_token_count", 0) if hasattr(meta, "thoughts_token_count") else 0
        total_tokens = getattr(meta, "total_token_count", 0)

        result = json.loads(response.text)

        return {
            "label": label,
            "thinking_budget": thinking_budget,
            "elapsed_s": round(elapsed, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "solar_panels": result.get("solar_panels", 0),
            "solar_visible": result.get("solar_visible", False),
            "crops": result.get("crops", 0),
            "trees": result.get("trees", 0),
            "built": result.get("built", 0),
            "bare": result.get("bare", 0),
            "water": result.get("water", 0),
            "description": result.get("description", ""),
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "label": label,
            "thinking_budget": thinking_budget,
            "elapsed_s": round(elapsed, 1),
            "error": str(e),
        }


def main():
    # Filter to images that exist
    images = [(label, p) for label, p in TEST_IMAGES if Path(p).exists()]
    print(f"Testing {len(images)} images with thinking ON vs OFF\n")

    results_thinking = []
    results_no_thinking = []

    for label, img_path in images:
        print(f"--- {label} ({img_path}) ---")

        # With thinking (default)
        print(f"  [thinking=default] ...", end=" ", flush=True)
        r1 = run_single(img_path, label, thinking_budget=None)
        if r1.get("error"):
            print(f"ERROR: {r1['error']}")
        else:
            print(f"{r1['elapsed_s']}s, solar={r1['solar_panels']:.1f}%, "
                  f"in={r1['input_tokens']}, out={r1['output_tokens']}, "
                  f"think={r1['thinking_tokens']}")
        results_thinking.append(r1)

        # Without thinking
        print(f"  [thinking=0]       ...", end=" ", flush=True)
        r2 = run_single(img_path, label, thinking_budget=0)
        if r2.get("error"):
            print(f"ERROR: {r2['error']}")
        else:
            print(f"{r2['elapsed_s']}s, solar={r2['solar_panels']:.1f}%, "
                  f"in={r2['input_tokens']}, out={r2['output_tokens']}, "
                  f"think={r2['thinking_tokens']}")
        results_no_thinking.append(r2)

        # Brief comparison
        if not r1.get("error") and not r2.get("error"):
            diff = abs(r1["solar_panels"] - r2["solar_panels"])
            if diff > 2:
                print(f"  ** Solar difference: {diff:.1f} pp **")
        print()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\n{'Label':<30} {'Mode':<12} {'Time':>6} {'In tok':>8} {'Out tok':>8} "
          f"{'Think tok':>10} {'Solar%':>7} {'Crops%':>7}")
    print("-" * 90)

    for r1, r2 in zip(results_thinking, results_no_thinking):
        for r, mode in [(r1, "thinking"), (r2, "no-think")]:
            if r.get("error"):
                print(f"{r['label']:<30} {mode:<12} {'ERR':>6}")
                continue
            print(f"{r['label']:<30} {mode:<12} {r['elapsed_s']:>5.1f}s "
                  f"{r.get('input_tokens',0):>8} {r.get('output_tokens',0):>8} "
                  f"{r.get('thinking_tokens',0):>10} {r.get('solar_panels',0):>6.1f}% "
                  f"{r.get('crops',0):>6.1f}%")

    # Cost projection
    print("\n" + "=" * 90)
    print("COST PROJECTION FOR 36,166 IMAGES")
    print("=" * 90)

    for label, results in [("With thinking", results_thinking), ("No thinking", results_no_thinking)]:
        valid = [r for r in results if not r.get("error")]
        if not valid:
            continue
        avg_input = sum(r["input_tokens"] for r in valid) / len(valid)
        avg_output = sum(r["output_tokens"] for r in valid) / len(valid)
        avg_thinking = sum(r.get("thinking_tokens", 0) for r in valid) / len(valid)
        avg_time = sum(r["elapsed_s"] for r in valid) / len(valid)

        cost_input = (avg_input * 36166 / 1e6) * PRICE_INPUT_1M
        cost_output = (avg_output * 36166 / 1e6) * PRICE_OUTPUT_1M
        cost_thinking = (avg_thinking * 36166 / 1e6) * PRICE_THINKING_1M
        total_cost = cost_input + cost_output + cost_thinking

        print(f"\n  {label}:")
        print(f"    Avg tokens/image: input={avg_input:.0f}, output={avg_output:.0f}, thinking={avg_thinking:.0f}")
        print(f"    Avg latency: {avg_time:.1f}s")
        print(f"    Projected cost: ${cost_input:.2f} (input) + ${cost_output:.2f} (output) "
              f"+ ${cost_thinking:.2f} (thinking) = ${total_cost:.2f}")

    # Quality comparison
    print("\n" + "=" * 90)
    print("QUALITY COMPARISON")
    print("=" * 90)

    solar_diffs = []
    crop_diffs = []
    agree_solar_visible = 0
    total_valid = 0

    for r1, r2 in zip(results_thinking, results_no_thinking):
        if r1.get("error") or r2.get("error"):
            continue
        total_valid += 1
        solar_diffs.append(r1["solar_panels"] - r2["solar_panels"])
        crop_diffs.append(r1["crops"] - r2["crops"])
        if r1["solar_visible"] == r2["solar_visible"]:
            agree_solar_visible += 1

    if total_valid:
        import statistics
        print(f"\n  Solar % difference (thinking - no_think): "
              f"mean={statistics.mean(solar_diffs):+.2f}, "
              f"stdev={statistics.stdev(solar_diffs):.2f}" if len(solar_diffs) > 1 else "")
        print(f"  Crops % difference (thinking - no_think): "
              f"mean={statistics.mean(crop_diffs):+.2f}, "
              f"stdev={statistics.stdev(crop_diffs):.2f}" if len(crop_diffs) > 1 else "")
        print(f"  Solar visible agreement: {agree_solar_visible}/{total_valid} "
              f"({100*agree_solar_visible/total_valid:.0f}%)")
        print(f"  Max |solar diff|: {max(abs(d) for d in solar_diffs):.1f} pp")

    # Save raw results
    output = {
        "test_date": "2026-03-16",
        "n_images": len(images),
        "thinking_results": results_thinking,
        "no_thinking_results": results_no_thinking,
    }
    out_path = "data/vlm_thinking_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved raw results: {out_path}")


if __name__ == "__main__":
    main()
