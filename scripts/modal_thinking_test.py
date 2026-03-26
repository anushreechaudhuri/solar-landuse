"""
Quick Modal diagnostic: Gemini 2.5 Flash with vs without thinking tokens.
Uses the NEW google-genai SDK (not deprecated google-generativeai).
Runs 10 images each with thinking ON and thinking OFF.

Usage:
    MODAL_PROFILE=solar-landuse modal run scripts/modal_thinking_test.py
"""
import modal

app = modal.App("solar-vlm-thinking-test-v2")

vol = modal.Volume.from_name("solar-landuse-data", create_if_missing=False)
VOL_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("google-genai", "Pillow")
)

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


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gemini-api-key")],
    volumes={VOL_PATH: vol},
    timeout=180,
)
def classify_one(site_id: str, year: int, thinking_budget: int | None) -> dict:
    """Run one classification with specified thinking budget using google-genai SDK."""
    import json
    import os
    import time
    from pathlib import Path

    img_path = Path(f"{VOL_PATH}/s2_images/{site_id}_{year}.png")
    if not img_path.exists():
        return {"error": "no_image", "site_id": site_id, "year": year}

    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"])
    img = Image.open(img_path)

    # Build generation config
    config_kwargs = {
        "response_mime_type": "application/json",
        "temperature": 0.1,
    }
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    config_used = f"thinking_budget={thinking_budget}" if thinking_budget is not None else "default_thinking"

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[PROMPT, img],
            config=types.GenerateContentConfig(**config_kwargs),
        )
        elapsed = time.time() - t0

        # Extract token counts from usage_metadata
        meta = response.usage_metadata
        input_tokens = meta.prompt_token_count or 0
        output_tokens = meta.candidates_token_count or 0
        thinking_tokens = getattr(meta, "thoughts_token_count", 0) or 0
        total_tokens = meta.total_token_count or 0

        result = json.loads(response.text)

        return {
            "site_id": site_id,
            "year": year,
            "config": config_used,
            "elapsed_s": round(elapsed, 2),
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
        }
    except Exception as e:
        return {
            "site_id": site_id,
            "year": year,
            "config": config_used,
            "elapsed_s": round(time.time() - t0, 2),
            "error": str(e),
        }


@app.local_entrypoint()
def main():
    import json

    # Test sites: mix of pre/post, BD + India, different sizes
    test_cases = [
        ("BA_0048", 2018),  # Manikganj pre
        ("BA_0048", 2023),  # Manikganj post
        ("BA_0088", 2020),  # Feni pre
        ("BA_0088", 2025),  # Feni post
        ("BA_0052", 2019),  # Mongla pre
        ("BA_0052", 2023),  # Mongla post
        ("IN_0100", 2022),  # India site
        ("IN_0100", 2025),
        ("IN_0200", 2020),
        ("IN_0200", 2024),
    ]

    print(f"Testing {len(test_cases)} images x 2 modes = {len(test_cases)*2} API calls")
    print(f"SDK: google-genai (new, supports thinking_config)\n")

    # Run WITH thinking (default — no budget set)
    print("=== WITH THINKING (default) ===")
    results_thinking = []
    for result in classify_one.map(
        [t[0] for t in test_cases],
        [t[1] for t in test_cases],
        [None] * len(test_cases),
    ):
        results_thinking.append(result)
        if "error" in result:
            print(f"  {result['site_id']}_{result['year']}: ERROR {result['error']}")
        else:
            print(f"  {result['site_id']}_{result['year']}: {result['elapsed_s']}s | "
                  f"in={result['input_tokens']} out={result['output_tokens']} "
                  f"think={result['thinking_tokens']} total={result['total_tokens']} | "
                  f"solar={result['solar_panels']:.1f}% crops={result['crops']:.1f}%")

    # Run WITHOUT thinking (budget=0)
    print("\n=== WITHOUT THINKING (budget=0) ===")
    results_no_thinking = []
    for result in classify_one.map(
        [t[0] for t in test_cases],
        [t[1] for t in test_cases],
        [0] * len(test_cases),
    ):
        results_no_thinking.append(result)
        if "error" in result:
            print(f"  {result['site_id']}_{result['year']}: ERROR {result['error']}")
        else:
            print(f"  {result['site_id']}_{result['year']}: {result['elapsed_s']}s | "
                  f"in={result['input_tokens']} out={result['output_tokens']} "
                  f"think={result['thinking_tokens']} total={result['total_tokens']} | "
                  f"solar={result['solar_panels']:.1f}% crops={result['crops']:.1f}%")

    # Summary
    valid_t = [r for r in results_thinking if "error" not in r]
    valid_nt = [r for r in results_no_thinking if "error" not in r]

    if not valid_t or not valid_nt:
        print("\nInsufficient data for comparison")
        return

    def avg(lst, key): return sum(r[key] for r in lst) / len(lst)

    print("\n" + "=" * 95)
    print("COMPARISON (google-genai SDK)")
    print("=" * 95)
    print(f"\n{'Metric':<30} {'Thinking':>15} {'No Thinking':>15} {'Diff':>12}")
    print("-" * 75)
    print(f"{'Images tested':<30} {len(valid_t):>15} {len(valid_nt):>15}")
    print(f"{'Avg latency (s)':<30} {avg(valid_t, 'elapsed_s'):>15.1f} {avg(valid_nt, 'elapsed_s'):>15.1f} {avg(valid_t, 'elapsed_s') - avg(valid_nt, 'elapsed_s'):>+12.1f}")
    print(f"{'Avg input tokens':<30} {avg(valid_t, 'input_tokens'):>15.0f} {avg(valid_nt, 'input_tokens'):>15.0f}")
    print(f"{'Avg output tokens':<30} {avg(valid_t, 'output_tokens'):>15.0f} {avg(valid_nt, 'output_tokens'):>15.0f}")
    print(f"{'Avg thinking tokens':<30} {avg(valid_t, 'thinking_tokens'):>15.0f} {avg(valid_nt, 'thinking_tokens'):>15.0f}")
    print(f"{'Avg total tokens':<30} {avg(valid_t, 'total_tokens'):>15.0f} {avg(valid_nt, 'total_tokens'):>15.0f}")

    # Per-image quality comparison
    print(f"\n{'--- Per-image comparison ---'}")
    print(f"{'Site':<20} {'Think Solar%':>12} {'NoThink Solar%':>15} {'Diff':>8} {'Think Crops%':>13} {'NoThink Crops%':>15}")
    print("-" * 90)
    solar_diffs = []
    crop_diffs = []
    solar_visible_agree = 0
    total_pairs = 0
    for rt, rnt in zip(results_thinking, results_no_thinking):
        if "error" in rt or "error" in rnt:
            continue
        if rt["site_id"] != rnt["site_id"] or rt["year"] != rnt["year"]:
            continue
        total_pairs += 1
        sd = rt["solar_panels"] - rnt["solar_panels"]
        cd = rt["crops"] - rnt["crops"]
        solar_diffs.append(sd)
        crop_diffs.append(cd)
        if rt["solar_visible"] == rnt["solar_visible"]:
            solar_visible_agree += 1
        flag = " ***" if abs(sd) > 5 else ""
        print(f"{rt['site_id']}_{rt['year']:<6} {rt['solar_panels']:>12.1f} {rnt['solar_panels']:>15.1f} {sd:>+8.1f} {rt['crops']:>13.1f} {rnt['crops']:>15.1f}{flag}")

    if solar_diffs:
        import statistics
        print(f"\nSolar % diff (think - nothink): mean={statistics.mean(solar_diffs):+.2f}, stdev={statistics.stdev(solar_diffs):.2f}, max|diff|={max(abs(d) for d in solar_diffs):.1f}")
        print(f"Crops % diff (think - nothink): mean={statistics.mean(crop_diffs):+.2f}, stdev={statistics.stdev(crop_diffs):.2f}")
        print(f"Solar visible agreement: {solar_visible_agree}/{total_pairs} ({100*solar_visible_agree/total_pairs:.0f}%)")

    # Cost projection
    PRICING = {
        "Standard": {"input": 0.30, "output": 2.50},
        "Batch":    {"input": 0.15, "output": 1.25},
    }
    N = 36166

    print(f"\n{'--- Cost projection for {N:,} images ---':}")
    for label, valid in [("WITH thinking", valid_t), ("WITHOUT thinking", valid_nt)]:
        avg_in = avg(valid, "input_tokens")
        avg_out = avg(valid, "output_tokens")
        avg_think = avg(valid, "thinking_tokens")
        billable_output = avg_out + avg_think  # thinking billed at output rate

        print(f"\n  {label} (avg {avg_in:.0f} in + {avg_out:.0f} out + {avg_think:.0f} think = {avg_in + billable_output:.0f} billable):")
        for tier, prices in PRICING.items():
            cost_in = (avg_in * N / 1e6) * prices["input"]
            cost_out = (billable_output * N / 1e6) * prices["output"]
            total_cost = cost_in + cost_out
            print(f"    {tier:>10}: ${cost_in:.2f} input + ${cost_out:.2f} output = ${total_cost:.2f}")

    # Save results
    output = {
        "sdk": "google-genai",
        "thinking": results_thinking,
        "no_thinking": results_no_thinking,
    }
    with open("data/modal_thinking_test_v2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: data/modal_thinking_test_v2.json")
