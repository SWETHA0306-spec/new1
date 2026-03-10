"""
main.py — Drug AI Console Application
========================================
Full pipeline entry point. Supports two input modes:

  Mode 1 (IMAGE):  python main.py --image path/to/label.jpg
  Mode 2 (TEXT):   python main.py --drug "Drug Name"
  Interactive:     python main.py   (prompts for input)

Pipeline:
  Input (image/text)
    → OCR (image mode only)
    → Drug Resolver   → canonical identity + authenticity score
    → Interaction Engine → DDI risk from RxNorm + OpenFDA
    → Risk Report     → console display + JSON export

Install all deps:
  pip install pillow requests opencv-python

No API keys needed except OpenRouter (for OCR image mode only).
Set your key in core/ocr_fixed.py  →  API_KEY = "your_key_here"
"""

import sys
import os
import argparse

# ── Add core/ to path ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from resolver     import resolve_drug, print_identity
from interactions import check_interactions, print_interactions
from report       import build_report, print_report, export_report


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _banner():
    print("""
╔═══════════════════════════════════════════════════════╗
║           DRUG AUTHENTICITY & INTERACTION AI          ║
║         Counterfeit Detection + DDI Analysis          ║
╚═══════════════════════════════════════════════════════╝""")


def _ask_other_drugs() -> list[str]:
    """Prompt user for other medications."""
    print("\n  Enter other medications you are currently taking.")
    print("  Separate multiple drugs with commas.")
    print("  Press ENTER to skip.\n")
    raw = input("  Other medications: ").strip()
    if not raw:
        return []
    return [d.strip() for d in raw.split(",") if d.strip()]


def _export_prompt(report: dict):
    """Ask user if they want to export the JSON report."""
    ans = input("\n  Export full report to JSON? [y/N]: ").strip().lower()
    if ans == "y":
        path = export_report(report, output_dir="reports")
        print(f"\n  📄 Report saved → {path}")


# ══════════════════════════════════════════════════════════════
#  IMAGE MODE
# ══════════════════════════════════════════════════════════════

def run_image_mode(image_path: str):
    """Full pipeline starting from a drug label image."""
    from ocr_fixed import run_ocr, print_result

    # Step 1: OCR
    ocr_result = run_ocr(image_path)
    print_result(ocr_result, image_path)

    if not ocr_result.get("success"):
        print("\n  ✗ OCR failed. Cannot continue.\n")
        sys.exit(1)

    ocr_result["_source_mode"] = "image"   # real label — enable full suspicion checks

    # Step 2: Resolve
    identity = resolve_drug(ocr_result)
    print_identity(identity)

    # Step 3: Get other drugs from user
    other_drugs = _ask_other_drugs()

    # Step 4: Interaction check
    interaction_result = check_interactions(identity, other_drugs)
    print_interactions(interaction_result)

    # Step 5: Final report
    report = build_report(identity, interaction_result, source_mode="image")
    print_report(report)

    # Step 6: Export option
    _export_prompt(report)


# ══════════════════════════════════════════════════════════════
#  TEXT MODE
# ══════════════════════════════════════════════════════════════

def run_text_mode(drug_name: str):
    """Full pipeline starting from a typed drug name (no OCR)."""

    # Build a minimal OCR-like dict for the resolver
    mock_ocr = {
        "success":       True,
        "_source_mode":  "text",       # tells resolver: no physical label to inspect
        "brand_name":    drug_name,
        "generic_name":  drug_name,
        "dosage":        None,
        "dosage_form":   None,
        "batch_no":      None,
        "mfg_date":      None,
        "exp_date":      None,
        "manufacturer":  None,
        "license_no":    None,
        "storage":       None,
        "raw_text":      drug_name,
    }

    print(f"\n  Input mode : TEXT")
    print(f"  Drug name  : {drug_name}")

    # Step 1: Resolve
    identity = resolve_drug(mock_ocr)
    print_identity(identity)

    # Step 2: Get other drugs
    other_drugs = _ask_other_drugs()

    # Step 3: Interaction check
    interaction_result = check_interactions(identity, other_drugs)
    print_interactions(interaction_result)

    # Step 4: Final report
    report = build_report(identity, interaction_result, source_mode="text")
    print_report(report)

    # Step 5: Export option
    _export_prompt(report)


# ══════════════════════════════════════════════════════════════
#  INTERACTIVE MODE (no arguments)
# ══════════════════════════════════════════════════════════════

def run_interactive():
    """Prompt user to choose input mode interactively."""
    print("\n  How would you like to input the drug?")
    print("  [1] Upload / provide an image path")
    print("  [2] Type the drug name manually")
    print()

    while True:
        choice = input("  Enter choice [1/2]: ").strip()
        if choice == "1":
            path = input("  Image path: ").strip().strip('"').strip("'")
            if not os.path.exists(path):
                print(f"  ✗ File not found: {path}")
                continue
            run_image_mode(path)
            break
        elif choice == "2":
            name = input("  Drug name: ").strip()
            if not name:
                print("  ✗ Please enter a drug name.")
                continue
            run_text_mode(name)
            break
        else:
            print("  Please enter 1 or 2.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    _banner()

    parser = argparse.ArgumentParser(
        description="Drug Authenticity & Interaction Checker",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--image", metavar="PATH",
                        help="Path to drug label image (activates OCR mode)")
    parser.add_argument("--drug",  metavar="NAME",
                        help='Drug name in quotes e.g. --drug "metformin"')

    args = parser.parse_args()

    if args.image:
        if not os.path.exists(args.image):
            print(f"\n  ✗ Image not found: {args.image}\n")
            sys.exit(1)
        run_image_mode(args.image)

    elif args.drug:
        run_text_mode(args.drug)

    else:
        run_interactive()


if __name__ == "__main__":
    main()