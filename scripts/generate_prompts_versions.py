#!/usr/bin/env python3
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_BASE = ROOT / "configs" / "prompts" / "Targeted-Prompts"


SORS = [
    "AsbestosBagAndBoard",
    "CertificateOfCompliance",
    "FuseReplacement",
    "MeterConsolidationE4",
    "NeutralLinkInstallation",
    "PlugInMeterRemoval",
    "ServiceProtectionDevices",
    "SwitchInstallation",
]

VERSIONS = ["v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]


FALLBACK = {
    "AsbestosBagAndBoard": {
        "positive": ["asbestos bag", "asbestos sticker", "identifier", "date", "valid claim"],
        "negative": ["no bag", "no sticker", "missing identifier", "invalid claim", "unknown"],
    },
    "CertificateOfCompliance": {
        "positive": [
            "certificate of compliance", "electrical", "work", "safety", "electrician",
            "licence", "install", "signature", "date", "valid certificate"
        ],
        "negative": ["no certificate", "missing", "not found", "invalid", "unknown"],
    },
    "FuseReplacement": {
        # Use the more explicit v2.1 positives (include visible fuse body cues)
        "positive": [
            "fuses", "even count", "replacement", "valid claim", "clear image",
            "cylindrical", "printed", "label", "barrel", "cartridge", "visible fuse body"
        ],
        "negative": [
            "no fuses", "odd count", "invalid claim", "unclear", "missing", "unknown",
            "holder only", "empty holder", "plastic case", "enclosure", "black box"
        ],
    },
    "MeterConsolidationE4": {
        "positive": ["consolidation", "fewer meters", "removed", "reduction"],
        "negative": ["no change", "same number", "unchanged", "no consolidation", "unknown"],
    },
    "NeutralLinkInstallation": {
        "positive": ["neutral link", "brass block", "terminal block"],
        # Use v2.1 negatives to exclude meter-related terms explicitly
        "negative": [
            "no neutral link", "no brass block", "unchanged", "same number",
            "no installation", "unknown", "smart meter", "digital meter", "meter"
        ],
    },
    "PlugInMeterRemoval": {
        "positive": [
            "plug-in meter", "plug-in device", "black mounting plate", "removed", "fewer meters", "reduction"
        ],
        "negative": ["no change", "same number", "unchanged", "no removal", "unknown"],
    },
    "ServiceProtectionDevices": {
        "positive": ["SPD", "MPD", "black box", "translucent box", "protection device", "small box"],
        "negative": ["no device", "unchanged", "same number", "no addition", "unknown", "meter"],
    },
    "SwitchInstallation": {
        "positive": ["switch", "installed", "new switch", "additional switch", "switch added"],
        "negative": [
            "no switch", "unchanged", "same number", "no installation", "unknown",
            "meter", "smart meter", "digital meter"
        ],
    },
}


REQUIRED_FIELDS = {
    "AsbestosBagAndBoard": ["National_Meter_Identifier", "Date_On_Bag", "Valid_Claim", "Notes"],
    "CertificateOfCompliance": ["Electrical_Work_Present", "Valid_Certificate", "Notes"],
    "FuseReplacement": ["fuse_image", "fuse_count", "is_even_count", "valid_claim", "notes"],
    # Use v2/v2.1 richer schema
    "MeterConsolidationE4": ["init_count", "final_count", "init_image", "final_image", "consolidation", "notes"],
    "NeutralLinkInstallation": ["init_count", "final_count", "init_image", "final_image", "neutral_link_installed", "notes"],
    "PlugInMeterRemoval": ["init_count", "final_count", "meters_removed", "notes"],
    "ServiceProtectionDevices": ["init_count", "final_count", "devices_added", "notes"],
    "SwitchInstallation": ["init_count", "final_count", "init_image", "final_image", "switch_installed", "notes"],
}


def json_block(sor: str) -> str:
    if sor == "AsbestosBagAndBoard":
        return (
            '{\n'
            '  "National_Meter_Identifier": "<number_or_code>",\n'
            '  "Date_On_Bag": "<date>",\n'
            '  "Valid_Claim": <true|false>,\n'
            '  "Notes": "<≤25 words>"\n'
            '}'
        )
    if sor == "CertificateOfCompliance":
        return (
            '{\n'
            '  "Electrical_Work_Present": <true|false>,\n'
            '  "Valid_Certificate": <true|false>,\n'
            '  "Notes": "<=25 words"\n'
            '}'
        )
    if sor == "FuseReplacement":
        return (
            '{\n'
            '  "fuse_image": "<filename_of_image_showing_most_fuses>",\n'
            '  "fuse_count": <integer_total_number_of_visible_fuses_only>,\n'
            '  "is_even_count": <true|false>,\n'
            '  "valid_claim": <true|false>,\n'
            '  "notes": "<brief (≤25 words) justification mentioning key visual evidence>"\n'
            '}'
        )
    if sor == "MeterConsolidationE4":
        return (
            '{\n'
            '  "init_count": <integer>,\n'
            '  "final_count": <integer>,\n'
            '  "init_image": "<init filename>",\n'
            '  "final_image": "<final filename>",\n'
            '  "consolidation": <true|false>,\n'
            '  "notes": "<=25 words citing key visual evidence"\n'
            '}'
        )
    if sor == "NeutralLinkInstallation":
        return (
            '{\n'
            '  "init_count": <integer>,\n'
            '  "final_count": <integer>,\n'
            '  "init_image": "<init filename>",\n'
            '  "final_image": "<final filename>",\n'
            '  "neutral_link_installed": <true|false>,\n'
            '  "notes": "<=25 words citing key visual evidence (must mention \'neutral link\' or \'no neutral link\')"\n'
            '}'
        )
    if sor == "PlugInMeterRemoval":
        return (
            '{\n'
            '  "init_count": <integer>,\n'
            '  "final_count": <integer>,\n'
            '  "meters_removed": <true|false>,\n'
            '  "notes": "<≤25 words>"\n'
            '}'
        )
    if sor == "ServiceProtectionDevices":
        return (
            '{\n'
            '  "init_count": <integer>,\n'
            '  "final_count": <integer>,\n'
            '  "devices_added": <true|false>,\n'
            '  "notes": "<≤25 words>"\n'
            '}'
        )
    if sor == "SwitchInstallation":
        return (
            '{\n'
            '  "init_count": <integer>,\n'
            '  "final_count": <integer>,\n'
            '  "init_image": "<init filename>",\n'
            '  "final_image": "<final filename>",\n'
            '  "switch_installed": <true|false>,\n'
            '  "notes": "<≤25 words citing key visual evidence>"\n'
            '}'
        )
    raise ValueError(f"Unknown SOR {sor}")


def base_system_prompt(sor: str) -> str:
    if sor == "AsbestosBagAndBoard":
        return (
            "You are an expert utility-meter technician and photo analyst. Analyze ONLY asbestos-related images to: \n"
            "  1) extract the NMI (10-digit) from the bag, 2) extract the date on the bag, 3) detect an asbestos sticker on the board,\n"
            "  4) set Valid_Claim true only if bag (with NMI/date) AND board sticker are present."
        )
    if sor == "CertificateOfCompliance":
        return (
            "You are an electrical compliance officer. Determine if a valid certificate of compliance is present in the provided images.\n"
            "Certificates may be photos of hand-filled forms or screenshots/photos of PDFs."
        )
    if sor == "FuseReplacement":
        return (
            "You are an expert electrical technician analyzing evidence of fuse replacement. Count ONLY visible fuse barrels (cartridge bodies).\n"
            "Holders/cases without visible barrels do not count. Both installed and loose fuses qualify if the barrel is visible."
        )
    if sor == "MeterConsolidationE4":
        return (
            "You will receive meter-board overviews. Count actual meters in earliest and latest images and determine if consolidation occurred (init_count > final_count).\n"
            "A valid meter shows a visible energy register and meter-specific labeling/manufacturer details."
        )
    if sor == "NeutralLinkInstallation":
        return (
            "You will receive meter-board images. Decide whether a brass neutral link was newly installed between earliest and latest views.\n"
            "Count as neutral link only if a brass block with rows of screw terminals and ≥2 connected conductors is visible."
        )
    if sor == "PlugInMeterRemoval":
        return (
            "You will receive ONLY meter-board overviews. Identify plug-in devices using black mounting plates/back-plates and count installed devices in earliest vs latest."
        )
    if sor == "ServiceProtectionDevices":
        return (
            "You will receive ONLY relevant images. Identify small protection devices (SPD/MPD): small rectangular black or translucent grey boxes,\n"
            "separate from meters, with smooth enclosures and no switches/levers. Count earliest vs latest and detect additions."
        )
    if sor == "SwitchInstallation":
        return (
            "You will receive relevant images (switch close-ups and board overviews). Detect installation of 1–3 phase isolating switches between earliest and latest views.\n"
            "A valid switch has 1–3 linked toggles and ON/OFF or ISOLATOR labels."
        )
    raise ValueError(f"Unknown SOR {sor}")


def examples_block(sor: str) -> str:
    if sor == "AsbestosBagAndBoard":
        return (
            "Examples — Count this: bag shows 10-digit NMI and date; board sticker includes 'asbestos'.\n"
            "Don’t count: generic caution label; bag text without a 10-digit NMI."
        )
    if sor == "CertificateOfCompliance":
        return (
            "Examples — Valid: certificate title, electrician/licence, signature, date.\n"
            "Invalid: unrelated invoice/photo without compliance semantics."
        )
    if sor == "FuseReplacement":
        return (
            "Examples — Count this: ceramic/coloured cylindrical barrel with printed rating.\n"
            "Don’t count: empty holders, busbars, breakers, plastic enclosures with no visible barrel."
        )
    if sor == "MeterConsolidationE4":
        return (
            "Examples — Count this: device with LCD/dials and kWh label.\n"
            "Don’t count: black back-plate only; breaker/switch enclosure."
        )
    if sor == "NeutralLinkInstallation":
        return (
            "Examples — Count this: brass block with ≥5 screw terminals and ≥2 connected conductors.\n"
            "Don’t count: DIN rail blocks, plastic blocks, cable bundles."
        )
    if sor == "PlugInMeterRemoval":
        return (
            "Examples — Count this: device body seated on a black plate.\n"
            "Don’t count: empty black plate without device body."
        )
    if sor == "ServiceProtectionDevices":
        return (
            "Examples — Count this: small smooth black/translucent box with no lever.\n"
            "Don’t count: breakers, switches, meters."
        )
    if sor == "SwitchInstallation":
        return (
            "Examples — Count this: 1–3 linked toggles with ON/OFF.\n"
            "Don’t count: 4+ breakers, unlabeled toggles."
        )
    return ""


def tiling_addendum(sor: str) -> str:
    common = (
        "You may receive one low-res overview plus multiple tiles with filenames/coordinates.\n"
        "Count/read uniquely across tiles using a center-in-safe-zone rule; ignore objects clipped at tile edges to avoid duplicates."
    )
    if sor in {"FuseReplacement", "ServiceProtectionDevices"}:
        return common + "\nEmphasize unique counting across tiles when objects span tile boundaries."
    if sor == "AsbestosBagAndBoard":
        return common + "\nUse tiles to read small text (NMI/date). If the same text appears in multiple tiles, de-duplicate; return one value."
    return common


def image_selection_rule(sor: str) -> str:
    if sor in {"AsbestosBagAndBoard", "MeterConsolidationE4", "NeutralLinkInstallation", "PlugInMeterRemoval", "ServiceProtectionDevices", "SwitchInstallation"}:
        return (
            "Use deterministic selection: Earliest = lowest timestamp/filename; Latest = highest.\n"
            "If tied, choose the image with greater board coverage."
        )
    return ""


def base_main_prompt(sor: str) -> str:
    # Default structure; per-SOR specifics appended below
    if sor == "AsbestosBagAndBoard":
        return (
            "Tasks:\n"
            "  - Extract NMI (10-digit) from bag and the date on the bag.\n"
            "  - Detect an 'asbestos' sticker on the board.\n"
            "  - Valid_Claim = true only if bag (NMI/date) AND board sticker are present; else false.\n"
            "  - Provide brief notes (≤25 words).\n"
        )
    if sor == "CertificateOfCompliance":
        return (
            "Tasks:\n"
            "  - Detect electrical context and confirm presence of a valid certificate.\n"
            "  - Provide brief reason (≤25 words).\n"
        )
    if sor == "FuseReplacement":
        return (
            "Tasks:\n"
            "  1) Select the image showing the most fuse barrels.\n"
            "  2) Count ONLY visible fuse barrels (printed text/colour body cues).\n"
            "  3) is_even_count reflects parity; valid_claim true only if even.\n"
            "  4) Provide notes (≤25 words) mentioning key evidence.\n"
        )
    if sor == "MeterConsolidationE4":
        return (
            "Tasks:\n"
            "  1) Choose earliest and latest full-board images.\n"
            "  2) Count ONLY actual meters (visible register + meter labels/manufacturer).\n"
            "  3) consolidation = true if init_count > final_count.\n"
            "  4) Provide notes (≤25 words).\n"
        )
    if sor == "NeutralLinkInstallation":
        return (
            "Tasks:\n"
            "  1) Choose earliest and latest full-board images.\n"
            "  2) Count neutral links only if brass block with rows of screws and ≥2 conductors is visible.\n"
            "  3) neutral_link_installed = true if final_count > init_count and at least one valid block is present.\n"
            "  4) Provide notes (≤25 words).\n"
        )
    if sor == "PlugInMeterRemoval":
        return (
            "Tasks:\n"
            "  1) Choose earliest and latest full-board images.\n"
            "  2) Count plug-in devices ONLY when both black plate AND device body are present.\n"
            "  3) meters_removed = true if final_count < init_count.\n"
            "  4) Provide notes (≤25 words).\n"
        )
    if sor == "ServiceProtectionDevices":
        return (
            "Tasks:\n"
            "  1) Choose earliest and latest full-board images.\n"
            "  2) Count SPD/MPD units: small separate smooth boxes with no levers (not meters/breakers).\n"
            "  3) devices_added = true if final_count > init_count.\n"
            "  4) Provide notes (≤25 words).\n"
        )
    if sor == "SwitchInstallation":
        return (
            "Tasks:\n"
            "  1) Choose earliest and latest full-board images; use close-ups to confirm newness.\n"
            "  2) Detect 1–3 linked toggles labeled ON/OFF or ISOLATOR.\n"
            "  3) switch_installed = true if ≥1 new switch appears.\n"
            "  4) Provide notes (≤25 words).\n"
        )
    raise ValueError(f"Unknown SOR {sor}")


def apply_version_overrides(version: str, sor: str, sys_text: str, main_text: str) -> tuple[str, str]:
    # v3 — Precision-first uncertain -> negative
    if version == "v3":
        sys_text += ("\n\nWhen evidence is ambiguous or partially occluded, prefer NOT to count and set the decision negative.")

    # v4 — Add concise examples
    if version == "v4":
        sys_text += ("\n\n" + examples_block(sor))

    # v5 — Deterministic image selection rules
    if version == "v5":
        add = image_selection_rule(sor)
        if add:
            main_text = add + "\n\n" + main_text

    # v6 — Strong JSON compliance framing
    if version == "v6":
        main_text = (
            "Output one and only one JSON object. No markdown or extra keys.\n\n" + main_text +
            "\nReturn EXACTLY this JSON (no extra keys, no markdown):"
        )
    else:
        main_text = main_text + "\nReturn EXACTLY this JSON (no extra keys, no markdown):"

    # v7 — Tiling-aware addendum
    if version == "v7":
        main_text += ("\n\n" + tiling_addendum(sor))

    # v8 — Evidence-in-notes requirement
    if version == "v8":
        evidence = {
            "AsbestosBagAndBoard": "Notes must mention evidence terms like 'asbestos sticker', '10-digit NMI', or 'date on bag'.",
            "CertificateOfCompliance": "Notes must include a validity indicator such as 'licence', 'signature', or 'date'.",
            "FuseReplacement": "Notes must include terms like 'barrel' or 'printed rating'.",
            "MeterConsolidationE4": "Notes must include 'register/LCD' or 'kWh'.",
            "NeutralLinkInstallation": "Notes must include 'brass block', 'screw terminals', or 'neutrals connected'.",
            "PlugInMeterRemoval": "Notes must include 'black plate' and 'device body present/absent'.",
            "ServiceProtectionDevices": "Notes must include 'small smooth box' or 'no lever'.",
            "SwitchInstallation": "Notes must include 'linked toggles' or 'ON/OFF'.",
        }
        main_text = evidence[sor] + "\n\n" + main_text

    # v9 — Minimalist phrasing
    if version == "v9":
        minimal = {
            "AsbestosBagAndBoard": (
                "1) Read 10-digit NMI + date from bag. 2) Detect 'asbestos' sticker. 3) Valid_Claim = bag+sticker. 4) Notes ≤25 words."
            ),
            "CertificateOfCompliance": (
                "1) Confirm electrical context. 2) Validate certificate presence. 3) Notes ≤25 words."
            ),
            "FuseReplacement": (
                "1) Pick image with most barrels. 2) Count visible barrels only. 3) Even = valid. 4) Notes ≤25 words."
            ),
            "MeterConsolidationE4": (
                "1) Earliest vs latest overviews. 2) Count meters (visible register). 3) consolidation = init>final. 4) Notes ≤25 words."
            ),
            "NeutralLinkInstallation": (
                "1) Earliest vs latest. 2) Count brass neutral links (screws + ≥2 conductors). 3) installed = final>init. 4) Notes ≤25 words."
            ),
            "PlugInMeterRemoval": (
                "1) Earliest vs latest. 2) Count plug-ins only when body on black plate. 3) removed = final<init. 4) Notes ≤25 words."
            ),
            "ServiceProtectionDevices": (
                "1) Earliest vs latest. 2) Count SPD/MPD small smooth boxes (no levers). 3) added = final>init. 4) Notes ≤25 words."
            ),
            "SwitchInstallation": (
                "1) Earliest vs latest. 2) Detect 1–3 linked toggles with ON/OFF. 3) installed if ≥1 new. 4) Notes ≤25 words."
            ),
        }
        main_text = minimal[sor] + "\n\nReturn EXACTLY this JSON (no extra keys, no markdown):"

    # v10 — Rule hierarchy
    if version == "v10":
        hierarchy = {
            "AsbestosBagAndBoard": (
                "Primary: bag 10-digit NMI + date AND board 'asbestos' sticker.\n"
                "Supporting: board overview context.\n"
                "Forbidden: unrelated labels, non-10-digit codes."
            ),
            "CertificateOfCompliance": (
                "Primary: certificate indicators (title, licence, signature, date).\n"
                "Supporting: electrical context keywords.\n"
                "Forbidden: invoices/screenshots without certificate semantics."
            ),
            "FuseReplacement": (
                "Primary: visible fuse barrel.\n"
                "Supporting: printed rating/colour banding.\n"
                "Forbidden: holders/busbars/breakers/enclosures without barrels."
            ),
            "MeterConsolidationE4": (
                "Primary: visible energy register.\n"
                "Supporting: meter labels/manufacturer info.\n"
                "Forbidden: empty plates/enclosures/breakers."
            ),
            "NeutralLinkInstallation": (
                "Primary: brass block with rows of screws and ≥2 connected conductors.\n"
                "Supporting: labels/markings.\n"
                "Forbidden: plastic blocks/cable bundles."
            ),
            "PlugInMeterRemoval": (
                "Primary: device body on black plate.\n"
                "Supporting: plate alignment/fastening.\n"
                "Forbidden: empty black plates/white enclosures."
            ),
            "ServiceProtectionDevices": (
                "Primary: small separate smooth box with no lever.\n"
                "Supporting: separate from meters; typical SPD/MPD location.\n"
                "Forbidden: breakers, switches, meters."
            ),
            "SwitchInstallation": (
                "Primary: 1–3 linked toggles with ON/OFF/ISOLATOR.\n"
                "Supporting: proximity/labels.\n"
                "Forbidden: >3 breakers/unlabeled toggles."
            ),
        }
        sys_text = hierarchy[sor] + "\n\n" + sys_text

    return sys_text, main_text


def render_yaml(sor: str, version: str) -> str:
    sys_text = base_system_prompt(sor)
    main_text = base_main_prompt(sor)
    sys_text, main_text = apply_version_overrides(version, sor, sys_text, main_text)

    # YAML with two-space indentation, match existing style
    fb = FALLBACK[sor]
    rf = REQUIRED_FIELDS[sor]
    json_str = json_block(sor)

    # Build fallback arrays inline like existing files
    pos = ", ".join([f'"{p}"' for p in fb["positive"]])
    neg = ", ".join([f'"{n}"' for n in fb["negative"]])
    req = ", ".join([f'"{r}"' for r in rf])

    yaml = []
    yaml.append(f"{sor}:")
    yaml.append("  system_prompt: |")
    for line in sys_text.splitlines():
        yaml.append(f"    {line}")
    yaml.append("")
    yaml.append("  main_prompt: |")
    for line in main_text.splitlines():
        yaml.append(f"    {line}")
    yaml.append("    ")
    yaml.append("    " + "Return EXACTLY this JSON (no extra keys, no markdown):")
    for line in json_str.splitlines():
        yaml.append(f"    {line}")
    yaml.append("")
    yaml.append("  fallback_keywords:")
    yaml.append(f"    positive: [{pos}]")
    yaml.append(f"    negative: [{neg}]")
    yaml.append("")
    yaml.append("  response_format:")
    yaml.append(f"    required_fields: [{req}]")
    yaml.append("")
    yaml.append("  model_config: \"analysis\"")
    yaml.append("")

    return "\n".join(yaml)


def main():
    for version in VERSIONS:
        out_dir = TARGET_BASE / version
        out_dir.mkdir(parents=True, exist_ok=True)
        for sor in SORS:
            content = render_yaml(sor, version)
            out_file = out_dir / f"{sor}.yaml"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()


