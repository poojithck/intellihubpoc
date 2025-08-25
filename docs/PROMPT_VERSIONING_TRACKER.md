### Targeted-Prompts v1–v10: Objective Changes Only

Keep JSON schemas and keys unchanged. Below are text-only, per-SOR edits that make each version objectively distinct.

#### v1 — Baseline
- Existing prompts as currently shipped.

#### v2 — Clarify task focus and image selection
- General: Add an explicit “image selection” step where applicable.
- AsbestosBagAndBoard: Pick latest board overview for sticker; also select clearest bag for NMI/date (by timestamp/filename).
- CertificateOfCompliance: State “scan for electrical context keywords, then validate certificate presence.”
- FuseReplacement: “Select image showing the most fuse barrels before counting.”
- MeterConsolidationE4: “Choose earliest and latest full-board images by timestamp/filename order.”
- NeutralLinkInstallation: “Choose earliest and latest full-board overviews.”
- PlugInMeterRemoval: “Choose earliest and latest full-board overviews.”
- ServiceProtectionDevices: “Choose earliest and latest full-board overviews.”
- SwitchInstallation: “Use earliest/latest overviews; use close-ups only to confirm newness.”

#### v2.1 — Stricter visibility gating and disambiguation
- General: Raise qualification thresholds; expand “do not count” lists.
- AsbestosBagAndBoard: Require the word “asbestos” on the sticker; NMI is 10 digits (best guess allowed if partially obscured, note uncertainty).
- CertificateOfCompliance: Require signature and date (or explicit validity markers); ambiguous forms → invalid.
- FuseReplacement: Count only visible fuse barrels; do not count holders/cases unless a fuse is clearly visible inside.
- MeterConsolidationE4: Require visible energy register and meter labels/manufacturer info; black back-plates alone are not meters.
- NeutralLinkInstallation: Require brass block with ≥5 screw terminals and ≥2 connected conductors; plastic blocks/cable bundles are not neutral links.
- PlugInMeterRemoval: Count only when black plate and device body are both clearly present; empty plates not counted.
- ServiceProtectionDevices: Small separate smooth boxes without levers; breakers/meters excluded.
- SwitchInstallation: 1–3 linked toggles with ON/OFF or ISOLATOR; breakers and >3-linked clusters excluded.

#### v3 — Precision-first: uncertain → negative
- General: When ambiguous/partially occluded, prefer NOT to count and set the boolean to negative.
- AsbestosBagAndBoard: Sticker unreadable or bag text unclear → Valid_Claim false.
- CertificateOfCompliance: Illegible/partial forms → Valid_Certificate false.
- FuseReplacement: Barrels partially implied without text/colour cues → do not count.
- MeterConsolidationE4: Enclosures without a clearly visible register → not meters.
- NeutralLinkInstallation: Brass-like shapes without screw rows/conductors → not counted.
- PlugInMeterRemoval: Black plate without a clear device body → not counted.
- ServiceProtectionDevices: Unclear small boxes → not counted.
- SwitchInstallation: Ambiguous toggles without clear labels/linked levers → not counted.

#### v4 — Add concise positive/negative mini-examples (text only)
- General: Add 2–3 bullets per SOR: “Count this” vs “Don’t count this.”
- AsbestosBagAndBoard: Positive: bag with 10-digit NMI and date; board with “asbestos” sticker. Negative: generic caution label; bag without NMI.
- CertificateOfCompliance: Positive: titled certificate with electrician/licence, signature/date. Negative: unrelated invoice/photo.
- FuseReplacement: Positive: cylindrical ceramic/coloured barrel with printed rating. Negative: empty holders/busbars.
- MeterConsolidationE4: Positive: device with LCD/dials and kWh label. Negative: black back-plate only; breaker box.
- NeutralLinkInstallation: Positive: brass block with rows of screws and multiple neutrals connected. Negative: DIN block/cable ties.
- PlugInMeterRemoval: Positive: device body seated on black plate. Negative: empty plate/no device body.
- ServiceProtectionDevices: Positive: small smooth black/translucent box with no lever. Negative: breaker, switch, meter.
- SwitchInstallation: Positive: 1–3 linked toggles marked ON/OFF. Negative: 4+ breakers, unlabeled toggles.

#### v5 — Deterministic image selection rules
- General: Earliest = lowest timestamp/filename; Latest = highest. Tie-breaker = greater board coverage.
- Apply the rule explicitly in AsbestosBagAndBoard, MeterConsolidationE4, NeutralLinkInstallation, PlugInMeterRemoval, ServiceProtectionDevices, SwitchInstallation.

#### v6 — Strong JSON-compliance framing
- General: Repeat at top and bottom: “Return EXACTLY one JSON object; no markdown, no extra keys.”
- Add a single-line reminder immediately before the JSON block: “Output one and only one JSON object.”

#### v7 — Tiling-aware addendum (active only when tiling is enabled by config)
- General: Add hybrid-tiling paragraph: “You may receive one overview plus multiple tiles with filenames/coords. Count/read uniquely using a center-in-safe-zone rule; ignore edge-clipped duplicates.”
- FuseReplacement / ServiceProtectionDevices: Emphasize unique counting across tiles.
- AsbestosBagAndBoard: Emphasize reading NMI/date from tiles and de-duplicating repeated text.

#### v8 — Evidence-in-notes requirement
- General: Require notes to include at least one SOR-specific evidence token (≤25 words).
- AsbestosBagAndBoard: “asbestos sticker”, “10-digit NMI”, “date on bag”.
- CertificateOfCompliance: “licence”, “signature”, “date”.
- FuseReplacement: “barrel”, “printed rating”.
- MeterConsolidationE4: “register/LCD”, “kWh label”.
- NeutralLinkInstallation: “brass block”, “screw terminals”, “neutrals connected”.
- PlugInMeterRemoval: “black plate”, “device body present/absent”.
- ServiceProtectionDevices: “small smooth box”, “no lever”.
- SwitchInstallation: “linked toggles”, “ON/OFF label”.

#### v9 — Minimalist phrasing (brevity-first)
- General: Compress to short numbered steps; remove non-essential prose; keep JSON block identical.
- Apply uniformly across SORs to reduce instruction tokens without changing logic.

#### v10 — Rule hierarchy with explicit priorities
- General: Add “Primary rules > Supporting cues > Forbidden items”. When conflicts arise, Primary rules override.
- AsbestosBagAndBoard: Primary = NMI on bag (10 digits/best guess) and “asbestos” sticker present; Supporting = board context; Forbidden = unrelated labels.
- CertificateOfCompliance: Primary = certificate indicators (title, licence, signature, date); Supporting = context keywords; Forbidden = invoices/screenshots without certificate semantics.
- FuseReplacement: Primary = visible fuse barrel; Supporting = printed rating/colour banding; Forbidden = holders/busbars/breakers.
- MeterConsolidationE4: Primary = visible energy register; Supporting = meter labels/manufacturer; Forbidden = empty plates/enclosures.
- NeutralLinkInstallation: Primary = brass block with screw rows and connected neutrals; Supporting = labels; Forbidden = plastic blocks/cable bundles.
- PlugInMeterRemoval: Primary = device body on black plate; Supporting = plate alignment; Forbidden = empty plates/white enclosures.
- ServiceProtectionDevices: Primary = small separate smooth box with no lever; Supporting = location separate from meters; Forbidden = breakers/switches/meters.
- SwitchInstallation: Primary = 1–3 linked toggles with ON/OFF; Supporting = isolator labels; Forbidden = >3 breakers/unlabeled toggles.

Optional (version-aligned) fallback keyword tweaks
- v3: Expand negatives (e.g., “unclear”, “unknown”, “ambiguous”) to bias away from uncertain positives.
- v8: Add evidence tokens to positives to align with notes requirements (e.g., “barrel”, “kWh”, “black plate”).


