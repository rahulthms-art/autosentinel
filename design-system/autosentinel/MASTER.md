# AutoSentinel design system: MASTER

Generated with the UI UX Pro Max dataset (https://github.com/nextlevelbuilder/ui-ux-pro-max-skill, MIT).
Matched product types: **Analytics Dashboard (#7)** + **Legal / compliance (#40)**.
This file is the single source of truth. A page-specific file in `pages/<page>.md` overrides it.

## Pattern & style
- **Pattern:** Data-Dense + Drill-Down, with Trust & Authority + Minimal.
- **Style:** Data-Dense Dashboard + Minimalism & Swiss Style, Accessible & Ethical.
- Compact cards (12px padding, 8px gaps), KPI row first, filters above tables, sticky table headers.
- Subtle hover (200 to 250ms). No decorative motion. Respect reduced motion.

## Color tokens (light)
| Token | Hex | Use |
|---|---|---|
| primary | `#1E3A8A` | Authority navy: primary buttons, active tab, links |
| secondary | `#1E40AF` | Secondary emphasis |
| accent | `#B45309` | Amber: "new", warnings, attention (4.5:1 on white) |
| destructive | `#DC2626` | High severity, do-not-drive, unrecalled spike |
| success | `#15803D` | Recall exists / actioned |
| background | `#F8FAFC` | App background |
| card | `#FFFFFF` | Cards, tables |
| foreground | `#0F172A` | Body text |
| muted | `#E9EEF5` | Sidebar, secondary surfaces |
| muted-fg | `#475569` | Captions, labels |
| border | `#CBD5E1` | Card and table borders |

## Typography
- **IBM Plex Sans** (Financial Trust pairing #31: "conveys trust and professionalism, excellent for data") for all UI text.
- **IBM Plex Mono** for codes and numbers: campaign IDs, VINs, ratios.
- Scale: H1 28/600, H2 20/600, body 15/400, caption 13/400.

## Icons
- Material Symbols (Rounded) only. **No emoji icons.**

## Charts
- **Anomaly detection (complaint spikes):** line chart with highlights. Normal series in blue tones, each with a distinct dash style. Spike points are red circles with a text label. The recent 90-day window is an alert band (`#FFF3CD`). Always show the spike table as the accessible fallback.
- **Trend over time:** at most 6 series, direct labels, never distinguish series by hue alone.

## Must-haves
- Data export for every table (CSV) and the brief (Markdown).
- Filtering on every findings list.
- Severity is always **text + icon + color**, never color alone.

## Anti-patterns (do not)
- Ornate or decorative design, AI purple/pink gradients, emoji icons.
- Tables without filters or export.
- Color as the only status signal.

## Pre-delivery checklist
- [x] Text contrast ≥ 4.5:1 (navy/slate on white; amber #B45309 chosen over #F59E0B for contrast)
- [x] Visible focus states (2px navy outline)
- [x] No horizontal scroll at 375px; navigation and chips wrap (verified)
- [x] Every chart has a table fallback (spike table)
- [x] Loading and empty states on every data view
- [x] Data export on findings, fleet results, audit log and briefs

## Implementation notes (Streamlit)
- Theme tokens: `.streamlit/config.toml`. Streamlit takes only one font per URL.
- Navigation uses a persistent segmented control instead of `st.tabs`, which resets to the first tab on widget changes.
- Findings use the drill-down pattern: a read-only table with row selection opens a detail card with a status control.
