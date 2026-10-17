"""AutoSentinel: Recall & Regulation Radar.
An agent platform that continuously monitors recalls, regulations, supplier risk and owner-complaint
signals for automotive teams. NVIDIA Nemotron on Nebius Token Factory + Tavily + NHTSA open data.
Design system: design-system/autosentinel/MASTER.md (generated with UI UX Pro Max).

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

import ui
from radar import analyst, fleet
from radar.anonymize import Anonymizer
from radar.budget import BudgetExceeded, Guard, LimitReached, budget_usd, call_cost, demo_mode
from radar.config import TOPICS, Settings
from radar.db import DB
from radar.llm import Router, list_nvidia_models, make_client, pick_default
from radar.pipeline import run_watchlist
from radar.seed import ensure_default
from radar.snapshot import load_if_fresh, migrate_legacy
from radar.signals import detect_spikes, monthly_series, recall_data_suspect
from radar.sources import nhtsa

st.set_page_config(page_title="AutoSentinel Radar", page_icon=":material/radar:", layout="wide")
ui.inject_css()

# Streamlit Cloud secrets → env, so Settings() picks them up
try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass


@st.cache_resource
def get_db(path: str) -> DB:
    db = DB(path)
    db.mark_interrupted_runs()  # runs once per server process
    ensure_default(db)
    return db


@st.cache_data(ttl=3600, show_spinner=False)
def nvidia_models(key: str, base_url: str) -> list[str]:
    return list_nvidia_models(make_client(key, base_url))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_nhtsa(make: str, model: str, years: tuple) -> tuple[list, list]:
    recs, comps = [], []
    for y in years:
        try:
            recs += nhtsa.recalls(make, model, y)
            comps += nhtsa.complaints(make, model, y)
        except Exception:
            pass
    return recs, comps


settings = Settings()
db = get_db(settings.db_path)
if demo_mode():  # checked every run, so turning DEMO_MODE on later still loads the stored demo results
    migrate_legacy(db)
    load_if_fresh(db, Path(__file__).parent / "demo" / "snapshot.json")

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### :material/radar: AutoSentinel")
    st.caption("Recall & Regulation Radar")

    wls = db.watchlists()
    names = {w["name"]: w["id"] for w in wls}
    ui.eyebrow("Watchlist")
    anon_default = st.query_params.get("anon", os.getenv("ANONYMIZE", "")) in ("1", "true", "yes")
    anon = st.session_state.get("anon", anon_default)
    wl_name = st.selectbox("Watchlist", list(names), label_visibility="collapsed",
                           format_func=lambda n: Anonymizer(enabled=anon)(n)) if names else None
    wl_id = names.get(wl_name) if wl_name else None
    wl = db.watchlist(wl_id) if wl_id else None
    anon = st.toggle("Anonymize brands", value=anon_default, key="anon",
                     help="For screen recordings: hides vehicle makes, models, suppliers, VINs, recall numbers and "
                          "links. Display only; the data and AI stay real.")
    A = Anonymizer((wl or {}).get("config", {}).get("vehicles", []), enabled=anon)

    st.divider()
    ui.eyebrow("Connections")
    if not settings.nebius_api_key:
        settings.nebius_api_key = st.text_input("Nebius Token Factory key", type="password")
    if not settings.tavily_api_key:
        settings.tavily_api_key = st.text_input("Tavily key", type="password")

    models: list[str] = []
    if settings.nebius_api_key:
        try:
            models = nvidia_models(settings.nebius_api_key, settings.nebius_base_url)
        except Exception as e:
            st.error(f"Token Factory: {e}", icon=":material/error:")
    st.badge("Nemotron · Token Factory" if models else "Nemotron · not connected",
             icon=":material/check_circle:" if models else ":material/radio_button_unchecked:",
             color="green" if models else "gray")
    st.badge("Tavily search" if settings.tavily_api_key else "Tavily · not connected",
             icon=":material/check_circle:" if settings.tavily_api_key else ":material/radio_button_unchecked:",
             color="green" if settings.tavily_api_key else "gray")
    st.badge("NHTSA open data", icon=":material/check_circle:", color="green")

    if models:
        with st.expander("Model routing", icon=":material/alt_route:"):
            fast_def = settings.fast_model or pick_default(models, ["nano", "super", "nemotron"])
            deep_def = settings.deep_model or pick_default(models, ["ultra", "super", "nemotron"])
            settings.fast_model = st.selectbox("Fast tier: planning, triage", models,
                                               index=models.index(fast_def) if fast_def in models else 0)
            settings.deep_model = st.selectbox("Reasoning tier: briefs, signals, analyst", models,
                                               index=models.index(deep_def) if deep_def in models else 0)
    ai_ready = bool(models and settings.tavily_api_key)
    if not ai_ready:
        st.caption("Connect Nebius and Tavily to run the agents. Early Warning and Fleet Check work without keys.")

    guard = Guard(db)
    spent, cap = guard.spent(), budget_usd()
    st.divider()
    ui.eyebrow("AI budget")
    st.progress(min(spent / cap, 1.0) if cap else 1.0,
                text=f"${spent:.2f} of ${cap:.2f} estimated Nemotron spend")
    if demo_mode():
        st.caption("Public demo: AI runs and questions are rate-limited per day.")


def make_router() -> Router:
    # Every call is budget-checked first and logged (priced) afterwards
    return Router(make_client(settings.nebius_api_key, settings.nebius_base_url),
                  settings.fast_model, settings.deep_model,
                  on_call=lambda rec: db.log_llm(None, rec), before_call=guard.check_budget)


SECTIONS = [":material/dashboard: Dashboard", ":material/monitoring: Early warning",
            ":material/directions_car: Fleet check", ":material/forum: Ask the analyst",
            ":material/visibility: Watchlists", ":material/receipt_long: Runs & audit"]
# Persistent navigation (st.tabs resets to the first tab when a widget inside another tab changes)
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="nav",
                               label_visibility="collapsed") or SECTIONS[0]

# ---------------------------------------------------------------- Dashboard
if section == SECTIONS[0]:
    if not wl:
        st.info("Create a watchlist to start monitoring.", icon=":material/info:")
    else:
        head = st.columns([4, 1], vertical_alignment="bottom")
        with head[0]:
            ui.eyebrow("Watchlist")
            st.title(A(wl["name"]), anchor=False)
            cfg = wl["config"]
            st.caption(" · ".join(filter(None, [
                ", ".join(A.vehicle(v["make"], v["model"]) for v in cfg.get("vehicles", [])),
                ", ".join(cfg.get("markets", [])),
                ", ".join(cfg.get("topics", []))])))
        run_clicked = head[1].button("Run radar", icon=":material/play_arrow:", type="primary",
                                     disabled=not ai_ready, width="stretch",
                                     help=None if ai_ready else "Connect Nebius and Tavily first")
        if run_clicked:
            try:
                guard.allow("run")
            except (LimitReached, BudgetExceeded) as e:
                st.warning(str(e), icon=":material/savings:")
                run_clicked = False
        if run_clicked:
            guard.record("run")
            with st.status("Radar running...", expanded=True) as status:
                try:
                    out = run_watchlist(db, make_router(), settings, wl_id, progress=lambda m: st.write(A(m)))
                    status.update(label=f"Run {out['run_id']} complete", state="complete", expanded=False)
                except BudgetExceeded as e:
                    status.update(label="Stopped: budget reached", state="error")
                    st.warning(str(e), icon=":material/savings:")
                except Exception as e:
                    status.update(label="Run failed", state="error")
                    st.exception(e)

        runs = db.runs(wl_id)
        last_ok = next((r for r in runs if r["status"] == "ok"), None)
        stats = last_ok["stats"] if last_ok else {}
        all_f = db.findings(wl_id)
        open_f = [f for f in all_f if f["status"] in ("new", "acknowledged")]

        k = st.columns(5)
        k[0].metric("Open items", len(open_f), border=True)
        k[1].metric("Open · High severity", sum(f["severity"] == "High" for f in open_f), border=True)
        k[2].metric("New in last run", stats.get("relevant", 0), border=True)
        k[3].metric("Early-warning signals", stats.get("unrecalled_signals", 0), border=True,
                    help="Complaint spikes with no matching recall")
        k[4].metric("Last run", (last_ok["finished_at"] or "")[5:16].replace("T", " ") if last_ok else "Never",
                    border=True)

        if stats.get("source") == "snapshot":  # disappears after the first live run
            when = pd.Timestamp(last_ok["finished_at"]).strftime("%-d %b %Y")
            st.info(f"Showing stored results from a real run on {when}. Press **Run radar** for a live update: "
                    "only items that are new since then will be added.", icon=":material/history:")
        if stats.get("web_search_errors"):
            st.warning(f"{stats['web_search_errors']} web search(es) failed in the last run (Tavily). "
                       "Results may be incomplete. Run again later.", icon=":material/cloud_off:")
        if last_ok and last_ok["brief"]:
            with st.expander("Risk brief: latest run", icon=":material/summarize:", expanded=True):
                st.markdown(A(last_ok["brief"]))
                st.download_button("Download brief", A(last_ok["brief"]), icon=":material/download:",
                                   file_name=f"radar-brief-run{last_ok['id']}.md")
        elif not runs:
            st.info("No runs yet. Connect your keys and press **Run radar**: the agent collects recalls, "
                    "regulations and supplier news, then writes a cited brief.", icon=":material/rocket_launch:")

        st.subheader("Findings", anchor=False)
        f1, f2, f3 = st.columns(3)
        sev = f1.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        typ = f2.multiselect("Type", ["recall", "regulation", "supplier", "other"],
                             default=["recall", "regulation", "supplier"])
        sts = f3.multiselect("Status", ["new", "acknowledged", "actioned", "dismissed"],
                             default=["new", "acknowledged"])
        rows = [f for f in all_f if f["severity"] in sev and f["type"] in typ and f["status"] in sts]
        if rows:
            table = pd.DataFrame([{
                "Severity": ui.SEVERITY.get(f["severity"], ui.SEVERITY["Low"])["label"],
                "Status": f["status"], "Type": f["type"], "Market": f["market"],
                "Title": f["title"], "Affected": f["affected"], "Published": f["published"],
            } for f in rows])
            sel = st.dataframe(ui.style_findings(A.frame(table)), hide_index=True, width="stretch",
                               on_select="rerun", selection_mode="single-row", key="findings_table",
                               column_config={"Title": st.column_config.TextColumn(width="large")})
            st.download_button("Export findings (CSV)", A.frame(pd.DataFrame(rows).drop(columns=["raw"])).to_csv(index=False),
                               icon=":material/download:", file_name="findings.csv")

            picked = sel.selection.rows if sel and sel.selection else []
            if picked:
                f = rows[picked[0]]
                with st.container(border=True):
                    ui.eyebrow(f"{f['type']} · {f['market'] or 'Global'} · {f['source'].upper()}")
                    st.markdown(f"#### {A(f['title'])}")
                    ui.severity_badge(f["severity"])
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Affected**  \n{A(f['affected']) or 'n/a'}")
                    c2.markdown(f"**Recommended action**  \n{A(f['action']) or 'n/a'}")
                    st.markdown(A(f["summary"]) or "")
                    if f["url"] and not anon:
                        st.link_button("Open source", f["url"], icon=":material/open_in_new:")
                    new_status = st.segmented_control(
                        "Status", ["new", "acknowledged", "actioned", "dismissed"], default=f["status"],
                        format_func=lambda s: s.capitalize(), key=f"status_{f['id']}")
                    if new_status and new_status != f["status"]:
                        db.set_status(f["id"], new_status)
                        st.toast(f"Marked {new_status}", icon=":material/check:")
                        st.rerun()
            else:
                st.caption("Select a row to see details and update its status.")
        else:
            st.caption("No findings match these filters.")

# ---------------------------------------------------------------- Early warning
if section == SECTIONS[1]:
    ui.eyebrow("Recall early warning")
    st.title("Complaint spikes", anchor=False)
    st.caption("Owner complaints filed with NHTSA, by component. A spike with **no matching recall** is a "
               "candidate future recall: worth investigating before the regulator does.")
    if wl and wl["config"].get("vehicles"):
        vehicles = wl["config"]["vehicles"]
        labels = [f"{A.vehicle(v['make'], v['model'])} ({min(v['years'])}–{max(v['years'])})" for v in vehicles]
        pick, sens = st.columns([3, 2])
        v = vehicles[pick.selectbox("Vehicle", range(len(vehicles)), format_func=lambda i: labels[i])]
        sensitivity = sens.segmented_control(
            "Sensitivity", ["Standard", "High"], default="Standard",
            help="Standard: components at 1.5x+ their usual complaint rate. High: any component with 3+ "
                 "complaints in 90 days. It catches about twice as many future recalls, with more noise.") or "Standard"
        with st.spinner("Loading NHTSA data..."):
            recs, comps = cached_nhtsa(v["make"], v["model"], tuple(v["years"]))
        spikes = detect_spikes(comps, recs, **({"min_ratio": 0.0} if sensitivity == "High" else {}))
        suspect = recall_data_suspect(comps, recs)
        if suspect:  # don't claim "no recall" when the recall lookup itself probably failed
            for s in spikes:
                s["recalled"] = True
            st.warning("NHTSA returned complaints but **no recalls** for this model name, so recall coverage can't be "
                       "verified. NHTSA's recall lookup fails on some names (e.g. with hyphens). Try the name exactly as "
                       "NHTSA lists it.", icon=":material/help:")
        unrec = [s for s in spikes if not s["recalled"]]
        c = st.columns(4)
        c[0].metric("Complaints (all years)", f"{len(comps):,}", border=True)
        c[1].metric("Recall campaigns", len({r["campaign"] for r in recs}), border=True)
        c[2].metric("Spikes (90 days)", len(spikes), border=True)
        c[3].metric("Spikes with no recall", len(unrec), border=True)

        if unrec:
            top = unrec[0]
            (st.error if top["priority"] == "Priority" else st.warning)(
                f"**{top['priority']}: {top['component'].title()}**: {top['recent']} complaints in 90 days, "
                f"×{top['ratio']} the expected rate, and **no recall** covers this component.",
                icon=":material/warning:")

        chart = ui.anomaly_chart(monthly_series(comps), spikes)
        if chart is not None:
            with st.container(border=True):
                # A fresh element per vehicle/sensitivity: updating a layered chart in place leaves it blank
                st.altair_chart(chart, use_container_width=True, theme=None,
                                key=f"ew-{v['make']}-{v['model']}-{sensitivity}")

        st.subheader("Spike table", anchor=False)
        if spikes:
            st.dataframe(pd.DataFrame([{
                "Priority": {"Priority": "▲ Priority", "Watch": "■ Watch", "Low": "▽ Low"}[s["priority"]],
                "Component": s["component"].title(), "Last 90 days": s["recent"],
                "Baseline / 90 days": s["baseline"], "× Baseline": s["ratio"],
                "Recall": "Unverified" if suspect else ("Covered" if s["recalled"] else "▲ None"),
                "Crashes": s["severity"]["crash"], "Fires": s["severity"]["fire"],
                "Injuries": s["severity"]["injuries"]} for s in spikes]), hide_index=True, width="stretch")
            st.caption("**How far to trust a tier** (backtest on 8 held-out vehicles, 100 past recalls): "
                       "*Priority* alerts were followed by a recall on that component within 12 months 22% of the "
                       "time, *Watch* 8%, *Low* 4%. Treat alerts as leads to investigate, not predictions.")
        else:
            st.success("No abnormal complaint spikes in the last 90 days.", icon=":material/check_circle:")

        stored = [s for s in db.latest_signals(wl_id) if s["analysis"] and s["vehicle"] == f"{v['make']} {v['model']}"]
        for s in stored:
            with st.expander(f"Nemotron analysis: {s['component'].title()} (×{s['ratio']})",
                             icon=":material/psychology:", expanded=True):
                st.markdown(A(s["analysis"]))
        if not stored:
            st.caption("Run the radar with *Early warning* enabled to get Nemotron's failure-mode analysis of each spike.")
    st.caption("Source: NHTSA (US market). Shared global platforms often share failure modes.")

# ---------------------------------------------------------------- Fleet check
if section == SECTIONS[2]:
    ui.eyebrow("Fleet exposure")
    st.title("Fleet recall check", anchor=False)
    st.caption("Upload a CSV with a `VIN` column (fleet, dealer stock, customer base) or paste VINs. "
               "Every VIN is decoded and matched to official recalls.")
    c1, c2 = st.columns(2)
    up = c1.file_uploader("CSV with a VIN column", type=["csv"])
    pasted = c2.text_area("Or paste VINs, one per line", height=120)
    raw = []
    if up is not None:
        frame = pd.read_csv(up, dtype=str)
        col = next((c for c in frame.columns if c.strip().lower() == "vin"), frame.columns[0])
        raw += frame[col].dropna().tolist()
    raw += pasted.splitlines()
    vins, bad = fleet.clean_vins(raw)
    st.caption(f"{len(vins)} valid VINs" + (f" · {len(bad)} invalid skipped" if bad else ""))
    if st.button("Check fleet", icon=":material/search:", type="primary", disabled=not vins):
        if len(vins) > 2000:
            st.warning("Checking the first 2,000 VINs.", icon=":material/warning:")
            vins = vins[:2000]
        with st.status("Checking...", expanded=False) as status:
            rows = fleet.check_fleet(vins, progress=lambda m: st.write(A(m)))
            status.update(label="Done", state="complete")
        df = pd.DataFrame(rows)
        df["Do-not-drive / park outside"] = df["Do-not-drive / park outside"].map({True: "▲ Yes", False: "No"})
        m = st.columns(3)
        m[0].metric("Vehicles", len(df), border=True)
        m[1].metric("With ≥ 1 recall", int((df["Recalls"] > 0).sum()), border=True)
        m[2].metric("Do-not-drive / park outside", int((df["Do-not-drive / park outside"] != "No").sum()), border=True)
        if anon:  # makes/models outside the watchlist still get neutral aliases
            df["Make"] = df["Make"].map(A.make)
            df["Model"] = df["Model"].map(lambda m: A(m) if A(m) != m else "Model")
        df = A.frame(df)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Export results (CSV)", df.to_csv(index=False), icon=":material/download:",
                           file_name="fleet-recall-exposure.csv")
        st.caption("Matches recalls by make, model and year (NHTSA, US). Confirm per-VIN remedy status on the OEM portal.")

# ---------------------------------------------------------------- Analyst
if section == SECTIONS[3]:
    ui.eyebrow("Agent")
    st.title("Ask the analyst", anchor=False)
    st.caption("A Nemotron agent that plans, calls tools (tracked findings, NHTSA recalls and complaints, "
               "live web, VIN decoder) and answers with citations.")
    st.session_state.setdefault("analyst_chat", [])
    if not st.session_state.analyst_chat:
        st.caption(A("Try: *Are there open brake issues for the Bolt EV, and what EU ADAS rules change in 2027?*"))
    elif st.button("Clear conversation", icon=":material/delete_sweep:"):
        st.session_state.analyst_chat = []
        st.rerun()
    for msg in st.session_state.analyst_chat:
        with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/radar:"):
            st.markdown(A(msg["content"]))
    q = st.chat_input("Ask about recalls, regulations, suppliers or a VIN", disabled=not (ai_ready and wl_id))
    if q:
        asked = sum(m["role"] == "user" for m in st.session_state.analyst_chat)
        try:
            guard.allow("question", session_count=asked)
            guard.record("question")
        except (LimitReached, BudgetExceeded) as e:
            st.warning(str(e), icon=":material/savings:")
            q = None
    if q:
        st.session_state.analyst_chat.append({"role": "user", "content": q})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(A(q))
        with st.chat_message("assistant", avatar=":material/radar:"):
            with st.status("Researching...", expanded=True) as status:
                def show(s):
                    if s.get("final"):
                        st.write(f"Step {s['step']}: answer ready")
                    else:
                        st.write(A(f"Step {s['step']}: `{s['tool']}` {s['args']}"))
                try:
                    ans = analyst.ask(db, make_router(), settings, wl_id, q,
                                      st.session_state.analyst_chat[:-1], on_step=show)
                    status.update(label="Done", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Failed", state="error")
                    ans = f"Error: {e}"
            st.markdown(A(ans))
        st.session_state.analyst_chat.append({"role": "assistant", "content": ans})

# ---------------------------------------------------------------- Watchlists
if section == SECTIONS[4]:
    ui.eyebrow("Configuration")
    st.title("Watchlists", anchor=False)
    if anon and wl:
        st.info("Editing is hidden while brands are anonymized.", icon=":material/visibility_off:")
        st.dataframe(pd.DataFrame([{"Vehicle": A.vehicle(v["make"], v["model"]),
                                    "Years": f"{min(v['years'])}–{max(v['years'])}"} for v in wl["config"]["vehicles"]]),
                     hide_index=True, width="stretch")
        st.caption(A(f"Markets: {', '.join(wl['config'].get('markets', []))} · Components: "
                     f"{wl['config'].get('components', '')} · Suppliers: {wl['config'].get('suppliers', '')}"))
        st.stop()
    mode = st.segmented_control("Mode", ["Edit selected", "Create new"], default="Edit selected" if wl else "Create new",
                                label_visibility="collapsed") or "Edit selected"
    editing = bool(wl and mode == "Edit selected")
    base = (wl["config"] if editing else
            {"role": "", "vehicles": [], "markets": ["EU"], "components": "", "suppliers": "",
             "topics": TOPICS + ["Early warning"], "time_range": "month"})

    def years_str(ys):
        return f"{min(ys)}-{max(ys)}" if ys and list(ys) == list(range(min(ys), max(ys) + 1)) else ",".join(map(str, ys))

    def parse_years(s):
        s = str(s).strip()
        if "-" in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in s.replace(" ", "").split(",") if x]

    with st.form("wl_form", border=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", wl["name"] if editing else "")
        role = c2.text_input("Team / role", base.get("role", ""))
        st.markdown("**Vehicles**: make, model, years such as `2021-2024`")
        vdf = st.data_editor(pd.DataFrame([{"make": v["make"], "model": v["model"], "years": years_str(v["years"])}
                                           for v in base.get("vehicles", [])] or
                                          [{"make": "", "model": "", "years": ""}]),
                             num_rows="dynamic", width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        markets = c1.multiselect("Markets", ["EU", "UK", "US", "Canada", "Japan", "Australia", "Global"],
                                 default=base.get("markets", []))
        topics = c2.multiselect("Monitors", TOPICS + ["Early warning"], default=base.get("topics", []))
        components = st.text_input("Components to watch", base.get("components", ""))
        suppliers = st.text_input("Suppliers to watch", base.get("suppliers", ""))
        time_range = st.selectbox("Web look-back per run", ["day", "week", "month", "year"],
                                  index=["day", "week", "month", "year"].index(base.get("time_range", "month")))
        saved = st.form_submit_button("Save watchlist", icon=":material/save:", type="primary")
    if saved:
        try:
            vehicles = [{"make": str(r["make"]).strip(), "model": str(r["model"]).strip(), "years": parse_years(r["years"])}
                        for _, r in vdf.iterrows() if str(r["make"]).strip() and str(r["model"]).strip()]
            assert name.strip(), "Name is required"
            cfg = {"role": role, "vehicles": vehicles, "markets": markets, "components": components,
                   "suppliers": suppliers, "topics": topics, "time_range": time_range}
            db.save_watchlist(name.strip(), cfg, wl_id if editing else None)
            st.toast("Watchlist saved", icon=":material/check:")
            st.rerun()
        except Exception as e:
            st.error(f"Could not save: {e}", icon=":material/error:")
    if editing and st.button("Delete this watchlist", icon=":material/delete:"):
        db.delete_watchlist(wl_id)
        st.rerun()

# ---------------------------------------------------------------- Runs & audit
if section == SECTIONS[5]:
    ui.eyebrow("Governance")
    st.title("Runs & audit trail", anchor=False)
    st.caption("Every run and every Nemotron call is logged with model, latency and tokens, for cost control and auditability.")
    if wl:
        runs = db.runs(wl_id)
        if runs:
            runs_df = pd.DataFrame([{"Run": r["id"], "Started": r["started_at"], "Status": r["status"],
                                     **r["stats"], "Error": A(r["error"])} for r in runs])
            st.dataframe(runs_df, hide_index=True, width="stretch")
            rid = st.selectbox("Inspect run", [r["id"] for r in runs])
            calls = pd.DataFrame(db.llm_calls(rid))
            if len(calls):
                calls["tokens"] = calls["prompt_tokens"].fillna(0) + calls["completion_tokens"].fillna(0)
                calls["cost_usd"] = [round(call_cost(m, p, c), 5) for m, p, c in
                                     zip(calls["model"], calls["prompt_tokens"], calls["completion_tokens"])]
                st.markdown(f"**Token usage by model**: estimated cost of this run **${calls['cost_usd'].sum():.3f}**")
                st.dataframe(calls.groupby("model").agg(calls=("step", "count"), tokens=("tokens", "sum"),
                                                        cost_usd=("cost_usd", "sum"),
                                                        seconds=("seconds", "sum")).reset_index(),
                             hide_index=True, width="stretch")
                st.dataframe(calls, hide_index=True, width="stretch")
                st.download_button("Export audit log (CSV)", calls.to_csv(index=False), icon=":material/download:",
                                   file_name=f"audit-run{rid}.csv")
            brief = next(r["brief"] for r in runs if r["id"] == rid)
            if brief:
                with st.expander("Brief from this run", icon=":material/summarize:"):
                    st.markdown(A(brief))
        else:
            st.caption("No runs yet.")
