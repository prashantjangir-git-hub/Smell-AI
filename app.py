"""
AI that can smell - the interactive app.

    streamlit run app.py

Tabs
  👃 Sniff          pick a smell, watch the 8 sensors react, see the nose's guess
                    and the smell model's description
  🧪 Mix lab        blend two smells and try to confuse the nose
  🔬 Molecule lab   type any molecule and see what the AI thinks it smells like
  🗺️ Odor map       all 4,983 molecules laid out by how they smell
  🎮 Guess game     you vs the AI: name the mystery smell from sensor readings
  📡 Real sensor    classify recordings from a real BME688 nose
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import sensor
import train_nose
from smell_model import FAMILIES, MODEL_PATH, OTHER_FAMILY, SmellModel, draw_svg, family_of

ROOT = Path(__file__).resolve().parent
alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="AI that can smell", page_icon="👃", layout="wide")

CHANNEL_COLORS = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#9d755d"]
FAMILY_NAMES = list(FAMILIES) + [OTHER_FAMILY]
FAMILY_COLORS = ["#8c6d31", "#ad494a", "#5b3a29", "#17becf", "#f2c80f", "#e377c2",
                 "#ff7f0e", "#d62728", "#2ca02c", "#98df8a", "#c49c94", "#7f7f7f"]
EXPOSURE = (5.0, 25.0)
TRACE_SECONDS = 40.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@st.cache_resource
def get_smell_model():
    return SmellModel.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_resource
def get_nose():
    if not train_nose.SIM_PATH.exists():
        bundle = train_nose.train_simulated(verbose=False)
        train_nose.MODEL_DIR.mkdir(exist_ok=True)
        import joblib

        joblib.dump(bundle, train_nose.SIM_PATH, compress=3)
    return train_nose.load_nose(train_nose.SIM_PATH)


@st.cache_resource
def get_real_nose():
    return train_nose.load_nose(train_nose.REAL_PATH) if train_nose.REAL_PATH.exists() else None


@st.cache_data
def get_samples():
    return sensor.load_samples()


@st.cache_data
def get_library():
    with open(ROOT / "molecules.json", encoding="utf-8") as f:
        return json.load(f)


sm = get_smell_model()
if sm is None:
    st.title("👃 AI that can smell")
    st.error("The smell model isn't trained yet. In your terminal run:\n\n`python smell_model.py`\n\nthen refresh this page.")
    st.stop()

with st.spinner("Warming up the nose ..."):
    nose = get_nose()
real_nose = get_real_nose()
samples = get_samples()
library = get_library()
SMELLS = list(samples)
ALL_CLASSES = SMELLS + [sensor.CLEAN_AIR]


def label(name: str) -> str:
    if name == sensor.CLEAN_AIR:
        return "🌬️ clean air"
    return f"{samples[name]['emoji']} {name}"


def comp_of(name: str) -> dict[str, float]:
    return {} if name == sensor.CLEAN_AIR else sensor.components(samples[name])


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)


def channel_chart(reading, height: int = 230, title: str | None = None):
    df = pd.DataFrame({"channel": sensor.CHANNELS, "response": np.asarray(reading, dtype=float)})
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("channel:N", sort=None, title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("response:Q", scale=alt.Scale(domain=[-0.1, 1.0]), title="sensor response"),
            color=alt.Color("channel:N", sort=None, legend=None, scale=alt.Scale(range=CHANNEL_COLORS)),
            tooltip=["channel", alt.Tooltip("response:Q", format=".3f")],
        )
        .properties(height=height)
    )
    return chart.properties(title=title) if title else chart


def trace_chart(times, readings, upto: int | None = None, height: int = 280):
    upto = len(times) if upto is None else upto
    df = pd.DataFrame(readings[:upto], columns=sensor.CHANNELS)
    df["time (s)"] = times[:upto]
    long = df.melt("time (s)", var_name="channel", value_name="response")
    window = alt.Chart(pd.DataFrame({"start": [EXPOSURE[0]], "end": [EXPOSURE[1]], "txt": ["smell in"]}))
    shade = window.mark_rect(opacity=0.10, color="#f58518").encode(x="start:Q", x2="end:Q")
    txt = window.mark_text(align="left", dx=4, dy=8, color="#f58518").encode(x="start:Q", y=alt.value(0), text="txt:N")
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time (s):Q", scale=alt.Scale(domain=[0, float(times[-1])])),
            y=alt.Y("response:Q", scale=alt.Scale(domain=[-0.1, 1.0]), title="sensor response"),
            color=alt.Color("channel:N", sort=None, scale=alt.Scale(domain=sensor.CHANNELS, range=CHANNEL_COLORS),
                            legend=alt.Legend(orient="bottom", columns=4, title=None)),
        )
    )
    return alt.layer(shade, txt, lines).properties(height=height)


def words_chart(scores: pd.Series, n: int = 8, height: int | None = None):
    scores = scores[scores.index != "odorless"].head(n)
    df = pd.DataFrame({"smell": scores.index, "score": scores.values})
    df["family"] = [family_of([w]) for w in df["smell"]]
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("smell:N", sort="-x", title=None),
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="how strongly the model thinks so"),
            color=alt.Color("family:N", scale=alt.Scale(domain=FAMILY_NAMES, range=FAMILY_COLORS),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["smell", "family", alt.Tooltip("score:Q", format=".2f")],
        )
        .properties(height=height or 28 * len(df) + 40)
    )


def probs_chart(probs: dict[str, float]):
    df = pd.DataFrame({"smell": [label(k) for k in probs], "confidence": list(probs.values())})
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("smell:N", sort="-x", title=None),
            x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.value("#4c78a8"),
            tooltip=["smell", alt.Tooltip("confidence:Q", format=".1%")],
        )
        .properties(height=28 * len(df) + 30)
    )


def svg_tag(smiles: str, width: int = 160) -> str:
    svg = draw_svg(smiles)
    if not svg:
        return "<div style='height:120px'>?</div>"
    b64 = base64.b64encode(svg.encode()).decode()
    return f"<img src='data:image/svg+xml;base64,{b64}' width='{width}' style='background:#fff;border-radius:8px'/>"


def molecule_gallery(sample: dict):
    cards = []
    for m in sample["molecules"]:
        cards.append(
            "<div style='text-align:center;width:170px'>"
            f"{svg_tag(m['smiles'])}<br><small><b>{m['name']}</b><br>amount {m['weight']:.2f}</small></div>"
        )
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:14px'>{''.join(cards)}</div>", unsafe_allow_html=True)


def plateau(times, readings) -> np.ndarray:
    """What the nose looks at: the average of the last 5 s with the smell in."""
    mask = (times >= EXPOSURE[1] - 5) & (times < EXPOSURE[1])
    return readings[mask].mean(axis=0)


def word_list(words) -> str:
    return " · ".join(f"**{w}**" for w in words) if words else "_nothing: it's clean air_"


# ---------------------------------------------------------------------------
# Sidebar: conditions
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🌡️ Sniffing conditions")
    humidity = st.slider("Humidity (%)", 5, 95, 50, help="The nose was trained on 20-80 %. Go outside that and watch it struggle.") / 100
    strength = st.slider("Smell strength", 0.2, 2.5, 1.0, 0.1, help="Trained on 0.5-1.5. A faint whiff is much harder.")
    noise = st.slider("Sensor noise", 0.0, 4.0, 1.0, 0.25, help="1 = realistic. Crank it up to see the nose fail.")
    st.divider()
    st.header("📊 Models")
    st.metric("Smell model ROC-AUC", f"{sm.test_auc:.2f}", help="On molecules it never saw. 0.5 = guessing, 1 = perfect.")
    st.metric("Nose accuracy (simulated)", f"{nose['test_accuracy']:.0%}", help="On new simulated sniffs.")
    if real_nose:
        acc = real_nose.get("test_accuracy")
        st.metric("Real nose accuracy", f"{acc:.0%}" if acc is not None else "n/a")
    st.caption(f"Smell model knows {len(sm.labels)} smell words from {len(sm.odor_map):,} molecules.")
    st.caption("The nose here is **simulated**. See the 📡 tab and README Part 3 for a real sensor.")

st.title("👃 AI that can smell")
st.caption("A smell model that reads molecules + an electronic nose that tells smells apart. Built on OpenPOM data.")

tabs = st.tabs(["👃 Sniff", "🧪 Mix lab", "🔬 Molecule lab", "🗺️ Odor map", "🎮 Guess game", "📡 Real sensor"])

# ---------------------------------------------------------------------------
# Tab 1: Sniff
# ---------------------------------------------------------------------------

with tabs[0]:
    left, right = st.columns([1, 2])
    with left:
        choice = st.selectbox("What's in front of the nose?", ALL_CLASSES, format_func=label, key="sniff_choice")
        if choice != sensor.CLEAN_AIR:
            st.caption(samples[choice]["description"])
        animate = st.checkbox("Animate the sensors", value=True, key="animate")
        go = st.button("👃 Sniff", type="primary", key="sniff")

    if go:
        rng = np.random.default_rng()
        times, readings = sensor.trace(comp_of(choice), TRACE_SECONDS, EXPOSURE, 0.5, humidity, strength, noise, rng)
        if animate:
            with right:
                ph = st.empty()
                for i in range(2, len(times) + 1, 3):
                    ph.altair_chart(trace_chart(times, readings, i))
                    time.sleep(0.03)
                ph.empty()
        reading = plateau(times, readings)
        st.session_state["sniff_result"] = {
            "truth": choice, "times": times, "readings": readings, "reading": reading,
            "probs": train_nose.predict(nose, reading),
        }

    res = st.session_state.get("sniff_result")
    if res is None:
        with right:
            st.info("Pick a smell and press **Sniff**.")
    else:
        with right:
            show_chart(trace_chart(res["times"], res["readings"]))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1 · Sensors")
            show_chart(channel_chart(res["reading"]))
        with c2:
            st.subheader("2 · Nose's guess")
            guess, conf = next(iter(res["probs"].items()))
            ok = guess == res["truth"]
            (st.success if ok else st.error)(f"{label(guess)} - {conf:.0%} sure" + ("" if ok else f"  (it was {label(res['truth'])})"))
            top3 = list(res["probs"].items())[:3]
            for name, p in top3:
                st.progress(float(p), text=f"{label(name)}  {p:.0%}")
        with c3:
            st.subheader("3 · Smell model says")
            comp = comp_of(guess)
            st.markdown(word_list(sm.describe(comp)))
            if comp:
                show_chart(words_chart(sm.scores(comp), 8))
        if res["truth"] != sensor.CLEAN_AIR:
            with st.expander(f"🔬 The molecules in {label(res['truth'])}"):
                molecule_gallery(samples[res["truth"]])
        st.caption(
            "How it fits together: a real gas sensor only gives electrical readings, so the nose names the sample first. "
            "Then the sample's list of known molecules tells the smell model what to describe."
        )

# ---------------------------------------------------------------------------
# Tab 2: Mix lab
# ---------------------------------------------------------------------------

with tabs[1]:
    st.markdown("Blend two smells. The nose only ever learned **pure** smells, so can you find a mix that fools it?")
    a, b, c = st.columns([2, 2, 3])
    with a:
        smell_a = st.selectbox("Smell A", SMELLS, index=0, format_func=label, key="mix_a")
    with b:
        smell_b = st.selectbox("Smell B", SMELLS, index=SMELLS.index("vanilla candle") if "vanilla candle" in SMELLS else 1,
                               format_func=label, key="mix_b")
    with c:
        frac_b = st.slider("How much of B", 0, 100, 50, 5, format="%d%%", key="mix_frac") / 100
    mix_comp = sensor.mix([(comp_of(smell_a), 1 - frac_b), (comp_of(smell_b), frac_b)])
    if st.button("🧪 Sniff the mix", type="primary", key="mix_go"):
        reading = sensor.sniff(mix_comp, humidity, strength, noise)
        st.session_state["mix_result"] = {"reading": reading, "probs": train_nose.predict(nose, reading),
                                   "a": smell_a, "b": smell_b, "frac": frac_b}
    mres = st.session_state.get("mix_result")
    if mres:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.subheader("Sensors")
            show_chart(channel_chart(mres["reading"]))
        with m2:
            st.subheader("Nose's opinion")
            show_chart(probs_chart(mres["probs"]))
            top = next(iter(mres["probs"]))
            if top not in (mres["a"], mres["b"]):
                st.warning(f"Fooled! It thinks it's {label(top)}.")
        with m3:
            st.subheader("If we knew the molecules")
            mcomp = sensor.mix([(comp_of(mres["a"]), 1 - mres["frac"]), (comp_of(mres["b"]), mres["frac"])])
            st.markdown(word_list(sm.describe(mcomp)))
            show_chart(words_chart(sm.scores(mcomp), 8))
        st.caption("A real e-nose has the same problem: it can only name what it was trained on. "
                   "To handle mixtures you'd train it on mixtures too.")

# ---------------------------------------------------------------------------
# Tab 3: Molecule lab
# ---------------------------------------------------------------------------

with tabs[2]:
    st.markdown("Pick a famous molecule or type any **SMILES** (the text way of writing a molecule). "
                "The smell model has never seen most of these as named molecules - it reads the structure.")
    TYPE_OWN = "✏️ type my own SMILES"
    l1, l2 = st.columns([1, 1])
    with l1:
        pick = st.selectbox("Molecule", [TYPE_OWN] + list(library), index=1, key="lab_pick")
        default = library.get(pick, "CC(C)CCOC(C)=O")
        smiles = st.text_input("SMILES", value=default, key=f"lab_smiles_{pick}",
                               help="Examples: CCO (ethanol), O=Cc1ccccc1 (benzaldehyde). Search a molecule on PubChem to find its SMILES.")
    scores = sm.scores({smiles: 1.0}) if smiles.strip() else pd.Series(dtype=float)
    if scores.empty:
        st.error("That isn't a valid molecule. Check the SMILES text.")
    else:
        with l2:
            st.markdown(svg_tag(smiles, 260), unsafe_allow_html=True)
        top_words = sm.describe({smiles: 1.0}, 5)
        st.markdown(f"### Smells like: {word_list(top_words)}")
        st.caption(f"Family: **{family_of(top_words[:3])}**")
        r1, r2 = st.columns(2)
        with r1:
            show_chart(words_chart(scores, 12))
        with r2:
            st.markdown("**How the simulated nose would see it** (noise-free)")
            show_chart(channel_chart(sensor.sniff({smiles: 1.0}, noise=0)))
            pos = sm.map_position({smiles: 1.0})
            if pos:
                st.session_state["lab_point"] = {"x": pos[0], "y": pos[1], "name": pick if pick != TYPE_OWN else smiles}
                om = sm.odor_map
                d = np.hypot(om["x"] - pos[0], om["y"] - pos[1])
                near = om.assign(distance=d).nsmallest(5, "distance")[["smells", "smiles", "family"]]
                st.markdown("**Nearest neighbours on the odor map**")
                st.dataframe(near, hide_index=True)
        st.caption("Tip: open the 🗺️ Odor map tab to see where this molecule lands (the ⭐).")

# ---------------------------------------------------------------------------
# Tab 4: Odor map
# ---------------------------------------------------------------------------

with tabs[3]:
    st.markdown(
        "Every dot is one of the molecules the smell model learned from, placed so that molecules that "
        "**smell alike sit close together**. It's a simple 2-D version of Google's *Principal Odor Map*. "
        "Hover for details, scroll to zoom, drag to pan."
    )
    fams = st.multiselect("Show families", FAMILY_NAMES, default=FAMILY_NAMES, key="map_fams")
    om = sm.odor_map[sm.odor_map["family"].isin(fams)]
    points = (
        alt.Chart(om)
        .mark_circle(size=22, opacity=0.55)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("family:N", scale=alt.Scale(domain=FAMILY_NAMES, range=FAMILY_COLORS),
                            legend=alt.Legend(orient="bottom", columns=4, title=None)),
            tooltip=["smells", "family", "smiles"],
        )
        .interactive()
    )
    marks = []
    for name in SMELLS:
        p = sm.map_position(comp_of(name))
        if p:
            marks.append({"x": p[0], "y": p[1], "icon": samples[name]["emoji"], "name": name})
    lp = st.session_state.get("lab_point")
    if lp:
        marks.append({"x": lp["x"], "y": lp["y"], "icon": "⭐", "name": str(lp["name"])[:28]})
    layers = [points]
    if marks:
        mdf = pd.DataFrame(marks)
        layers.append(alt.Chart(mdf).mark_text(fontSize=24).encode(x="x:Q", y="y:Q", text="icon:N", tooltip=["name"]))
        layers.append(alt.Chart(mdf).mark_text(fontSize=12, fontWeight="bold", dy=20).encode(x="x:Q", y="y:Q", text="name:N"))
    show_chart(alt.layer(*layers).properties(height=620))
    st.caption("Big icons = our 8 samples. ⭐ = the last molecule you tried in the Molecule lab.")

# ---------------------------------------------------------------------------
# Tab 5: Guess game
# ---------------------------------------------------------------------------

with tabs[4]:
    st.markdown("A mystery smell is under the nose. You only get the **sensor readings**. Can you beat the AI?")
    score = st.session_state.setdefault("score", {"you": 0, "ai": 0, "rounds": 0})
    s1, s2, s3 = st.columns(3)
    s1.metric("🧑 You", score["you"])
    s2.metric("🤖 AI", score["ai"])
    s3.metric("Rounds", score["rounds"])

    if st.button("🎲 New mystery smell", type="primary", key="game_new"):
        rng = np.random.default_rng()
        secret = str(rng.choice(SMELLS))
        reading = sensor.sniff(comp_of(secret), rng.uniform(0.3, 0.7), rng.uniform(0.6, 1.3), 1.0, rng)
        st.session_state["game_state"] = {"secret": secret, "reading": reading, "done": False,
                                    "ai": next(iter(train_nose.predict(nose, reading)))}
    game = st.session_state.get("game_state")
    if game:
        g1, g2 = st.columns([3, 2])
        with g1:
            show_chart(channel_chart(game["reading"], title="Mystery reading"))
        with g2:
            your = st.radio("Your guess", SMELLS, format_func=label, key="game_guess", disabled=game["done"])
            if not game["done"] and st.button("🔒 Lock in", key="game_lock"):
                game["done"] = True
                game["yours"] = your
                score["rounds"] += 1
                score["you"] += int(your == game["secret"])
                score["ai"] += int(game["ai"] == game["secret"])
                st.rerun()
            if game["done"]:
                st.markdown(f"It was **{label(game['secret'])}**")
                st.markdown(f"🧑 You said {label(game['yours'])} {'✅' if game['yours'] == game['secret'] else '❌'}")
                st.markdown(f"🤖 AI said {label(game['ai'])} {'✅' if game['ai'] == game['secret'] else '❌'}")
        with st.expander("🃏 Cheat sheet: what each smell usually looks like"):
            cols = st.columns(4)
            for i, name in enumerate(SMELLS):
                with cols[i % 4]:
                    show_chart(channel_chart(sensor.sniff(comp_of(name), noise=0), height=150, title=label(name)))
            st.caption("Hint: S4 (Pt-doped) loves sulfur, S6 (ZnO) loves aromatic rings, S3 (WO3) loves oxygen.")
    if score["rounds"]:
        if st.button("Reset score", key="game_reset"):
            st.session_state["score"] = {"you": 0, "ai": 0, "rounds": 0}
            st.session_state.pop("game_state", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 6: Real sensor
# ---------------------------------------------------------------------------

with tabs[5]:
    st.markdown(
        "Got a **BME688** sensor? Record smells with `record_bme688.py`, train with "
        "`python train_nose.py --real data/real`, then drop a recording here to see what the real nose thinks."
    )
    if real_nose is None:
        st.info("No real nose trained yet. Follow README Part 3: wire the sensor, record a few smells on different "
                "days, then run `python train_nose.py --real data/real` and refresh.")
    else:
        st.success(f"Real nose loaded: knows {', '.join(real_nose['classes'])}.")
        up = st.file_uploader("A recording CSV (from record_bme688.py)", type="csv", key="real_upload")
        if up is not None:
            df = pd.read_csv(up)
            cols = real_nose["feature_columns"]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                st.error(f"This file is missing columns: {', '.join(missing)}")
            else:
                df = df.dropna(subset=cols)
                X = sensor.to_features(df[cols].to_numpy())
                pred = real_nose["model"].predict(X)
                counts = pd.Series(pred).value_counts()
                st.metric("The real nose says", str(counts.index[0]))
                st.caption(f"{counts.iloc[0] / len(pred):.0%} of {len(pred)} readings agree")
                st.bar_chart(counts)
                if "label" in df.columns:
                    st.caption(f"The file's own label says: {df['label'].iloc[0]}")
