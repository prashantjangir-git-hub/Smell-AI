# Prompt: build "AI that can smell"

Paste everything below into Claude Code (or any coding agent), in an empty folder.

---

Build a Python 3.10+ project called `smell-ai`. It joins a **smell model** that predicts what a molecule smells like to a **simulated electronic nose** that tells smells apart, inside an interactive Streamlit app. It must run on a normal laptop in a few minutes, with no GPU.

**Dependencies:** numpy, pandas, scipy, scikit-learn, joblib, rdkit, streamlit, altair. Put pyserial in a separate `requirements-hardware.txt`.

**1. `smell_model.py`**
- Download the OpenPOM curated GoodScents + Leffingwell dataset (`curated_GS_LF_merged_4983.csv`, 4,983 molecules, 138 smell-word columns of 0/1) from the OpenPOM GitHub repo into `data/`, and accept `--csv` for a local copy. Detect the SMILES column and the 0/1 label columns automatically.
- Turn each molecule into an RDKit Morgan count fingerprint (radius 2, 2048 bits, log1p) and store it as a sparse matrix.
- Model: one-vs-rest logistic regression (liblinear, class_weight balanced). Tune C over (0.03, 0.1, 0.3) on a validation split, report the macro ROC-AUC on a 20 % held-out test set, then refit on all molecules. Add `--fast` to skip tuning.
- Build a 2-D "odor map": PCA of the logit of each molecule's predicted probabilities. Colour it by a smell family (citrus, floral, smoky & roasted ...) taken from the molecule's labels.
- Provide a `SmellModel` class with `scores({smiles: weight})`, `describe(...)` (top 5 words) and `map_position(...)`, plus `draw_svg(smiles)`. Add `--predict SMILES`.

**2. `samples.json`**
Eight smells: coffee, vanilla candle, mint, lemon, banana, wood smoke, rose and garlic. Each has an emoji, a description and its real key aroma molecules, given as name, SMILES and relative weight.

**3. `molecules.json`**
About 35 famous odorants (name → SMILES), for example geosmin, vanillin, skatole and cis-3-hexenol.

**4. `sensor.py`**
A simulated array of 8 metal-oxide-style gas sensors that needs no RDKit: a tiny regex SMILES parser counts atoms.
- Each molecule's descriptors: size, oil-loving-ness, O/N/S fractions, aromatic fraction, unsaturation, C=O groups.
- Each channel gives softplus(W · descriptors) × volatility. A sample's signal is the weighted sum of its molecules' responses, saturating as 1 − exp(−k·signal).
- Add per-channel humidity cross-sensitivity, gain drift, and additive and multiplicative noise.
- Provide `sniff(comp, humidity, intensity, noise)`, `trace(...)` (rise and decay over time), `mix(...)`, and `to_features(reading)` (raw values plus the L2-normalised pattern).

**5. `train_nose.py`**
- Sniff every sample plus "clean air" 200 times with random humidity (20-80 %), strength (0.5-1.5) and noise. Train a 300-tree random forest. Test on fresh sniffs from a new seed and print the accuracy and a confusion matrix. Aim for about 98-99 %.
- `--real DIR` trains on BME688 CSV recordings and tests leave-one-session-out.

**6. `app.py` (Streamlit, wide layout)**
- Sidebar: sliders for humidity, strength and noise, and metrics for both models.
- Tabs:
  - **Sniff:** an animated live sensor trace, a channel bar chart, the nose's top-3 guesses, and the smell model's words for the guessed sample. An expander shows the sample's molecules as RDKit drawings.
  - **Mix lab:** blend two smells, show the nose's probabilities, and flag when it's fooled.
  - **Molecule lab:** a library pick or any SMILES. Show the drawing, the top smell words chart, the simulated sensor pattern, and nearest neighbours on the odor map.
  - **Odor map:** an interactive Altair scatter with the samples and the last lab molecule marked.
  - **Guess game:** you vs the AI from the sensor readings only, with a score and a cheat sheet.
  - **Real sensor:** upload a recording CSV and classify it with the real nose.
- If the nose model is missing, train it automatically. If the smell model is missing, show the command to run.

**7. Hardware**
- `firmware/esp32_bme688/esp32_bme688.ino`: the Adafruit BME680 library on an ESP32 over I2C (SDA 21, SCL 22). Step the heater through 200, 250, 300 and 350 °C and print `BME,ms,temp,hum,r200,r250,r300,r350`.
- `record_bme688.py`: read that serial output. Record a clean-air baseline, prompt the user to put the sensor in the jar, record, divide by the baseline, and save `data/real/<session>_<label>.csv`. Add `--simulate` to make fake data from `sensor.py`.

**8. Tests, docs and git**
- `test_app.py`: tests the sensor, the nose accuracy and the smell model, and runs the app headlessly with `streamlit.testing.v1.AppTest`, clicking Sniff, Mix and the game. It should run with plain `python` as well as with pytest.
- `README.md`: run steps, how it works, the real-sensor guide and GitHub upload steps.
- `.gitignore`: ignore `data/*.csv`, `models/` and `.venv/`.
- `LICENSE`: MIT.

Credit Lee et al., Science 2023, and OpenPOM.
