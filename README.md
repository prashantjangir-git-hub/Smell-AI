# 👃 AI that can smell

A **smell model** that reads a molecule and predicts what it smells like, joined to an **electronic nose** that tells smells apart, wrapped in an interactive app.

It runs on any laptop in a few minutes, with no GPU and no hardware. It's built on **OpenPOM**, the open-source rebuild of Google's smell research, and its dataset of 4,983 molecules labelled by perfume experts.

> **Note:** the nose is **simulated** in code so you can see the whole pipeline working. [Part 3](#part-3-give-it-a-real-nose) shows how to swap in a real BME688 gas sensor (about ₹1,400).

## What you can do in the app

| Tab | What happens |
| --- | --- |
| 👃 **Sniff** | Pick a smell, press **Sniff**, and watch 8 sensor channels react live. The nose names the smell with a confidence score, then the smell model describes it in words. |
| 🧪 **Mix lab** | Blend two smells (say 70 % coffee + 30 % vanilla) and try to fool a nose that only learned pure smells. |
| 🔬 **Molecule lab** | Type **any** molecule (as SMILES) or pick one of 35 famous ones (geosmin = the smell of rain, skatole, vanillin ...). See its structure, its predicted smell words, what the sensors would see, and its nearest neighbours. |
| 🗺️ **Odor map** | All 4,983 molecules on a zoomable 2-D map where similar smells sit together, a simple version of Google's *Principal Odor Map*. |
| 🎮 **Guess game** | You vs the AI: read the mystery sensor pattern and name the smell. The game keeps score. |
| 📡 **Real sensor** | Drop in a recording from your own BME688 and the trained real nose classifies it. |

In the sidebar you can change **humidity, smell strength and sensor noise**. Push them past what the nose was trained on and watch its confidence fall.

## Part 1: run it

You need **Python 3.10 or newer**.

```bash
cd smell-ai
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smell_model.py              # downloads the dataset, trains the smell model (1-3 min the first time)
python train_nose.py               # trains the nose (a few seconds)
streamlit run app.py               # opens the app in your browser
```

To check everything works without opening the browser:

```bash
python test_app.py
```

Try a single molecule from the terminal:

```bash
python smell_model.py --predict "COc1cc(C=O)ccc1O"     # vanillin
```

## Part 2: how it works

| File | What it does |
| --- | --- |
| `smell_model.py` | Turns each molecule into a Morgan fingerprint (counts of its small chemical fragments). For each of the 138 smell words (fruity, minty, smoky ...) a logistic regression learns which fragments go with that word. It also builds the 2-D odor map. |
| `samples.json` | Eight real smells (coffee, vanilla candle, mint, lemon, banana, wood smoke, plus rose and garlic). Each is a list of the real aroma molecules in it. |
| `molecules.json` | 35 famous smelly molecules for the Molecule lab. |
| `sensor.py` | A simulated array of 8 gas sensors. Each reacts differently to a molecule's size, oil-loving-ness, oxygen, sulfur, aromatic rings and C=O groups. It adds humidity, drift and noise, and makes a live time trace. |
| `train_nose.py` | Sniffs every smell (plus clean air) 200 times with random noise, humidity and strength, then trains a random forest to name the smell. With `--real` it trains on BME688 recordings instead. |
| `app.py` | The app you play with. |
| `record_bme688.py` | Records real sensor data from an ESP32 into CSV files. |
| `firmware/esp32_bme688/` | The Arduino sketch for the ESP32 + BME688. |
| `test_app.py` | Tests for everything, including a headless run of the app. |

**Why two models?** The smell model reads molecule structures, but a real gas sensor only gives electrical readings. So the nose names the sample first, and the sample's list of known molecules then tells the smell model what to describe.

**What to expect** (your numbers will differ a little):

- Smell model: about **0.8 ROC-AUC** on molecules it never saw (0.5 is guessing, 1 is perfect). The script prints the exact number.
- Nose: about **98-99 %** on new simulated sniffs. Its few mistakes are between similar smells, like wood smoke and vanilla, which are both sweet and phenolic.

That nose score is high because the nose is tested on the same simulator it learned from. A real sensor is harder, which is what Part 3 is for.

## Part 3: give it a real nose

A **BME688** gas sensor (around ₹1,400) can step its heater through different temperatures, so one chip gives you several readings per cycle.

1. **Wire** the BME688 to an ESP32 over I2C: VIN → 3V3, GND → GND, SCL → GPIO 22, SDA → GPIO 21.
2. **Flash** `firmware/esp32_bme688/esp32_bme688.ino` with the Arduino IDE. You'll need the *Adafruit BME680* and *Adafruit Unified Sensor* libraries and the *ESP32 Dev Module* board.
3. **Burn it in:** leave it powered for 24 hours before you trust any reading.
4. **Record** each smell in its own jar for 30 minutes, with clean air in between. Do at least 3 sessions on different days:
   ```bash
   pip install -r requirements-hardware.txt
   python record_bme688.py --list-ports
   python record_bme688.py --port COM3 --label "clean air" --session day1
   python record_bme688.py --port COM3 --label coffee --session day1
   ```
   Each recording starts with 2 minutes of clean air, and every reading is divided by that clean-air value.
5. **Train and test** on a session it never saw:
   ```bash
   python train_nose.py --real data/real
   ```
   Then open the 📡 tab in the app and upload a recording.

No sensor yet? Try the whole workflow with fake data first:

```bash
python record_bme688.py --simulate --label coffee --session day1 --minutes 1 --baseline 10
```

Two traps to avoid:

- Hot coffee mostly teaches a sensor to detect steam, so use room-temperature grounds.
- Very similar smells are hard to tell apart, so start with very different ones (coffee, mint, garlic, clean air).

## Part 4: go further

- For a stronger smell model, swap in **OpenPOM** itself, a graph neural network that reached 0.89 ROC-AUC in its authors' tests. It needs Python 3.10, PyTorch and DeepChem, and a GPU helps (Google Colab works).
- Add your own smells to `samples.json`: a name, an emoji and a list of molecules with SMILES and amounts. Then re-run `python train_nose.py`.
- Train the nose on mixtures so the Mix lab can't fool it.

## Put it on GitHub

1. On github.com, signed in to the account you want, click **New repository** and name it `smell-ai`. Leave *Add a README* and *Add .gitignore* **unticked**, because this project already has them.
2. In the project folder:
   ```bash
   git init
   git config user.name "Your Name"
   git config user.email "you@example.com"     # the email of the GitHub account you're pushing to
   git add .
   git commit -m "AI that can smell"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/smell-ai.git
   git push -u origin main
   ```
3. When git asks for a password, paste a **personal access token** (GitHub → Settings → Developer settings → Personal access tokens). If it pushes as the wrong account, clear the saved GitHub login in your system's credential manager and push again.

The dataset and trained models are listed in `.gitignore`. The scripts download and rebuild them, so the repo stays small.

Before you push, put your name in `LICENSE`.

## Troubleshooting

- **`pip install rdkit` fails:** use Python 3.10-3.13 and upgrade pip (`python -m pip install --upgrade pip`).
- **The dataset download fails:** download `curated_GS_LF_merged_4983.csv` from the OpenPOM GitHub repo by hand, then run `python smell_model.py --csv path/to/file.csv`.
- **Windows says scripts are disabled when you activate the venv:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell once.
- **Training is too slow:** run `python smell_model.py --fast`.

## Credits

- Research: Lee et al., *A principal odor map unifies diverse tasks in olfactory perception*, Science (2023), from Google Research, Osmo and the Monell Center.
- Model rebuild and dataset: [OpenPOM](https://github.com/BioMachineLearning/openpom) (MIT licence), built on the GoodScents and Leffingwell odour data.
- Molecule drawings: RDKit.

The dataset is not included in this repo. It is downloaded from OpenPOM when you first run the scripts.
