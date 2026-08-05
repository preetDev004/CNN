# Self-Driving Car Simulation Using CNN

DPS920 Final Project. A CNN trained to predict steering angle from the
front-camera image, driving the car autonomously in the Udacity self-driving
car simulator.

## Demo Video
[Check It Out Here!](https://youtu.be/gsXG8_7GrOI)

## Approach

The model is Nvidia's PilotNet architecture: 5 convolutional layers feeding
into a 1164 -> 100 -> 50 -> 10 -> 1 fully-connected stack, trained as a
regression problem (MSE loss, no output activation) to predict a single
continuous steering angle from a single center-camera frame.

Pipeline, end to end:

1. **Data collection** - drive the track in the simulator's Training Mode,
   producing `driving_log.csv` (center/left/right image paths, steering,
   throttle, brake, speed) and an `IMG/` folder of frames.
2. **Balancing** - raw driving data is dominated by near-zero steering
   angles (straight-line driving). `src/utils.py` bins the steering
   distribution and caps each bin so the model isn't trained mostly on
   "go straight."
3. **Train/validation split** - 80/20 split on the balanced data.
4. **Augmentation** (training split only) - random pan, zoom, rotation,
   brightness shift, and horizontal flip (which also negates the steering
   label), implemented in `src/augment.py`.
5. **Preprocessing** - crop to the road area, convert to YUV, Gaussian
   blur, resize to 200x66 (Nvidia's input size), normalize to [0, 1].
   Implemented once in `src/data_preprocessing.py` and imported by both
   `train.py` and `TestSimulation.py`, so training and live inference see
   pixels prepared identically.
6. **Training** - `train.py` streams batches through a generator
   (`src/batch.py`), with `ModelCheckpoint` saving only the
   best-validation-loss epoch and `EarlyStopping` stopping once validation
   loss stops improving, so the shipped model isn't just whatever epoch
   happened to run last.
7. **Testing** - `TestSimulation.py` loads the saved model and drives the
   car live in the simulator's Autonomous Mode over a Socket.IO connection.

## Project structure

```
.
├── src/
│   ├── data_preprocessing.py   # crop, YUV, blur, resize, normalize
│   ├── augment.py               # pan, zoom, rotate, brightness, flip
│   ├── batch.py                 # training/validation batch generator
│   ├── model.py                 # Nvidia PilotNet architecture
│   └── utils.py                 # CSV loading, histogram balancing, split
├── train.py                     # training entry point
├── TestSimulation.py            # live inference in the simulator
├── model/
│   └── model.h5                 # trained model
├── training_plots/              # loss curve + steering histograms
├── pyproject.toml
├── requirements.txt
└── data/                        # driving_log.csv + IMG/, not committed
```

## Environment setup

Requires Python 3.13, managed with `uv`.

```bash
uv sync
```

This installs everything pinned in `pyproject.toml`, including
`python-socketio==4.6.1` and `python-engineio==3.13.2`. Those two versions
are load-bearing, not arbitrary: the simulator's embedded Socket.IO client
speaks an older protocol version than current `python-socketio` defaults
speak, and the mismatch causes silent connection failures during testing
(see Challenges below). A pinned `requirements.txt` is also included as a
plain-pip fallback if `uv` isn't available.

## Data

Collect data via the simulator's Training Mode, then place it here:

```
data/
├── IMG/
└── driving_log.csv
```

## Training

```bash
uv run train.py --data-dir data --epochs 25 --samples-per-bin 50
```

Key arguments:

| Argument | Default | Purpose |
|---|---|---|
| `--data-dir` | `data` | Folder with `driving_log.csv` and `IMG/` |
| `--epochs` | 10 | Max epochs (early stopping may end sooner) |
| `--batch-size` | 100 | Samples per batch |
| `--steps-per-epoch` | 300 | Batches drawn per training epoch |
| `--val-steps` | 200 | Batches drawn per validation epoch |
| `--learning-rate` | 1e-3 | Adam learning rate |
| `--samples-per-bin` | 50 | Cap per steering histogram bin when balancing |
| `--output` | `model/model.h5` | Where to save the trained model |

`--samples-per-bin` needs to be tuned to your dataset size, not left at
whatever default: `balance_data()` logs the actual per-bin counts on every
run, use that to pick a sensible cap. Too high and balancing removes
nothing; too low and you throw away most of your data.

Output: `model/model.h5` (best validation-loss epoch) and
`training_plots/loss_curve.png`, `steering_before.png`, `steering_after.png`.

## Testing in the simulator

```bash
uv run TestSimulation.py --model model/model.h5 --port 4567
```

Then launch the simulator, select the same track used for data collection,
and choose Autonomous Mode. A `Connected: <sid>` print and a continuous
stream of `steering=... throttle=... speed=...` lines confirms the
simulator and server are talking; the car should then drive itself.

## Challenges faced and how they were addressed

- **Cross-platform file paths.** `driving_log.csv` stores absolute paths
  from whichever machine recorded the data, sometimes Windows
  (backslash-separated). `os.path.basename` only splits on the current
  OS's separator, so a Windows path passed through unchanged on macOS/Linux
  and the image files couldn't be found. Fixed by stripping both `/` and
  `\` explicitly in `load_log()` regardless of which OS is training.

- **Unmaintained augmentation dependency.** The original augmentation code
  used `imgaug`, which has been unmaintained since 2020 and breaks outright
  on NumPy >= 2.0 (`np.sctypes` was removed). Replaced with equivalent
  pan/zoom/rotate/brightness transforms written directly against OpenCV, no
  external augmentation library required.

- **Steering distribution imbalance.** Raw driving data is overwhelmingly
  straight-line steering, which biases the model toward "always go
  straight" and hurts cornering. Addressed with histogram-based balancing
  before training, with the per-bin counts logged so the cap can be tuned
  to the actual dataset size instead of guessed.

- **Model checkpointing.** Training loss doesn't always improve
  monotonically to the last epoch; in early runs the final epoch's model
  was measurably worse (higher validation loss) than an earlier epoch.
  Fixed by adding `ModelCheckpoint(save_best_only=True)` and
  `EarlyStopping(restore_best_weights=True)`, so the model actually shipped
  is the best-validated one, not just the last one trained.

- **Keras 3 `.h5` loading failure.** Loading a saved model with
  `load_model('model.h5')` raised `ValueError: Could not deserialize
  'keras.metrics.mse' because it is not a KerasSaveable subclass`, a known
  Keras 3 bug in legacy-H5 compile-config deserialization. Since
  `TestSimulation.py` only runs inference (`model.predict`), it never needs
  the optimizer/loss state, so loading with `compile=False` sidesteps the
  broken code path entirely.

- **Simulator never responded in Autonomous Mode.** The server accepted
  the TCP connection but never fired its `connect` handler, and debug
  logging showed every `telemetry` event arriving as `"received event ...
  from None"` / `"None is not connected to namespace /"`. This is the
  signature of a Socket.IO protocol mismatch: the simulator's embedded
  client speaks the older Engine.IO protocol 3, while `python-socketio`
  5.x and `python-engineio` 4.x speak protocol 4 by default. Pinning
  `python-socketio==4.6.1` and `python-engineio==3.13.2` in
  `pyproject.toml` resolved it.


