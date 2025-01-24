from collections import defaultdict
import datetime
import json
import time
import jax
from vae import make_step_vae, VAE, mel_spec_base_jit
import jax.numpy as jnp
import optax
import equinox as eqx
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import datasets
import librosa
import numpy as np

RANDOM = jax.random.PRNGKey(69)
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10
CHECKPOINT_PATH = "checkpoints"
INIT_FROM = "scratch"

SAMPLE_RATE = 22050
SEGMENT_SIZE = 8192

def transform(sample):
    """Based off the original code that can be found here: https://github.com/jik876/hifi-gan/blob/master/meldataset.py

    Args:
        sample (dict): dict entry in HF Dataset

    Returns:
        dict: updated entry
    """
    k = jax.random.key(0)
    # global RANDOM, SAMPLE_RATE

    # RANDOM, k = jax.random.split(RANDOM)
    wav = sample["audio"]["array"]
    if sample["audio"]["sampling_rate"] != SAMPLE_RATE:
        librosa.resample(wav, sample["audio"]["sampling_rate"], SAMPLE_RATE)
    wav = librosa.util.normalize(wav) * 0.95
    if wav.shape[0] >= SEGMENT_SIZE:
        max_audio_start = wav.shape[0] - SEGMENT_SIZE
        audio_start = jax.random.randint(k, (1,), 0, max_audio_start)[0]
        wav = wav[audio_start : audio_start + SEGMENT_SIZE]

    wav = np.expand_dims(wav, 0)

    mel = mel_spec_base_jit(wav=wav)
    # print(mel.shape)
    return {"mel": np.array(mel), "audio": np.array(wav), "sample_rate": SAMPLE_RATE}


# lj_speech_data = datasets.load_dataset("keithito/lj_speech", trust_remote_code=True)
# lj_speech_data = lj_speech_data.map(transform)

lj_speech_data = datasets.load_from_disk("transformed_lj_speech")
lj_speech_data = lj_speech_data.with_format("jax")

lj_speech_data = lj_speech_data["train"].train_test_split(0.01)

train_data, eval_data = lj_speech_data["train"], lj_speech_data["test"]


model = VAE(key=RANDOM)

if INIT_FROM == "scratch":
    print("Starting training from scratch")

    # current_step = 0
    starting_epoch = 0
elif INIT_FROM == "resume":
    print(f"Resuming training from {CHECKPOINT_PATH}")

    def load(filename):
        with open(filename, "rb") as f:
            checkpoint_state = json.loads(f.readline().decode())
            return (
                eqx.tree_deserialise_leaves(f, model),
                checkpoint_state,
            )

    model, checkpoint_state = load(CHECKPOINT_PATH)
    # current_step = checkpoint_state["current_step"]
    current_epoch = checkpoint_state["current_epoch"]

optimizer = optax.adam(1e-4)
opt_state = optimizer.init(model)

writer = SummaryWriter(
    log_dir="./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: F821
)

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

step = 0

# Timing stats dictionary
timing_stats = defaultdict(list)

for epoch in range(starting_epoch, NUM_EPOCHS):
    epoch_start = time.time()
    wait_start = time.time()
    # print("wtf")
    # RANDOM, k = jax.random.split(RANDOM)
    # permutation = jax.random.permutation(k, jnp.arange(0, pouet_audio.shape[0]))
    for i, batch in enumerate(train_data.iter(batch_size=BATCH_SIZE)):

        wait_time = time.time() - wait_start

        # Measure data loading time
        data_load_start = time.time()
        # mels = pouet_mel[permutation[i : i + BATCH_SIZE]]
        # wavs = pouet_audio[permutation[i : i + BATCH_SIZE]]
        mels, wavs = batch["mel"], batch["audio"]
        data_load_time = time.time() - data_load_start

        # Measure training step time
        train_start = time.time()
        #    x = jnp.ones((10, 80, 100))
        results = (
            make_step_vae(model, optimizer, opt_state, mels)
        ) 
        # Force sync for accurate timing
        jax.block_until_ready(results)

        train_time = time.time() - train_start

        # Measure logging time
        log_start = time.time()
        (
            model, opt_state, total_loss, reconstruct_loss, output
        ) = results

        step += 1
        writer.add_scalar("Total loss", total_loss, step)
        writer.add_scalar("Reconstruct loss", reconstruct_loss, step)
        # writer.add_scalar("Commit loss", commit_loss, step)

        if step % 5 == 0:
            writer.add_figure(
                "generated/output",
                plot_spectrogram(np.array(output[0])),
                step,
            )
            writer.add_figure(
                "generated/input",
                plot_spectrogram(np.array(mels[0])),
                step,
            )
        log_time = time.time() - log_start

        # Store timing stats
        timing_stats["data_loading"].append(data_load_time)
        timing_stats["training"].append(train_time)
        timing_stats["logging"].append(log_time)
        timing_stats["wait_time"].append(wait_time)

        # Print running averages every 10 steps
        # if i % 1 == 0:
        # print(f"\nTiming stats (last 10 steps):")
        # print(f"Data loading: {np.mean(timing_stats['data_loading'][-1:]):.3f}s")
        # print(f"Training: {np.mean(timing_stats['training'][-1:]):.3f}s")
        # print(f"Logging: {np.mean(timing_stats['logging'][-1:]):.3f}s")
        # print(f"Iter wait: {np.mean(timing_stats['wait_time'][-1:]):.3f}s")

        wait_start = time.time()

    # Save model
    eqx.tree_serialise_leaves("./vae.eqx", model)