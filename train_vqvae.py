import jax
from vae import make_step, VQVAE
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter


key = jax.random.PRNGKey(69)

model = VQVAE(key=key)

optimizer = optax.adam(1e-4)
opt_state = optimizer.init(model)

writer = SummaryWriter(log_dir="./runs")

for i in range(0, 100):
    x = jnp.ones((10, 80, 100))
    model, opt_state, total_loss, reconstruct_loss, commit_loss, codebook_updates, y = (
        make_step(model, optimizer, opt_state, x)
    )
    print(f"Total loss: {total_loss}")

    # Log codebook updates to TensorBoard
    writer.add_scalar("Loss/Total", total_loss, i)
    writer.add_scalar("Loss/Reconstruct", reconstruct_loss, i)
    writer.add_scalar("Loss/Commit", commit_loss, i)
    # writer.add_histogram('Codebook Updates/Cluster Size', jnp.mean(codebook_updates[0], axis=0), i)
    # writer.add_scalar('Codebook Updates/Codebook Avg', jnp.mean(codebook_updates[1], axis=0), i)
    writer.add_histogram(
        "Codebook Updates/Code ids used", jnp.sum(codebook_updates[1], axis=(0)), i
    )
    writer.add_histogram(
        "Codebook Updates/Code means", jnp.mean(codebook_updates[0][2], axis=(0, 2)), i
    )
    writer.add_histogram(
        "Codebook Updates/Code stds", jnp.std(codebook_updates[0][2], axis=(0, 2)), i
    )