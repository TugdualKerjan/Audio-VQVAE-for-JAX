import equinox as eqx
import jax
import jax.numpy as jnp

def update_codebook_ema(model, updates: tuple, codebook_indices, key=None):
    avg_updates = jax.tree.map(lambda x: jax.numpy.mean(x, axis=0), updates)

    # Calculate which codes are too often used and yeet them. Prior is uniform.
    h = jnp.histogram(
        codebook_indices, bins=model.quantizer.K, range=(0, model.quantizer.K)
    )[0] / len(codebook_indices)
    part_that_should_be = 1 / model.quantizer.K
    mask = (h > 2 * part_that_should_be) | (h < 0.5 * part_that_should_be)
    rand_embed = (
        jax.random.normal(key, (model.quantizer.K, model.quantizer.D)) * mask[:, None]
    )
    avg_updates = (
        avg_updates[0],
        avg_updates[1],
        jnp.where(mask[:, None], rand_embed, avg_updates[2]),
    )

    def where(q):
        return q.quantizer.cluster_size, q.quantizer.codebook_avg, q.quantizer.codebook

    # Update the codebook and other trackers.
    model = eqx.tree_at(where, model, avg_updates)
    return model

@eqx.filter_value_and_grad(has_aux=True)
def calculate_losses(model, x):
    z_e, z_q, codebook_updates, y = jax.vmap(model)(x)

    # Are the inputs and outputs close?
    reconstruct_loss = jnp.mean(jnp.linalg.norm((x - y), ord=2, axis=(1, 2)))

    # Are the output vectors z_e close to the codes z_q ?
    commit_loss = jnp.mean(
        jnp.linalg.norm(z_e - jax.lax.stop_gradient(z_q), ord=2, axis=(1, 2))
    )
    codebook = jnp.mean(codebook_updates[0][2], axis=0) #| hide_line
    # print(codebook.shape) #| hide_line
    # print(codebook) #| hide_line
    # print(jnp.mean(codebook, axis=-1).shape) #| hide_line
    # print(f"STDR: {jnp.std(codebook, axis=-1)}") #| hide_line
    # print(f"log: {jnp.log(jnp.clip(jnp.mean(codebook, axis=-1), min=1e-5)) }") #| hide_line
    KL_loss = 0.5 * jnp.sum(jnp.mean(codebook, axis=-1)**2 + jnp.var(codebook, axis=-1) - jnp.log(jnp.clip(jnp.std(codebook, axis=-1), min=1e-6)) - 1)

    total_loss = reconstruct_loss + commit_loss + KL_loss

    return total_loss, (reconstruct_loss, commit_loss, codebook_updates, y)


@eqx.filter_jit
def make_step(model, optimizer, opt_state, x, key):
    (total_loss, (reconstruct_loss, commit_loss, codebook_updates, y)), grads = (
        calculate_losses(model, x)
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = update_codebook_ema(model, codebook_updates[0], codebook_updates[1], key)

    return (
        model,
        opt_state,
        total_loss,
        reconstruct_loss,
        commit_loss,
        codebook_updates,
        y,
    )