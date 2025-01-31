import tensorflow as tf
from helper.utils import image_colour_convert, standardise, standardise_tf, normalise
from scipy.stats import wasserstein_distance


@tf.function
def rescale(image):
    return (0.5 * image) + 0.5

# Discriminator takes half of the loss (practical advice from the paper)
@tf.function
def discriminator_loss(y_true, y_pred):
    y_true_loss = tf.keras.losses.MeanSquaredError()(y_true, tf.ones_like(input=y_true, dtype=y_true.dtype))
    y_pred_loss = tf.keras.losses.MeanSquaredError()(y_pred, tf.zeros_like(input=y_pred, dtype=y_pred.dtype))
    return 0.5 * (y_true_loss + y_pred_loss)

@tf.function
def generator_loss(y_pred):
    return tf.keras.losses.MeanSquaredError()(y_pred, tf.ones_like(input=y_pred, dtype=y_pred.dtype))

@tf.function
def cycle_loss(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

@tf.function
def identity_loss(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

@tf.function
def calculate_wasserstein_distance(y_true, y_pred):
    return tf.cast(wasserstein_distance(y_true, y_pred), tf.float32)

@tf.function
def extract_activations(images, model, mean, stddev):
    activations = []
    for (idx, img) in enumerate(tf.unstack(images)):
        img = (img + 1) * 127.5  # [0-255]
        img = image_colour_convert(img, 'rgb')
        img = standardise_tf(img)
        img = normalise(img, mean, stddev)
        activation = model(tf.expand_dims(img, axis=0), training=False)
        activations.append(tf.squeeze(activation, axis=0)) # Remove batch dimension

    activations = tf.stack(activations, axis=0)  # Convert list to tensor
    return tf.reduce_mean(activations, axis=[1, 2])

@tf.function
def domain_shift_loss(y_true, y_pred, model):
    y_true = extract_activations(y_true, model["model"], tf.cast(model["mean"], dtype=y_true.dtype),
                                 tf.cast(model["stddev"], dtype=y_true.dtype))
    print("y_true shape:", y_true.shape)
    print("y_true dtype:", y_true.dtype)
    y_pred = extract_activations(y_pred, model["model"], tf.cast(model["mean"], dtype=y_true.dtype),
                                 tf.cast(model["stddev"], dtype=y_true.dtype))
    print("y_pred shape:", y_pred.shape)
    print("y_pred dtype:", y_pred.dtype)

    tf.debugging.assert_shapes([(y_true, y_pred.shape)])
    tf.debugging.assert_type(y_true, tf.float32)
    tf.debugging.assert_type(y_pred, tf.float32)

    def calculate_wd(y_true, y_pred):
        return tf.cast(wasserstein_distance(y_true, y_pred), tf.float32)

    return tf.numpy_function(func=calculate_wd, inp=[y_true, y_pred], Tout=tf.float32)
    # return tf.numpy_function(func=calculate_wasserstein_distance, inp=[y_true, y_pred], Tout=tf.float32)

def combined_generator_loss(imgs_A, imgs_B, disc_fake_A, disc_fake_B, cyc_A, cyc_B, id_A, id_B,
                            weights, pretrained_model, fake_A, dsl_pool):
    # adversarial loss
    G_AB_adv_loss = generator_loss(disc_fake_B)
    G_BA_adv_loss = generator_loss(disc_fake_A)
    # cycle loss
    G_ABA_cycle_loss = weights["lambda_cycle"] * cycle_loss(imgs_A, cyc_A)
    G_BAB_cycle_loss = weights["lambda_cycle"] * cycle_loss(imgs_B, cyc_B)
    # identity loss
    G_AA_identity_loss = weights["lambda_id"] * identity_loss(imgs_A, id_A)
    G_BB_identity_loss = weights["lambda_id"] * identity_loss(imgs_B, id_B)

    # Compute additional loss when the buffer is full
    pool_result = dsl_pool.query(imgs_A, fake_A, cyc_A, id_A)
    if pool_result is not None:
        pool_img_A, pool_fake_A, pool_cyc_A, pool_id_A = pool_result
        print(domain_shift_loss(pool_img_A, pool_fake_A, pretrained_model))
        G_AfakeA_dsl = weights["lambda_domain_shift"] * domain_shift_loss(pool_img_A, pool_fake_A, pretrained_model)
        G_ABA_cyc_dsl = weights["lambda_domain_shift"] * domain_shift_loss(pool_img_A, pool_cyc_A, pretrained_model)
        G_AA_id_dsl = weights["lambda_domain_shift"] * domain_shift_loss(pool_img_A, pool_id_A, pretrained_model)
        dsl_loss = G_AfakeA_dsl + G_ABA_cyc_dsl + G_AA_id_dsl
    else:
        G_AfakeA_dsl = 0.0
        G_ABA_cyc_dsl = 0.0
        G_AA_id_dsl = 0.0
        dsl_loss = G_AfakeA_dsl + G_ABA_cyc_dsl + G_AA_id_dsl

    # adversarial loss + cycle loss + identity loss + segmentation_loss
    G_AB_loss = G_AB_adv_loss + G_ABA_cycle_loss + G_AA_identity_loss + dsl_loss
    G_BA_loss = G_BA_adv_loss + G_BAB_cycle_loss + G_BB_identity_loss
    loss = {"total_loss": G_AB_loss + G_BA_loss, "G_AB_loss": G_AB_loss, "G_BA_loss": G_BA_loss,
            "G_AB_adv_loss": G_AB_adv_loss, "G_BA_adv_loss": G_BA_adv_loss,
            "G_ABA_cyc_loss": G_ABA_cycle_loss, "G_BAB_cyc_loss": G_BAB_cycle_loss,
            "G_AA_id_loss": G_AA_identity_loss, "G_BB_id_loss": G_BB_identity_loss,
            "G_AfakeA_ds_loss": G_AfakeA_dsl, "G_ABA_cyc_ds_loss": G_ABA_cyc_dsl, "G_AA_id_ds_loss": G_AA_id_dsl}

    return loss


def combined_discriminator_loss(disc_real_A, disc_real_B, disc_fake_A, disc_fake_B):
    # calculate Discriminator loss
    D_A_loss = discriminator_loss(disc_real_A, disc_fake_A)
    D_B_loss = discriminator_loss(disc_real_B, disc_fake_B)
    return {"total_loss": D_A_loss + D_B_loss,
            "D_A_loss": D_A_loss,
            "D_B_loss": D_B_loss}
