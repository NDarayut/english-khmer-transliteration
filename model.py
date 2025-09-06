from tensorflow.keras.layers import (
    Input, Dense, Embedding, GRU, Bidirectional, TimeDistributed, AdditiveAttention, Concatenate
)
from tensorflow.keras.models import Model


def build_inference_models(model):
    """
    Build encoder and decoder models for inference from the trained Bidirectional Attention GRU model.
    Args:
        model: The trained Bidirectional Attention GRU model.
    Returns:
        encoder_model: The encoder model for inference.
        decoder_model: The decoder model for inference.
        gru_units: Number of GRU units in the decoder.
    """
    # Extract layers by name for clarity
    layers = {layer.name: layer for layer in model.layers}

    # Get the necessary layers (you might need to adjust layer names based on your actual model)
    encoder_embedding = None
    encoder_gru = None
    decoder_embedding = None
    decoder_gru = None
    encoder_proj_dense = None
    attention_layer = None
    decoder_dense = None

    # Find layers by type (since names might be auto-generated)
    for layer in model.layers:
        if isinstance(layer, Embedding) and encoder_embedding is None:
            encoder_embedding = layer
        elif isinstance(layer, Embedding) and encoder_embedding is not None:
            decoder_embedding = layer
        elif isinstance(layer, Bidirectional):
            encoder_gru = layer
        elif isinstance(layer, GRU):
            decoder_gru = layer
        elif isinstance(layer, TimeDistributed):
            encoder_proj_dense = layer
        elif isinstance(layer, AdditiveAttention):
            attention_layer = layer
        elif isinstance(layer, Dense) and layer.activation.__name__ == 'softmax':
            decoder_dense = layer

    # Encoder inference model
    encoder_inputs = Input(shape=(None,))
    enc_emb = encoder_embedding(encoder_inputs)
    encoder_outputs = encoder_gru(enc_emb)
    encoder_proj = encoder_proj_dense(encoder_outputs)
    encoder_model = Model(encoder_inputs, [encoder_outputs, encoder_proj])

    # Decoder inference model
    gru_units = decoder_gru.units

    # Input states for inference
    decoder_inputs = Input(shape=(None,))
    decoder_state_input = Input(shape=(gru_units,))
    encoder_outputs_input = Input(shape=(None, 2*gru_units))  # Bidirectional output
    encoder_proj_input = Input(shape=(None, gru_units))      # Projected encoder outputs

    # Single step decoder
    dec_emb = decoder_embedding(decoder_inputs)
    decoder_outputs_inf, decoder_state_inf = decoder_gru(dec_emb, initial_state=decoder_state_input)

    # Attention for inference
    context_inf = attention_layer([decoder_outputs_inf, encoder_proj_input])

    # Combine and get final output
    decoder_combined_inf = Concatenate(axis=-1)([decoder_outputs_inf, context_inf])
    decoder_outputs_final_inf = decoder_dense(decoder_combined_inf)

    decoder_model = Model(
        [decoder_inputs, decoder_state_input, encoder_proj_input],
        [decoder_outputs_final_inf, decoder_state_inf]
    )

    return encoder_model, decoder_model, gru_units