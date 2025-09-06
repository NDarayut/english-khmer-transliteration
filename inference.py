from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import clean_text, normalize_khmer
from model import build_inference_models
import numpy as np
import Levenshtein

# Load model and assets
model = load_model('./BiGRU-Attention/bi_gru_attention_khmer_transliterator.keras')
with open('./BiGRU-Attention/gru_transliteration_assets.pkl', 'rb') as f:
    assets = pickle.load(f)

eng_tokenizer = assets['eng_tokenizer']
khm_tokenizer = assets['khm_tokenizer']
max_eng_len = assets['max_eng_len']
max_khm_len = assets['max_khm_len']

encoder_model, decoder_model, gru_units = build_inference_models(model)

def transliterate_text(eng_input, beam_width=3, max_length=32):
    """
    Transliterate an English input sequence to Khmer using trained encoder-decoder model and beam search.
    Args:
        eng_input (str): The input English text.
        beam_width (int): The beam width for beam search.
        max_length (int): Maximum length of the output sequence.
    Returns:
        str: The translated Khmer text.
    """

    # Clean and encode input
    cleaned = clean_text(eng_input, is_khmer=False)
    enc_seq = eng_tokenizer.texts_to_sequences([cleaned])[0]
    enc_padded = pad_sequences([enc_seq], maxlen=max_eng_len, padding='post')
    
    # Get encoder outputs and projections
    _, encoder_proj = encoder_model.predict(enc_padded, verbose=0)
    
    # Initialize decoder state (zeros for GRU)
    initial_state = np.zeros((1, gru_units))
    
    # Start token
    start_token = khm_tokenizer.word_index.get('\t', 1)
    end_token = khm_tokenizer.word_index.get('\n', 2)
    
    # Initialize beam search
    beams = [{
        'seq': [start_token],
        'prob': 1.0,
        'state': initial_state,
        'finished': False
    }]
    
    finished_beams = []
    
    for step in range(max_length):
        if not beams:  # All beams finished
            break
            
        new_beams = []
        
        for beam in beams:
            if beam['finished']:
                finished_beams.append(beam)
                continue
                
            # Prepare decoder input (last token)
            last_token = beam['seq'][-1]
            target_seq = np.array([[last_token]])
            
            # Predict next token
            outputs, new_state = decoder_model.predict([
                target_seq, 
                beam['state'], 
                encoder_proj
            ], verbose=0)
            
            # Get probabilities for next token
            probs = outputs[0, -1, :]  # Last timestep, all vocab
            
            # Get top candidates
            top_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_indices:
                token_prob = probs[idx]
                if token_prob > 1e-8:  # Avoid very low probabilities
                    new_seq = beam['seq'] + [idx]
                    new_prob = beam['prob'] * token_prob
                    
                    # Check if finished (end token or max length)
                    finished = (idx == end_token) or (len(new_seq) >= max_length)
                    
                    new_beam = {
                        'seq': new_seq,
                        'prob': new_prob,
                        'state': new_state,
                        'finished': finished
                    }
                    
                    if finished:
                        finished_beams.append(new_beam)
                    else:
                        new_beams.append(new_beam)
        
        # Keep only top beams for next iteration
        beams = sorted(new_beams, key=lambda x: x['prob'], reverse=True)[:beam_width]
        
        # If we have enough finished beams, we can stop early
        if len(finished_beams) >= beam_width:
            break
    
    # Combine finished and unfinished beams
    all_beams = finished_beams + beams
    if not all_beams:
        return ""
    
    # Get best beam
    best_beam = max(all_beams, key=lambda x: x['prob'])
    best_seq = best_beam['seq']
    
    # Decode sequence to text
    decoded = []
    for idx in best_seq:
        if idx == start_token:  # Skip start token
            continue
        if idx == end_token:    # Stop at end token
            break
        if idx in khm_tokenizer.index_word:
            decoded.append(khm_tokenizer.index_word[idx])
    
    return ''.join(decoded)


def transliterate_top_n(eng_input, beam_width=5, max_length=32, n=3):
    """
    Get top-n transliterations for a given English input.
    Args:
        eng_input (str): The input English text.
        beam_width (int): The beam width for beam search.
        max_length (int): Maximum length of the output sequence.
        n (int): Number of top transliterations to return.
    Returns:
        list: List of top-n transliterated Khmer texts.
    """
    # Clean and encode input
    cleaned = clean_text(eng_input, is_khmer=False)
    enc_seq = eng_tokenizer.texts_to_sequences([cleaned])[0]
    enc_padded = pad_sequences([enc_seq], maxlen=max_eng_len, padding='post')
    
    # Get encoder outputs and projections
    encoder_outputs_val, encoder_proj_val = encoder_model.predict(enc_padded, verbose=0)
    
    # Initialize decoder state
    initial_state = np.zeros((1, gru_units))
    
    # Start and end tokens
    start_token = khm_tokenizer.word_index.get('\t', 1)
    end_token = khm_tokenizer.word_index.get('\n', 2)
    
    # Initialize beam search
    beams = [{
        'seq': [start_token],
        'prob': 1.0,
        'state': initial_state,
        'finished': False
    }]
    
    finished_beams = []
    
    for step in range(max_length):
        if not beams:
            break
        new_beams = []
        for beam in beams:
            if beam['finished']:
                finished_beams.append(beam)
                continue
            last_token = beam['seq'][-1]
            target_seq = np.array([[last_token]])
            
            # Decoder prediction
            outputs, new_state = decoder_model.predict(
                [target_seq, beam['state'], encoder_proj_val], verbose=0
            )
            
            probs = outputs[0, -1, :]
            top_indices = np.argsort(probs)[-beam_width:]
            
            for idx in top_indices:
                token_prob = probs[idx]
                if token_prob > 1e-8:
                    new_seq = beam['seq'] + [idx]
                    new_prob = beam['prob'] * token_prob
                    finished = (idx == end_token) or (len(new_seq) >= max_length)
                    
                    new_beam = {
                        'seq': new_seq,
                        'prob': new_prob,
                        'state': new_state,
                        'finished': finished
                    }
                    
                    if finished:
                        finished_beams.append(new_beam)
                    else:
                        new_beams.append(new_beam)
        
        # Keep top beams
        beams = sorted(new_beams, key=lambda x: x['prob'], reverse=True)[:beam_width]
        if len(finished_beams) >= n:
            break
    
    # Combine finished and unfinished beams
    all_beams = finished_beams + beams
    if not all_beams:
        return [''] * n

    # Get top `n` beams
    top_beams = sorted(all_beams, key=lambda x: x['prob'], reverse=True)[:n]
    variants = []
    
    for beam in top_beams:
        decoded = []
        for idx in beam['seq']:
            if idx == start_token:
                continue
            if idx == end_token:
                break
            decoded.append(khm_tokenizer.index_word.get(idx, ''))
        variants.append(''.join(decoded))
    
    return variants

def transliterate_with_dict(eng_input, beam_width=5, max_length=32, n=5, max_distance=2):
    """
    Transliterate using beam search and align results with a Khmer dictionary.
    If no exact match is found, fall back to the closest dictionary entry
    within max_distance edits.
    
    Args:
        eng_input (str): The input English text.
        beam_width (int): The beam width for beam search.
        max_length (int): Maximum length of the output sequence.
        n (int): Number of top transliterations to consider.
        max_distance (int): Maximum Levenshtein distance for fuzzy matching.
    
    Returns:
        list: List of top valid transliterated Khmer texts.
    """
    # Load Khmer dictionary
    with open('khmer_dictionary.txt', 'r', encoding='utf-8') as f:
        khmer_words = set(line.strip() for line in f)

    # Step 1: Get top candidates from model (beam search)
    top_candidates = transliterate_top_n(eng_input, beam_width, max_length, n=n)
    # Normalize model outputs
    top_candidates = [normalize_khmer(cand) for cand in top_candidates]
    print(f"Top Candidates from model: {top_candidates}")

    valid_candidates = []
    used = set()

    # Step 2: Exact dictionary matches
    for cand in top_candidates:
        if cand in khmer_words and cand not in used:
            valid_candidates.append(cand)
            used.add(cand)
    
    # Step 3: Fuzzy match if not enough valid candidates
    if len(valid_candidates) < n:
        for cand in top_candidates:
            best_match = None
            best_dist = float("inf")
            for word in khmer_words:
                # Optional: skip words that are too different in length
                if abs(len(word) - len(cand)) > 2:
                    continue
                d = Levenshtein.distance(cand, word)
                if d <= max_distance and d < best_dist:
                    best_match = word
                    best_dist = d
            if best_match and best_match not in used:
                valid_candidates.append(best_match)
                used.add(best_match)

    # Step 4: Deduplicate and preserve order
    valid_candidates = list(dict.fromkeys(valid_candidates))

    print(f"Valid Candidates after filtering: {valid_candidates}")
    return valid_candidates if valid_candidates else top_candidates