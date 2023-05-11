# Change dataset name on line 138 to run (must be .csv)
# Input is a .txt file with each tongue twister on a new line (line 65)
from g2p_en import G2p
import numpy as np
from pathlib import Path

# Initialise G2P
g2p = G2p()

def g2p_processing(tongue_twister):
    """
    Performs grapheme-to-phoneme (G2P) conversion using `g2p-en` and formats the output (inc. removing stress markers)
    :param tongue_twister: Plain text tongue-twister
    :return: Tongue-twister transcribed in ARPABET phonetic alphabet
    """
    phonetic_transcription = g2p(tongue_twister)
    formatted = " ".join(sound for sound in phonetic_transcription)
    formatted = formatted.split("   ")

    # Remove stress markers (numbers)
    for word_index, word in enumerate(formatted):
        sounds = word.split()
        for segment_position, phoneme in enumerate(sounds):
            if phoneme[-1].isnumeric():
                sounds[segment_position] = phoneme[:-1]
                formatted[word_index] = sounds
    return formatted


def phoneme_ratio(g2p_tongue_twister):
    """
    Calculates ratio of unique sounds to overall number of sounds
    :param g2p_tongue_twister: Tongue-twister that has been processed with G2P
    :return: ratio score (lower = better)
    """
    unique_phonemes = set()
    total_phonemes = 0
    for word in g2p_tongue_twister:
        for phoneme in word:
            # Ignore punctuation
            if phoneme.isalpha():
                total_phonemes += 1
                unique_phonemes.add(phoneme)
    ratio = len(unique_phonemes) / total_phonemes
    return ratio


def word_initial_ratio(g2p_tongue_twister):
    """
    Calculates ratio of unique word-initial sounds to overall number of words
    :param g2p_tongue_twister: Tongue-twister that has been processed with G2P
    :return: Ratio score (lower=better)
    """
    unique_phonemes = set()
    total_words = 0
    for word in g2p_tongue_twister:
        # Ignore punctuation (all phonemes are contained in lists)
        if type(word) is list:
            total_words += 1
            unique_phonemes.add(word[0])
    ratio = len(unique_phonemes) / total_words
    return ratio


def tt_measure(gen_file):
    with open(gen_file, "r") as file:
        init_list = []
        overall_list = []
        # dataset_init_score = 0
        # dataset_overall_score = 0
        line_count = 0
        for line in file:
            line_count += 1
            # (my test file had blank lines, hence the try/except)
            transcribed = g2p_processing(line)
            initial_score = word_initial_ratio(transcribed)
            overall_score = phoneme_ratio(transcribed)
            # dataset_init_score += initial_score
            # dataset_overall_score += overall_score
            init_list.append(initial_score)
            overall_list.append(overall_score)
            # print(f"Initial Score: {initial_score} \nOverall Score: {overall_score}")


        # MEANS
        # dataset_init_score /= no_lines
        # dataset_overall_score /= no_lines
        numpy_init = np.mean(init_list)
        numpy_overall = np.mean(overall_list)
        init_std = np.std(init_list)
        overall_std = np.std(overall_list)

        # print(f"Dataset = {dataset_init_score} and {dataset_overall_score}")
        print(f"Dataset = {numpy_init} (SD: {init_std}) and {numpy_overall} (SD: {overall_std})")

if __name__ == '__main__':
    pass