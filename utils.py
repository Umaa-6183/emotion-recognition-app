# utils.py

from collections import Counter


def majority_vote(emotions):
    """
    Returns the most common emotion from the given list.
    If a tie, returns one of the most common at random.
    """
    if not emotions:
        return None
    counter = Counter(emotions)
    most_common = counter.most_common(1)[0][0]
    return most_common
