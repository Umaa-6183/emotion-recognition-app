from collections import Counter


def fuse_emotions(facial, audio, text, context):
    emotions = [facial, audio, text]
    emotion_weights = Counter(emotions)

    if context['culture'] == 'South Asian' and text == 'Fear':
        emotion_weights['Anxiety'] += 1

    final = emotion_weights.most_common(1)[0][0]
    return final
