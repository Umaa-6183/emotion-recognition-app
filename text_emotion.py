from transformers import pipeline

classifier = pipeline('text-classification',
                      model='bhadresh-savani/bert-base-go-emotion')


def detect_text_emotion(text):
    result = classifier(text)[0]
    return result['label']
