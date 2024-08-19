import pandas as pd


def read_thresholds(threshold_path):
    df = pd.read_csv(threshold_path)
    return df


def select_threshold(thresholds, topic, strategy):
    topics = thresholds["topic"].tolist()
    key = topic.split(" ")[0]
    if key in topics:
        row = thresholds.loc[
            (thresholds["topic"] == key) & (thresholds["strategy"] == strategy)
        ]
        threshold = row["threshold"].values[0]
    else:
        print("topic not found, using the default threshold = 0.5")
        threshold = 0.5
    return threshold


def threshold_to_binary_labels(predicted_value, threshold=0.5, similarity=True):
    binary_labels = []
    if similarity:
        smaller_than_threshold = "excluded"
        larger_than_threshold = "included"
    else:
        smaller_than_threshold = "excluded"
        larger_than_threshold = "included"
    for v in predicted_value:
        if v <= threshold:
            binary_labels.append(smaller_than_threshold)
        else:
            binary_labels.append(larger_than_threshold)

    return binary_labels
