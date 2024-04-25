from . import np

class OneHotEncoder:
    def __init__(self):
        self.label_count = None

    def fit(self, labels):
        self.label_count = len(set(labels))

    def encode(self, labels):
        if self.label_count is None:
            raise ValueError("Fit the encoder first using the 'fit' method.")

        encoded = np.zeros((len(labels), self.label_count), dtype=int)
        for i, label in enumerate(labels):
            if label < 0 or label >= self.label_count:
                raise ValueError(f"Label {label} is out of range.")
            encoded[i, label] = 1
        return encoded

    def decode(self, one_hot_encoded):
        if self.label_count is None:
            raise ValueError("Fit the encoder first using the 'fit' method.")

        decoded_labels = np.argmax(one_hot_encoded, axis=1)
        return decoded_labels

