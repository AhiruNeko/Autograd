from Module import *
import os
from PIL import Image
from Loss import *
from Optimizer import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

class DigitModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = Linear(input_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x


def load_data(data_dir):
    images = []
    labels = []
    for i in range(100):
        filename = f"{i:02d}.png"
        path = os.path.join(data_dir, filename)
        img = Image.open(path).convert('L')
        img_arr = np.array(img)
        img_data = (255.0 - img_arr) / 255.0
        images.append(img_data.flatten())
        labels.append(i // 10)

    return np.array(images), np.array(labels)


def main():
    input_dim = 64
    hidden_dim = 32
    output_dim = 10
    lr = 0.01
    epochs = 50

    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DigitModel(input_dim, hidden_dim, output_dim)
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    print("-----Training-----")
    model.train()
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        total_loss = 0
        for i in range(len(X_train_shuffled)):
            x = Tensor(X_train_shuffled[i:i + 1])
            target_vec = np.zeros((1, 10))
            target_vec[0, y_train_shuffled[i]] = 1
            target = Tensor(target_vec)
            model.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Avg Loss: {total_loss / len(X_train):.4f}")
    print("\-----Testing-----")
    model.eval()
    correct = 0

    print(f"{'Sample ID':<10} | {'Prediction':<8} | {'Actual':<8} | {'Result':<8} | {'Probability'}")
    print("-" * 45)

    for i in range(len(X_test)):
        x_val = Tensor(X_test[i:i + 1])
        logits = model(x_val)
        probs = np.exp(logits.data - np.max(logits.data)) / np.sum(np.exp(logits.data - np.max(logits.data)))
        max_prob = round(np.max(probs), 4)
        prediction = np.argmax(logits.data)
        actual = y_test[i]

        if prediction == actual:
            correct += 1
        if i < 25:
            res = "√" if prediction == actual else "×"
            print(f"Sample {i:<3} | {prediction:<8} | {actual:<8} | {res:<8} | {max_prob:<8}")

    accuracy = (correct / len(X_test)) * 100
    print("-" * 45)
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()