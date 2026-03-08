from Module import *
from Loss import *
from Optimizer import *

class CharRNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_xh = Linear(input_size, hidden_size)
        self.w_hh = Linear(hidden_size, hidden_size)
        self.w_hy = Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        h_next = (self.w_xh(x) + self.w_hh(h_prev)).tanh()
        y = self.w_hy(h_next)
        return y, h_next


def main():
    text = "learning deep learning is fun"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    hidden_size = 64
    learning_rate = 0.05
    epochs = 500
    seq_length = 10

    model = CharRNN(vocab_size, hidden_size, vocab_size)
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    def to_one_hot(char):
        vec = np.zeros((1, vocab_size))
        vec[0, char_to_ix[char]] = 1
        return Tensor(vec)

    print(f"-----Training-----")
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(text) - seq_length, 3):
            input_chars = text[i: i + seq_length]
            target_chars = text[i + 1: i + seq_length + 1]

            h = Tensor(np.zeros((1, hidden_size)))
            model.zero_grad()
            seq_loss = Tensor(np.array([0.0]))

            for t in range(seq_length):
                x = to_one_hot(input_chars[t])
                target = to_one_hot(target_chars[t])

                y_pred, h = model(x, h)
                step_loss = criterion(y_pred, target)
                seq_loss = seq_loss + step_loss
            seq_loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    np.clip(p.grad, -5, 5, out=p.grad)

            optimizer.step()
            total_loss += seq_loss.data

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Avg Loss: {(total_loss / (len(text) / 3)).item():.4f}")
        if epoch > 0 and epoch % 100 == 0:
            optimizer.lr *= 0.5
    print("\n-----Testing-----")
    for start_char in ["l", "d", "f"]:
        result = [start_char]
        h_gen = Tensor(np.zeros((1, hidden_size)))
        curr_x = to_one_hot(start_char)

        for _ in range(len(text) - 1):
            y_pred, h_gen = model(curr_x, h_gen)
            logits = y_pred.data.flatten()
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            next_ix = np.random.choice(range(vocab_size), p=probs)
            next_char = ix_to_char[next_ix]

            result.append(next_char)
            curr_x = to_one_hot(next_char)

        print(f"'{start_char}' -> {''.join(result)}")


if __name__ == "__main__":
    main()