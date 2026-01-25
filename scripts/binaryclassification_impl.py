from datetime import datetime
import matplotlib.pyplot as plt

from mchnpkg.deepl import binary_classification


def main():
    # You can change these if you want
    d = 20
    n = 2000
    epochs = 10000
    eta = 0.001

    W1, W2, W3, W4, losses = binary_classification(d=d, n=n, epochs=epochs, eta=eta)

    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Binary Classification Training Loss")

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"crossentropyloss_{ts}.pdf"
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
