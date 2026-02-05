"""
Demo script for Homework:
- runs binary_classification
- plots loss vs epoch
- saves plot as PDF with timestamp: crossentropyloss_YYYYMMDDhhmmss.pdf

Run from repo root:
    python scripts/binaryclassification_impl.py
"""

from datetime import datetime
import matplotlib.pyplot as plt

from mchnpkg import binary_classification


def main():
    # Inputs (adjust if you want)
    d = 20
    n = 2000
    epochs = 10000
    eta = 0.001

    # Train
    W1, W2, W3, W4, losses = binary_classification(d=d, n=n, epochs=epochs, eta=eta)

    # Plot loss vs epochs
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss vs Epochs")
    plt.grid(True)

    # Save with required timestamp format
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf_name = f"crossentropyloss_{ts}.pdf"
    plt.savefig(pdf_name, format="pdf", bbox_inches="tight")
    plt.close()

    print("Saved loss plot:", pdf_name)
    print("Final loss:", losses[-1])
    print("W1:", W1.shape, "W2:", W2.shape, "W3:", W3.shape, "W4:", W4.shape)


if __name__ == "__main__":
    main()
