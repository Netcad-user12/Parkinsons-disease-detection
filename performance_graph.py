import matplotlib.pyplot as plt

# Your actual results
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
values = [92.26, 87.63, 87.83, 87.73, 95.48]

plt.figure(figsize=(8, 5))

bars = plt.bar(metrics, values)

plt.title("Performance Metrics Comparison")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)

# Show values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.2f}",
        ha='center'
    )

plt.tight_layout()

# Save graph
plt.savefig("performance_metrics.png")

plt.show()