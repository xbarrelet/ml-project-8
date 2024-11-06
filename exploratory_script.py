import os

from matplotlib import pyplot as plt

DATA_PATH = "resources/fruits-360_dataset/fruits-360/Test/"


def count_files_in_folders(data_path):
    label_counts = {}
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            file_count = len(os.listdir(label_path))
            label_counts[label] = file_count
    return label_counts


def plot_label_counts(label_counts):
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    mean_value = sum(counts) / len(counts)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.ylabel('Number of Images')
    plt.title('Number of Files per Label')
    plt.xticks(rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    label_counts = count_files_in_folders(DATA_PATH)
    plot_label_counts(label_counts)
