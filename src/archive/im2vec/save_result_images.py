import os
import cv2
import glob
import matplotlib.pyplot as plt
from im2vec.natural_keys import natural_keys
from archive.im2vec.paths import result_img_path, result_data_path, stored_img_path


def save_result_images():
    result_files = None
    results = None
    accuracies = None
    image = None
    result_file = None
    os.makedirs(result_img_path, exist_ok=True)
    result_files = sorted(glob.glob(f"{result_data_path}/*"), key=natural_keys)
    for i in range(len(result_files)):
        results = []
        accuracies = []
        with open(result_files[i], "r") as f:
            for line in f:
                results.append(f"{stored_img_path}/{line.split(',')[1]}")
                accuracies.append(line.split(",")[2])

        plt.figure(figsize=(80, 28))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0)
        image = list(range(1, 12))
        for j in range(1, 11):
            image[j] = cv2.imread(results[j])
            image[j] = cv2.cvtColor(image[j], cv2.COLOR_BGR2RGB)
            plt.subplot(2, 5, j)
            plt.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                bottom=False,
                left=False,
                right=False,
                top=False,
            )
            plt.title(round(float(accuracies[j]), 4), {"fontsize": 80})
            plt.imshow(image[j])
        result_file = f"{result_img_path}/result_image{str(i+1).zfill(3)}.png"
        plt.savefig(result_file, transparent=True)
        plt.close()
        print(f"隣接画像を '{result_file}' に保存しました。")
