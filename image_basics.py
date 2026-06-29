import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    image_path = Path("data/sample.jpg")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # OpenCV şəkli BGR formatında oxuyur
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print("Şəkil tapılmadı. data/sample.jpg yolunu yoxla.")
        return

    print("Image type:", type(image_bgr))
    print("BGR shape:", image_bgr.shape)
    print("BGR dtype:", image_bgr.dtype)
    print("First pixel BGR:", image_bgr[0, 0])

    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # RGB -> GRAY
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    print("RGB shape:", image_rgb.shape)
    print("Gray shape:", image_gray.shape)
    print("Gray dtype:", image_gray.dtype)

    # Nəticələri yadda saxla
    cv2.imwrite(str(output_dir / "gray_image.jpg"), image_gray)

    # RGB-ni OpenCV ilə saxlamaq üçün əvvəl BGR-ə qaytarmaq lazımdır
    rgb_to_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "rgb_image.jpg"), rgb_to_bgr)

    # Matplotlib ilə göstər
    plt.imshow(image_rgb)
    plt.title("RGB Image")
    plt.axis("off")
    plt.show()

    plt.imshow(image_gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()