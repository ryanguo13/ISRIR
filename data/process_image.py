from PIL import Image, ImageFilter
import os

def process_image(image_path, output_dir="."):
    """
    Resizes an image to 128x128 and 16x16, and blurs the 16x16 image.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output images.
    """
    try:
        img = Image.open(image_path)

        # Resize to 128x128
        img_128 = img.resize((128, 128))
        output_path_128 = os.path.join(output_dir, "prettygirl_128x128.png")
        img_128.save(output_path_128)
        print(f"Saved 128x128 image to: {output_path_128}")

        # Resize to 16x16
        img_16 = img.resize((16, 16))
        output_path_16 = os.path.join(output_dir, "prettygirl_16x16.png")
        img_16.save(output_path_16)
        print(f"Saved 16x16 image to: {output_path_16}")

        # Blur the 16x16 image
        img_16_blurred = img_16.filter(ImageFilter.GaussianBlur(radius=1)) # Using a radius of 1 for blur
        output_path_16_blurred = os.path.join(output_dir, "prettygirl_16x16_blurred.png")
        img_16_blurred.save(output_path_16_blurred)
        print(f"Saved 16x16 blurred image to: {output_path_16_blurred}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_image = "data/prettygirl.png"
    process_image(input_image)
