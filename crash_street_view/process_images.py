import base64
import cv2
import numpy as np
import requests

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyAcklfCnR8BAJ8EAmRABgDoNdCJIfgpWA0"  # Replace with your actual API key
LAT = 40.7128  # Example: New York City
LON = -74.0060

# Adjust these for optimal stitching
# More headings mean more images, better overlap, but more requests and processing
HEADINGS = [0, 45, 90, 135, 180, 225, 270, 315]
FOV_VALUE = 90 # Field of View
IMAGE_SIZE = "640x640" # Max allowed by Google Street View Static API for free tier is typically 640x640

def get_street_view_image_metadata(latitude, longitude):
    """
    Fetches a single Google Street View image and returns its metadata.
    """
    metadata_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={latitude},{longitude}&key={GOOGLE_API_KEY}"
    try:
        response = requests.get(metadata_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def get_street_view_image_bytes(latitude, longitude, heading, size="640x480", fov=90, pitch=0):
    """
    Fetches a single Google Street View image and returns its bytes.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{latitude},{longitude}",
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "key": GOOGLE_API_KEY
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            return response.content
        else:
            print(f"Error: API did not return an image for heading {heading}. Content-Type: {content_type}. Response: {response.text[:200]}...") # Print first 200 chars
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Street View image for heading {heading}: {e}")
        return None

def stitch_street_view_images_from_lat_lon(latitude, longitude, headings, image_size, fov_value, output_format=".jpg"):
    """
    Downloads overlapping Street View images as bytes, stitches them,
    and returns the stitched panorama as bytes in the specified format.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        headings (list[int]): List of compass headings for images.
        image_size (str): Desired size of individual images (e.g., "640x480").
        fov_value (int): Field of view for individual images.
        output_format (str): Format for the output panoramic image (e.g., ".jpg", ".png").

    Returns:
        bytes: The stitched panoramic image in the specified format, or None if stitching fails.
    """
    raw_image_bytes_list = []
    print(f"Fetching {len(headings)} Street View images as bytes for {latitude},{longitude}...")
    for heading in headings:
        img_bytes = get_street_view_image_bytes(latitude, longitude, heading, size=image_size, fov=fov_value)
        if img_bytes:
            raw_image_bytes_list.append(img_bytes)
        else:
            print(f"Skipping heading {heading} due to download error.")

    if not raw_image_bytes_list:
        print("No images successfully downloaded for stitching.")
        return None

    # Decode bytes to OpenCV (NumPy array) format
    images_np = []
    for img_bytes in raw_image_bytes_list:
        # Convert bytes to a numpy array, then decode
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_decoded is not None:
            images_np.append(img_decoded)
        else:
            print("Warning: Could not decode an image. Skipping.")

    if len(images_np) < 2:
        print("Not enough valid images decoded for stitching.")
        return None

    print(f"Attempting to stitch {len(images_np)} images...")

    # Create a stitcher object
    if cv2.__version__.startswith('3.'):
        stitcher = cv2.createStitcher()
    else:
        stitcher = cv2.Stitcher_create()

    # Stitch the images
    status, panorama_np = stitcher.stitch(images_np)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        # Encode the stitched NumPy array back into bytes
        is_success, buffer = cv2.imencode(output_format, panorama_np)
        if is_success:
            image_bytes = bytes(buffer)
            return image_bytes
        else:
            print(f"Failed to encode stitched image to {output_format} bytes.")
            return None
    else:
        status_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images or better overlap for stitching.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed. Check image overlap and quality.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed. Very little overlap or highly distorted images.",
        }
        error_message = status_messages.get(status, f"Stitching failed with unknown status code: {status}")
        print(f"Stitching failed: {error_message}")
        return None

def encode_image_to_base64_data_uri(image_bytes, format="jpeg"):
    """
    Encodes image bytes to a Base64 data URI for embedding in HTML.
    """
    base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/{format};base64,{base64_encoded_image}"
    
# --- Main execution ---
if __name__ == "__main__":
    print("Starting panorama generation...")
    panoramic_image_bytes = stitch_street_view_images_from_lat_lon(
        LAT, LON, HEADINGS, IMAGE_SIZE, FOV_VALUE, output_format=".png"
    )

    if panoramic_image_bytes:
        print(f"Successfully generated panoramic image bytes. Size: {len(panoramic_image_bytes)} bytes")

        # Example: If you want to save it to a file temporarily to verify
        # In a real application, you would pass these bytes to another function,
        # return them from an API endpoint, or display them in memory.
        temp_output_filename = "in_memory_stitched_panorama.jpg"
        with open(temp_output_filename, "wb") as f:
            f.write(panoramic_image_bytes)
        print(f"Temporary saved for verification: {temp_output_filename}")

        # Optional: Display the image using OpenCV (if you have a display environment)
        # You'd decode it again from bytes for display
        np_arr = np.frombuffer(panoramic_image_bytes, np.uint8)
        display_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if display_img is not None:
            cv2.imshow("Generated Panoramic Image", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        print("Failed to generate panoramic image bytes.")