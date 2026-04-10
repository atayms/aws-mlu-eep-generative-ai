import os
import json
import base64
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile
import textwrap
import boto3
from botocore.exceptions import ClientError
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_images(dir_name):

    # Get a list of all image file paths in the folder
    image_paths = [os.path.join(dir_name, file) for file in os.listdir(dir_name) if file.endswith(('.jpg', '.png', 'jpeg'))]

    # Calculate the number of rows and columns needed for the subplots
    num_images = len(image_paths)
    num_cols = 5  # Set the desired number of columns
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the required number of rows

    # Create a figure and axis objects for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axes = axes.ravel()  # Flatten the axes object to make indexing easier

    # Loop through the images and display them in the subplots
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')

    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Adjust subplot spacing and display the figure
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def truncate_text(text, max_width, font_size, dpi):
    """Truncate text to fit within a given width."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi)
    t = ax.text(0, 0, text, fontsize=font_size)
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)

    if bbox.width > max_width:
        ratio = max_width / bbox.width
        max_chars = int(len(text) * ratio)
        return text[:max_chars-3] + '...'
    return text

def plot_results(df):
    plt.style.use('seaborn-v0_8')
    # Calculate the number of rows and columns for the subplot grid
    n_images = len(df)
    n_cols = min(5, n_images)  # Max 5 columns
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12*n_rows), dpi=100)
    fig.tight_layout(pad=1.0)

    # Flatten the axes array if it's 2D
    if n_rows > 1:
        axes = axes.flatten()

    # Iterate through the DataFrame and plot each image
    for i, (_, row) in enumerate(df.iterrows()):
        if i < len(axes):
            # Read the image
            img = Image.open(row['image_path'])
            
            # Plot the image
            if n_rows == 1:
                ax = axes[i] if n_cols > 1 else axes
            else:
                ax = axes[i]
            
            ax.imshow(img)
            ax.axis('off')
            
            # Get the width of the image in pixels
            img_width = ax.get_window_extent().width
            
            # Truncate and wrap the text
            truncated_text = truncate_text(row['text'], img_width, 8, fig.dpi)
            wrapped_text = textwrap.fill(truncated_text, width=40)
            
            # Set the title (text)
            ax.set_title(wrapped_text, fontsize=18, wrap=True, y=1.05)

    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

    
def plot_scatter_plot(title, full_data, new_data):
    # Set the style to a dark background
    plt.style.use('dark_background')

    # Create a new figure with a specific background color
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1C1C1C')
    ax.set_facecolor('#1C1C1C')
    
    colors = ['#FF6B6B', '#4ECDC4', '#98FB98', '#FFD700', '#FF69B4', '#90EE90', '#E6E6FA', '#ADD8E6', '#FF00FF', '#FFA07A']
    colors_index = 0
    
    for key in full_data:
        ax.scatter(full_data[key][:, 0], full_data[key][:, 1], c=colors[colors_index], label=key, s=75, edgecolors='white')
        colors_index +=1
    
    for key in new_data:
        ax.scatter(new_data[key][:, 0], new_data[key][:, 1], c=colors[colors_index], label=key, s=200, edgecolors='white')
        colors_index +=1
    
#     ax.scatter(reduced_cars_embeddings[:, 0], reduced_cars_embeddings[:, 1], c='#4ECDC4', label='Cars', s=75, edgecolors='white')
    
    
#     ax.scatter(reduced_new_car_embeddings[:, 0], reduced_new_car_embeddings[:, 1], c='#FFA07A', label='New Car Image', s=150, edgecolors='white')
#     ax.scatter(reduced_new_cat_embeddings[:, 0], reduced_new_cat_embeddings[:, 1], c='#98FB98', label='New Cat Image', s=150, edgecolors='white')
#     ax.scatter(reduced_car_text_embeddings[:, 0], reduced_car_text_embeddings[:, 1], c='#FFD700', label='New Car Text', s=150, edgecolors='white')
#     ax.scatter(reduced_cat_text_embeddings[:, 0], reduced_cat_text_embeddings[:, 1], c='#FF69B4', label='New Cat Text', s=150, edgecolors='white')

    # Customize labels and title
    ax.set_xlabel('Feature 1', fontsize=18, color='white')
    ax.set_ylabel('Feature 2', fontsize=18, color='white')
    ax.set_title(title, fontsize=20, color='white', fontweight='bold')

    # Customize legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor='#2F2F2F', edgecolor='none', fontsize=14)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    
def reduce_dimensionality(array, n_components=2):
    from sklearn.decomposition import PCA

    # Create a PCA object with 2 components
    pca = PCA(n_components=n_components)

    # Fit the PCA model to your data and transform it
    return pca.fit(array)


def process_images(folder_path):
    # Set the maximum file size (in bytes) and resolution
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_RESOLUTION = (720, 720)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.png', 'jpeg')):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, filename)

            # Open the image
            with Image.open(image_path) as img:
                # Check if the image exceeds the maximum file size
                if os.path.getsize(image_path) > MAX_FILE_SIZE:
                    # Resize the image while preserving the aspect ratio
                    img.thumbnail(MAX_RESOLUTION)

                    # Save the resized image
                    img.save(image_path)
                    print(f"Resized {filename} due to file size.")

                # Check if the image exceeds the maximum resolution
                elif img.size > MAX_RESOLUTION:
                    # Resize the image while preserving the aspect ratio
                    img.thumbnail(MAX_RESOLUTION)

                    # Save the resized image
                    img.save(image_path)
                    print(f"Resized {filename} due to resolution.")

                else:
                    print(f"{filename} is already compliant.")
    
def pdf2imgs(pdf_path, pdf_pages_dir="data/lab4a/Accessibility/pdf_pages"):
    """
    Convert a PDF file to individual PNG images for each page.

    Args:
        pdf_path (str): The path to the PDF file.
        pdf_pages_dir (str, optional): The directory to save the PNG images. Defaults to "data/lab4a/Accessibility/pdf_pages".

    Returns:
        str: The path to the directory containing the PNG images.
    """
    import pypdfium2 as pdfium
    # Open the PDF document
    pdf = pdfium.PdfDocument(pdf_path)

    # Create the directory to save the PNG images if it doesn't exist
    os.makedirs(pdf_pages_dir, exist_ok=True)

    # Get the resolution of the first page to determine the scale factor
    resolution = pdf.get_page(0).render().to_numpy().shape
    scale = 1 if max(resolution) >= 1620 else 300 / 72  # Scale factor based on resolution

    # Get the number of pages in the PDF
    n_pages = len(pdf)

    # Loop through each page and save as a PNG image
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        pil_image = page.render(
            scale=scale,
            rotation=0,
            crop=(0, 0, 0, 0),
            may_draw_forms=False,
            fill_color=(255, 255, 255, 255),
            draw_annots=False,
            grayscale=False,
        ).to_pil()
        image_path = os.path.join(pdf_pages_dir, f"page_{page_number:03d}.png")
        pil_image.save(image_path)

    return pdf_pages_dir
                  
def invoke_nova_lite_multimodal(prompt, images, image_types):
    """
    Invoke the Nova Lite multimodal model from Anthropic using AWS Bedrock runtime.

    Args:
        prompt (str): The text prompt to provide to the model.
        images (list): A list of base64-encoded image data.
        image_types (list): A list of MIME types corresponding to the images.

    Returns:
        str: The model's response text.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    
    # Initialize the Amazon Bedrock runtime client
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    # select model
    model_id = "amazon.nova-lite-v1:0"
    # Build the message to call Converse API
    
    message_content = []
    for img, img_type in zip(images, image_types):
        message_content.append({"image": {"format": img_type, "source": {"bytes": img}}},)    
    message_content.append({"text": prompt})

    messages = [
            {
                "role": "user",
                "content": message_content,
            }
      ]
    inf_params = {"maxTokens": 2048, "topP": 1.0, "temperature": 0.2}

    try:
        response = client.converse(
            modelId=model_id, messages=messages, inferenceConfig=inf_params
        )
        # Process and return the response
        result = json.dumps(response, indent=2)

        return response["output"]["message"]["content"][0]["text"]
    except ClientError as err:
        logger.error(
            "Couldn't invoke Nova Lite %s model. Here's why: %s: %s",
            model_id.capitalize(),
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise

def get_base64_encoded_image(image_paths):
    """
    Encode one or more image files or URLs into base64 strings.

    Args:
        image_paths (str or list): A single file path/URL or a list of file paths/URLs.

    Returns:
        tuple: A tuple containing two lists:
            - A list of base64-encoded image strings.
            - A list of corresponding image types.

    Raises:
        ValueError: If an unsupported image format is encountered.
    """
    # Convert input to list if it's a single string
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images, image_types = [], []

    # Iterate over the image paths/URLs
    for path in image_paths:
        # Check if the path is a URL
        if path.startswith("https://"):
            url_content = requests.get(path).content
            base64_encoded_data = base64.b64encode(url_content)
            base64_string = base64_encoded_data.decode('utf-8')
        # Otherwise, assume it's a file path
        else:
            with open(path, "rb") as image_file:
                binary_data = image_file.read()
                base64_encoded_data = base64.b64encode(binary_data)
                base64_string = base64_encoded_data.decode('utf-8')

        # Determine the image type based on the file extension
        path = path.lower()
        if path.endswith('.png'):
            image_type = 'image/png'
        elif path.endswith('.jpg') or path.endswith('.jpeg'):
            image_type = 'image/jpeg'
        elif path.endswith('.webp'):
            image_type = 'image/webp'
        elif path.endswith('.gif'):
            image_type = 'image/gif'
        else:
            raise ValueError("Only 'jpeg', 'png', 'webp', and 'gif' image formats are currently supported")

        images.append(base64_string)
        image_types.append(image_type)

    return images, image_types

def _detect_image_type_from_bytes(binary_data):
    """Detect image format from file magic bytes. Returns type string or None.
    Returns 'invalid' if the content is clearly not an image (HTML, XML, text, etc.)."""
    if not binary_data or len(binary_data) < 8:
        return 'invalid'
    if binary_data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    elif binary_data[:2] == b'\xff\xd8':
        return 'jpeg'
    elif binary_data[:4] == b'RIFF' and len(binary_data) > 12 and binary_data[8:12] == b'WEBP':
        return 'webp'
    elif binary_data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    # Check if content is text-based (HTML, XML, JSON error pages, etc.)
    stripped = binary_data.lstrip()[:500]
    try:
        text_start = stripped.decode('utf-8', errors='ignore').lower()
        if any(marker in text_start for marker in ['<!doctype', '<html', '<head', '<body', '<?xml', '<error', '<message', '{"error']):
            return 'invalid'
    except Exception:
        pass
    return None


def prepare_image(image_paths):
    """
    Prepare one or more image files or URLs into binary format.

    Args:
        image_paths (str or list): A single file path/URL or a list of file paths/URLs.

    Returns:
        tuple: A tuple containing two lists:
            - A list of binary images.
            - A list of corresponding image types.

    Raises:
        ValueError: If an unsupported image format is encountered.
    """
    # Convert input to list if it's a single string
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    images, image_types = [], []

    # Iterate over the image paths/URLs
    for path in image_paths:
        # Check if the path is a URL
        if path.startswith("https://") or path.startswith("http://"):
            response = requests.get(path, timeout=10)
            binary_data = response.content

            # Detect actual format from the binary content (magic bytes)
            image_type = _detect_image_type_from_bytes(binary_data)
            if image_type == 'invalid':
                # Try to show a snippet of what was returned for debugging
                snippet = binary_data[:200].decode('utf-8', errors='replace')
                raise ValueError(
                    f"URL did not return a valid image (status {response.status_code}): {path}\n"
                    f"Content starts with: {snippet}"
                )
            if image_type is None:
                # Fall back to URL file extension
                url_path = path.split("?")[0].lower()
                if url_path.endswith('.png'):
                    image_type = 'png'
                elif url_path.endswith(('.jpg', '.jpeg')):
                    image_type = 'jpeg'
                elif url_path.endswith('.webp'):
                    image_type = 'webp'
                elif url_path.endswith('.gif'):
                    image_type = 'gif'
                else:
                    image_type = 'jpeg'  # Default for unknown URL types

        # Otherwise, assume it's a file path
        else:
            with open(path, "rb") as image_file:
                binary_data = image_file.read()

            # Detect actual format from the binary content first
            image_type = _detect_image_type_from_bytes(binary_data)
            if image_type == 'invalid':
                raise ValueError(f"File does not contain a valid image: {path}")
            if image_type is None:
                # Fall back to file extension
                if path.endswith('.png'):
                    image_type = 'png'
                elif path.endswith(('.jpg', '.jpeg')):
                    image_type = 'jpeg'
                elif path.endswith('.webp'):
                    image_type = 'webp'
                elif path.endswith('.gif'):
                    image_type = 'gif'
                else:
                    raise ValueError(
                        "Only 'jpeg', 'png', 'webp', and 'gif' image formats are currently supported"
                    )

        images.append(binary_data)
        image_types.append(image_type)

    return images, image_types