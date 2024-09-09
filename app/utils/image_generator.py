from PIL import Image, ImageDraw, ImageFont
import os
import textwrap


def generate_title_image(output_path: str, title: str):
    """
    Generate a title image with the given title and save it to the specified output path.

    Args:
        output_path (str): The path where the generated title image will be saved.
        title (str): The title to be displayed on the image.

    Raises:
        ValueError: If the length of the title exceeds 125 characters.

    Returns:
        None
    """
    if len(title) > 175:
        raise ValueError(
            "Title is too long. Max 175 Characters. This is due to restrictions on the title image template."
        )

    title = title.strip()

        
    try:
        template_path = os.path.join("assets", "reddit_title_template.png")
        image = Image.open(template_path)
    except Exception as e:
        raise ValueError(f"Error loading title template: {e}")

    draw = ImageDraw.Draw(image)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the correct font path
    font_path = os.path.join(current_dir, '..', '..', 'assets', 'Ubuntu-Medium.ttf')

    try:
        font = ImageFont.truetype(font_path, 38)
    except Exception as e:
        raise ValueError(f"Error loading font: {e}")

    x = 365
    y = 745
    if len(title) < 85:
        y = 795

    line_height = 45

    color = "rgb(0,0,0)"

    lines = textwrap.wrap(title, width=46)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        line_position = (
            x,
            y,
        )

        draw.text(line_position, line, fill=color, font=font)
        y += line_height

    image.save(output_path)


if __name__ == "__main__":
    """
    Some utility functions to generate title images for debugging purposes.
    """
    generate_title_image(
        "tmp/test/output_path.png",
        "This is a very long title that needs to be wrapped because it exceeds the maximum width. but if its even longer what happens.",
    )

    generate_title_image(
        "tmp/test/output_path2.png",
        "This is a shorter title that doesn't need to be wrapped.",
    )

    generate_title_image(
        "tmp/test/output_path3.png",
        "This is shorter",
    )
