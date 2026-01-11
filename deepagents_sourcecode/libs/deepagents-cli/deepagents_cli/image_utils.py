"""클립보드에서 이미지 붙여넣기(paste)를 처리하는 유틸리티입니다.

Utilities for handling image paste from clipboard.
"""

import base64
import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


@dataclass
class ImageData:
    """Represents a pasted image with its base64 encoding."""

    base64_data: str
    format: str  # "png", "jpeg", etc.
    placeholder: str  # Display text like "[image 1]"

    def to_message_content(self) -> dict:
        """Convert to LangChain message content format.

        Returns:
            Dict with type and image_url for multimodal messages
        """
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{self.format};base64,{self.base64_data}"},
        }


def get_clipboard_image() -> ImageData | None:
    """Attempt to read an image from the system clipboard.

    Supports macOS via `pngpaste` or `osascript`.

    Returns:
        ImageData if an image is found, None otherwise
    """
    if sys.platform == "darwin":
        return _get_macos_clipboard_image()
    # Linux/Windows support could be added here
    return None


def _get_macos_clipboard_image() -> ImageData | None:
    """Get clipboard image on macOS using pngpaste or osascript.

    First tries pngpaste (faster if installed), then falls back to osascript.

    Returns:
        ImageData if an image is found, None otherwise
    """
    # Try pngpaste first (fast if installed)
    pngpaste_path = shutil.which("pngpaste")
    if pngpaste_path:
        try:
            result = subprocess.run(  # noqa: S603
                [pngpaste_path, "-"],
                capture_output=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout:
                # Successfully got PNG data
                try:
                    Image.open(io.BytesIO(result.stdout))  # Validate it's a real image
                except (UnidentifiedImageError, OSError):
                    pass  # Invalid image data
                else:
                    base64_data = base64.b64encode(result.stdout).decode("utf-8")
                    return ImageData(
                        base64_data=base64_data,
                        format="png",  # 'pngpaste -' always outputs PNG
                        placeholder="[image]",
                    )
        except subprocess.TimeoutExpired:
            pass  # pngpaste timed out

    # Fallback to osascript with temp file (built-in but slower)
    return _get_clipboard_via_osascript()


def _get_clipboard_via_osascript() -> ImageData | None:  # noqa: PLR0911
    """Get clipboard image via osascript using a temp file.

    osascript outputs data in a special format that can't be captured as raw binary,
    so we write to a temp file instead.

    Returns:
        ImageData if an image is found, None otherwise
    """
    osascript_path = shutil.which("osascript")
    if not osascript_path:
        return None

    # Create a temp file for the image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_file = Path(tmp.name)

    try:
        # First check if clipboard has PNG data
        check_result = subprocess.run(  # noqa: S603
            [osascript_path, "-e", "clipboard info"],
            capture_output=True,
            check=False,
            timeout=2,
            text=True,
        )

        if check_result.returncode != 0:
            return None

        # Check for PNG or TIFF in clipboard info
        clipboard_info = check_result.stdout.lower()
        if "pngf" not in clipboard_info and "tiff" not in clipboard_info:
            return None

        # Try to get PNG first, fall back to TIFF
        if "pngf" in clipboard_info:
            get_script = f"""
            set pngData to the clipboard as «class PNGf»
            set theFile to open for access POSIX file "{temp_file.as_posix()}" with write permission
            write pngData to theFile
            close access theFile
            return "success"
            """
        else:
            get_script = f"""
            set tiffData to the clipboard as TIFF picture
            set theFile to open for access POSIX file "{temp_file.as_posix()}" with write permission
            write tiffData to theFile
            close access theFile
            return "success"
            """

        result = subprocess.run(  # noqa: S603
            [osascript_path, "-e", get_script],
            capture_output=True,
            check=False,
            timeout=3,
            text=True,
        )

        if result.returncode != 0 or "success" not in result.stdout:
            return None

        # Check if file was created and has content
        if not temp_file.exists() or temp_file.stat().st_size == 0:
            return None

        # Read and validate the image
        image_data = temp_file.read_bytes()

        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to PNG if it's not already (e.g., if we got TIFF)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ImageData(
                base64_data=base64_data,
                format="png",
                placeholder="[image]",
            )
        except (UnidentifiedImageError, OSError):
            return None

    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        # Clean up temp file
        with contextlib.suppress(OSError):
            temp_file.unlink()


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def create_multimodal_content(text: str, images: list[ImageData]) -> list[dict]:
    """Create multimodal message content with text and images.

    Args:
        text: Text content of the message
        images: List of ImageData objects

    Returns:
        List of content blocks in LangChain format
    """
    content_blocks = []

    # Add text block
    if text.strip():
        content_blocks.append({"type": "text", "text": text})

    # Add image blocks
    content_blocks.extend([image.to_message_content() for image in images])

    return content_blocks
