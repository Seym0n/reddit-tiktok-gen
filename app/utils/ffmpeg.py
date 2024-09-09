import ffmpeg
import math
import os
import re

from typing import cast, List, Tuple

from app.config import ffmpeg_config

from app.utils.logger import log


class FFMpegProcessingError(Exception):
    def __init__(self, message, stderr=None):
        super().__init__(message)
        self.stderr = stderr


def resize_video(video_path: str, output_path: str, config_preset="default"):
    """
    Resize a video to maintain a 16:9 aspect ratio on height.

    Args:
    video_path (str): The path to the input video file.
    output_file_path (str): The path to the output resized video file.
    config_preset (str): The configuration preset to use for FFmpeg commands.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Resizing video...")
    log.debug("Video path: %s", video_path)
    log.debug("Output file path: %s", output_path)

    settings = ffmpeg_config.get(config_preset, ffmpeg_config["default"])
    log.debug("FFmpeg settings: %s", settings)

    width, height = get_video_dimensions(video_path)
    log.debug("Video dimensions: %dx%d", width, height)
    target_width = min(width, height * 9 // 16)
    target_height = height

    try:
        (
            ffmpeg.input(video_path)
            .filter("crop", target_width, target_height)
            .output(
                output_path, vcodec="libx264", acodec="copy", preset=settings["preset"]
            )
            .global_args(*settings["global_args"])
            .run(
                overwrite_output=True,
                capture_stdout=settings["capture_stdout"],
                capture_stderr=settings["capture_stderr"],
            )
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during resize_video ffmpeg command", stderr=e.stderr
        )


def split_video_at_time(
    video_path: str,
    start_time: str,
    duration: float,
    output_path: str,
    config_preset="default",
):
    """
    Split a video at a specific time and save the segment as a new video file.

    Args:
    video_path (str): The path to the input video file.
    start_time (str): The time in HH:MM:SS format to start the split.
    duration (float): The duration of the segment in seconds.
    output_path (str): The path to save the output video file.
    config_preset (str): The configuration preset to use for FFmpeg commands.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Splitting video at time...")
    log.debug("Video path: %s", video_path)
    log.debug("Start time: %s", start_time)
    log.debug("Duration: %s", duration)
    log.debug("Output path: %s", output_path)

    settings = ffmpeg_config.get(config_preset, ffmpeg_config["default"])
    log.debug("FFmpeg settings: %s", settings)

    try:
        (
            ffmpeg.input(video_path, ss=start_time)
            .output(output_path, t=duration, c="copy")
            .global_args("-loglevel", "error")
            .global_args(*settings["global_args"])
            .run(
                overwrite_output=True,
                capture_stdout=settings["capture_stdout"],
                capture_stderr=settings["capture_stderr"],
            )
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during split_video_at_time ffmpeg command", stderr=e.stderr
        )


def split_video(video_path: str, duration: float, output_pattern_path: str):
    """
    Split a video into multiple segments of specified duration.
    Note: This has not been tested, this is just a utility function, for running as a script.

    Args:
    video_path (str): The path to the input video file.
    duration (int): The duration of each segment in seconds.
    output_pattern (str): The output filename pattern with a placeholder for the segment number. ie this/is/a/path/output%01d.mp4

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Splitting video...")
    log.debug("Video path: %s", video_path)
    log.debug("Duration: %d", duration)
    log.debug("Output pattern: %s", output_pattern_path)

    try:
        (
            ffmpeg.input(video_path)
            .output(
                output_pattern_path,
                format="segment",
                segment_time=duration,
                c="copy",
                reset_timestamps=1,
                map=0,
            )
            .global_args("-loglevel", "error")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during split_video ffmpeg command", stderr=e.stderr
        )


def loop_video_to_audio(
    audio_duration: float, video_path: str, output_path: str, config_preset="default"
):
    """
    Loop a video to match the specified audio duration and save it as a new video file.

    Args:
    audio_duration (float): The duration of the audio in seconds.
    video_path (str): The path to the input video file.
    output_video_path (str): The path to save the output video file.
    config_preset (str): The configuration preset to use for FFmpeg commands.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Looping video to audio duration...")
    log.debug("Audio duration: %s", audio_duration)
    log.debug("Video path: %s", video_path)
    log.debug("Output video path: %s", output_path)

    settings = ffmpeg_config.get(config_preset, ffmpeg_config["default"])
    log.debug("FFmpeg settings: %s", settings)

    video_duration = get_video_duration(video_path)
    log.debug("Video duration: %s", video_duration)
    number_of_repeats = math.ceil(audio_duration / video_duration)

    log.info(f"Looping video {number_of_repeats} times.")

    try:
        (
            ffmpeg.input(video_path, stream_loop=number_of_repeats)
            .output(
                output_path,
                vf=f"trim=duration={audio_duration}",
                acodec="copy",
                preset=settings["preset"],
            )
            .global_args(*settings["global_args"])
            .run(
                overwrite_output=True,
                capture_stdout=settings["capture_stdout"],
                capture_stderr=settings["capture_stderr"],
            )
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during loop_video_to_audio ffmpeg command", stderr=e.stderr
        )


def concatenate_audios(audio_path1: str, audio_path2: str, output_path: str):
    """
    Concatenate two audio files using ffmpeg-python.

    Args:
    audio_path1 (str): Path to the first audio file.
    audio_path2 (str): Path to the second audio file.
    output_path (str): Path where the concatenated output should be saved.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Concatenating audio files...")
    log.debug("Audio path 1: %s", audio_path1)
    log.debug("Audio path 2: %s", audio_path2)
    log.debug("Output path: %s", output_path)

    input_str = f"concat:{audio_path1}|{audio_path2}"
    try:
        (
            ffmpeg.input(input_str)
            .output(output_path, codec="copy")
            .global_args("-loglevel", "error")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        log.error(f"Error during ffmpeg command: {e.stderr}")
        raise


def overlay_image_on_video(
    video_path: str,
    image_path: str,
    duration: int,
    output_path: str,
    config_preset="default",
):
    """
    Overlay an image on a video

    Args:
    video_path (str): Path to the input video file.
    image_path (str): Path to the image file to overlay.
    duration (int): Duration in seconds for which the image should be visible on the video.
    output_path (str): Path to the output video file.
    config_preset (str): The configuration preset to use for FFmpeg commands.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Overlaying image on video...")
    log.debug("Video path: %s", video_path)
    log.debug("Image path: %s", image_path)
    log.debug("Duration: %s", duration)
    log.debug("Output path: %s", output_path)

    settings = ffmpeg_config.get(config_preset, ffmpeg_config["default"])
    log.debug("FFmpeg settings: %s", settings)

    try:
        input_video = ffmpeg.input(video_path)
        input_image = ffmpeg.input(image_path)

        overlay_filter = (
            ffmpeg.filter_(
                [input_video, input_image],
                "overlay",
                x="(W-w)/2",
                y="(H-h)/2",
                enable=f"between(t,0,{duration})",
            )
            .output(
                output_path, vcodec="libx264", acodec="copy", preset=settings["preset"]
            )
            .global_args(*settings["global_args"])
            .run(
                overwrite_output=True,
                capture_stdout=settings["capture_stdout"],
                capture_stderr=settings["capture_stderr"],
            )
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during overlay_image_on_video ffmpeg command", stderr=e.stderr
        )


def resize_image(image_path: str, target_width: int, output_path: str):
    """
    Resize an image to a specified width, keeping the aspect ratio.

    Args:
    image_path (str): Path to the input image file.
    target_width (int): Target width for the resized image.
    output_path (str): Output path for the new image file.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """
    log.info("Resizing image...")

    log.debug("Image path: %s", image_path)
    log.debug("Target width: %s", target_width)
    log.debug("Output path: %s", output_path)

    # TODO validate why we do this (this line comes from my original java implementation, but I don't remember why I did it)
    target_width += 200

    try:
        (
            ffmpeg.input(image_path)
            .filter("scale", target_width, -1)  # -1 in scale maintains the aspect ratio
            .output(output_path)
            .global_args("-loglevel", "error")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during resize_image ffmpeg command", stderr=e.stderr
        )


def buffer_audio(audio_path: str, pos: str, duration: float, output_path: str):
    """
    Buffer audio by adding silence at the start or end of the audio file.

    Args:
    audio_path (str): Path to the audio file.
    pos (str): Position to add the buffer ('START' or 'END').
    duration (float): Duration of the silence to add in seconds.
    output_path (str): Output path for the new image file.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """
    log.info("Buffering audio...")
    log.debug("Audio path: %s", audio_path)
    log.debug("Position: %s", pos)
    log.debug("Duration: %s", duration)
    log.debug("Output path: %s", output_path)

    # Configure FFmpeg command based on the position of buffering
    if pos == "START":
        log.info("Buffering audio at start...")
        filter_complex = f"adelay={duration}s:all=true"
    elif pos == "END":
        log.info("Buffering audio at end...")
        filter_complex = f"apad=pad_dur={duration}s"
    else:
        raise ValueError("Invalid position. Use 'START' or 'END'.")

    try:
        (
            ffmpeg.input(audio_path)
            .output(output_path, af=filter_complex)
            .global_args("-loglevel", "error")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during buffer_audio ffmpeg command", stderr=e.stderr
        )


def parse_srt_time(time_str: str) -> float:
    """Convert SRT time format to seconds."""
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def get_video_dimensions(video_path):
    """Get the dimensions of the input video."""
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def create_animated_ass(srt_path: str, ass_path: str, video_path: str):
    """Convert SRT to ASS with improved font, stronger outline, and bounce effect."""
    width, height = get_video_dimensions(video_path)
    fontsize = (int)((width / 606) * 60)

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Alignment, Outline, Shadow
Style: Default,Mont,{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,5,6,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def time_to_ass(time_str):
        h, m, s = time_str.replace(',', '.').split(':')
        return f"{int(h):01d}:{int(m):02d}:{float(s):.2f}"

    with open(srt_path, 'r', encoding='utf-8') as srt_file, open(ass_path, 'w', encoding='utf-8') as ass_file:
        ass_file.write(ass_header)
        
        srt_content = srt_file.read()
        subtitle_blocks = re.split(r'\n\n+', srt_content.strip())
        
        for i, block in enumerate(subtitle_blocks):
            lines = block.split('\n')
            if len(lines) >= 3:
                index = lines[0]
                timing = lines[1]
                text = ' '.join(lines[2:])  # Join all text lines
                
                start_time, end_time = timing.split(' --> ')
                start_time_ass = time_to_ass(start_time)
                end_time_ass = time_to_ass(end_time)
                
                # Calculate duration for timing the bounce effect
                start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.replace(',', '.').split(':'))))
                end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.replace(',', '.').split(':'))))
                duration = end_seconds - start_seconds
                
                # Bounce effect with fade in/out
                bounce_duration = min(duration * 0.3, 0.6)  # 30% of duration or max 600ms
                fade_duration = min(duration * 0.1, 0.2)  # 10% of duration or max 200ms for fade
                
                animation = (
                    "{\\fad(%(fade)d,%(fade)d)"  # Fade in and out
                    "\\t(0,%(bounce)d,\\fscx120\\fscy120)"  # Grow to 120%
                    "\\t(%(bounce)d,%(bounce_end)d,\\fscx100\\fscy100)}"  # Shrink back to 100%
                ) % {
                    'fade': int(fade_duration * 1000),
                    'bounce': int(bounce_duration * 1000),
                    'bounce_end': int(bounce_duration * 1500)  # 1.5x bounce duration for the shrink
                }
                
                ass_line = f"Dialogue: 0,{start_time_ass},{end_time_ass},Default,,0,0,0,,{animation}{text}\n"
                ass_file.write(ass_line)
                
                print(f"Subtitle {index}: {start_time_ass} -> {end_time_ass}: {text}")  # Debug print

    print(f"ASS file created at: {ass_path}")  # Debug print

def embed_srt_and_audio(
    video_path: str,
    audio_path: str,
    srt_path: str,
    output_path: str,
    config_preset="default",
):
    """
    Embeds animated subtitles and audio into a video file using FFmpeg.

    Args:
        video_path (str): The path to the input video file.
        audio_path (str): The path to the input audio file.
        srt_path (str): The path to the input subtitle file in SRT format.
        output_path (str): The path to save the output video file.
        config_preset (str): The configuration preset to use for FFmpeg commands.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """

    log.info("Embedding animated subtitles and audio...")
    log.debug("Video path: %s", video_path)
    log.debug("Audio path: %s", audio_path)
    log.debug("SRT path: %s", srt_path)
    log.debug("Output path: %s", output_path)

    settings = ffmpeg_config.get(config_preset, ffmpeg_config["default"])
    log.debug("FFmpeg settings: %s", settings)

    # Create ASS file with animated subtitles
    ass_path = os.path.splitext(srt_path)[0] + '.ass'
    create_animated_ass(srt_path, ass_path, video_path)

    subtitles_filter = f"ass={ass_path}"
    
    try:
        (
            ffmpeg.input(video_path)
            .output(
                ffmpeg.input(audio_path),
                output_path,
                vf=subtitles_filter,
                vcodec="libx264",
                acodec="aac",
                # Quality optimizations
                audio_bitrate="192k",
                crf=20,
                preset=settings["preset"],
            )
            .global_args(*settings["global_args"])
            .run(
                overwrite_output=True,
                capture_stdout=settings["capture_stdout"],
                capture_stderr=settings["capture_stderr"],
            )
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during embed_srt_and_audio ffmpeg command", stderr=e.stderr
        )
    finally:
        # Clean up the temporary ASS file
        if os.path.exists(ass_path):
            os.remove(ass_path)

def delay_srt(srt_path: str, delay: float, output_path: str):
    """
    Delay an SRT subtitle file by a certain number of seconds using ffmpeg-python.

    Args:
    srt_path (str): Path to the SRT file.
    delay (float): Amount of delay to add in seconds.
    output_path (str): Path where the delayed SRT file will be saved.

    Raises:
        FFMpegProcessingError: If an error occurs during the FFmpeg command execution.
    """
    log.info(f"Delaying srt by {delay} seconds.")
    log.debug("SRT path: %s", srt_path)
    log.debug("Output path: %s", output_path)

    latency = 0.1  # additional latency to add in seconds
    total_delay = delay + latency

    try:
        (
            ffmpeg.input(srt_path, itsoffset=total_delay)
            .output(output_path, codec="copy")
            .global_args("-loglevel", "error")
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        # log.error(f"Error during ffmpeg command: {e.stderr}")
        # raise
        raise FFMpegProcessingError(
            "Error during delay_srt ffmpeg command", stderr=e.stderr
        )


def get_video_duration(video_path: str) -> float:
    """Returns the duration of the video in seconds."""

    log.info("Getting video duration...")
    log.debug("Video path: %s", video_path)

    probe = ffmpeg.probe(video_path)
    duration = float(
        next(stream for stream in probe["streams"] if stream["codec_type"] == "video")[
            "duration"
        ]
    )
    return duration


def get_video_dimensions(video_path: str) -> tuple:
    """
    Get the dimensions of a video file.

    Args:
    video_path (str): The path to the video file.

    Returns:
    tuple (width:int, height:int): A tuple containing the width and height of the video.
    """

    log.info("Getting video dimensions...")
    log.debug("Video path: %s", video_path)

    probe = ffmpeg.probe(video_path)
    video_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    ]
    width = int(video_streams[0]["width"])
    height = int(video_streams[0]["height"])
    return width, height


def get_audio_duration(audio_path: str):
    """
    Get the duration of an audio file in seconds.

    Args:
    audio_path (str): The file path to the audio file.

    Returns:
    float: Duration of the audio in seconds.
    """

    log.info("Getting audio length...")
    log.debug("Audio path: %s", audio_path)

    probe = ffmpeg.probe(audio_path)
    audio_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    ]
    if audio_streams:
        duration = float(audio_streams[0]["duration"])
        return duration


# Some additional logic to compress test files
def compress_video(input_path, output_path, config_preset="default"):
    """Compress video files to a lower bitrate."""
    log.info("Compressing video...")
    log.debug("Input path: %s", input_path)
    log.debug("Output path: %s", output_path)

    settings = ffmpeg_config.get(config_preset, ffmpeg_config["default"])
    log.debug("FFmpeg settings: %s", settings)
    log

    try:
        (
            ffmpeg.input(input_path)
            .output(
                output_path,
                vcodec="libx264",
                crf=settings["crf"],
                preset=settings["preset"],
                acodec="aac",
                strict="experimental",
            )
            .global_args(*settings["global_args"])
            .run(
                overwrite_output=True,
                capture_stdout=settings["capture_stdout"],
                capture_stderr=settings["capture_stderr"],
            )
        )
    except ffmpeg.Error as e:
        raise FFMpegProcessingError(
            "Error during compress_video ffmpeg command", stderr=e.stderr
        )


def compress_directory(directory):
    """Compress all MP4 files in the specified directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, f"compressed_{filename}")
            log.info(f"Compressing {input_path} to {output_path}...")
            compress_video(input_path, output_path)


if __name__ == "__main__":
    """
    Provides a command-line interface for the ffmpeg utility functions.
    """
    import sys

    if sys.argv[1] == "resize_video":
        resize_video(sys.argv[2], "cmd_line_output.mp4")
    elif sys.argv[1] == "split_video":
        split_video(
            sys.argv[2], cast(float, sys.argv[3]), "cmd_line_looped_output%01d.mp4"
        )
    elif sys.argv[1] == "loop_video_to_audio":
        audio_duration = get_audio_duration(sys.argv[2])
        loop_video_to_audio(
            cast(float, audio_duration), sys.argv[3], "cmd_line_looped_video.mp4"
        )
    elif sys.argv[1] == "compress":
        compress_directory(sys.argv[2])
    else:
        print("Invalid command")
        sys.exit(1)
