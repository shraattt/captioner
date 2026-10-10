"""Identity, table layout and media filters shared by the dashboard."""

from captioner import __version__

APP_NAME = "Captioner"
APP_VERSION = __version__


MEDIA_FILTER = (
    "Media files (*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.wav *.m4a *.flac *.aac *.ogg *.opus);;"
    "All files (*)"
)
MEDIA_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".3gp",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
}
# waiting than medium (measured in scripts/model_sweep.py); large-v3 for when
# only the best will do. The batch default comes from config.
MODEL_CHOICES = ("tiny.en", "base.en", "small.en", "medium", "distil-large-v3", "large-v3")
COLUMNS = ("#", "File", "Status", "Model", "Progress", "Created", "Duration", "Details")
# A running job that has not checked in for this long is probably wedged.
STALL_WARNING_SECONDS = 30.0
