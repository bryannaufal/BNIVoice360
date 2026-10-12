"""Cache embedding artifacts for future use without loading vectors into memory."""

import shutil
import ssl
import tempfile
from pathlib import Path
from urllib.request import urlopen

from bnivoice360.config import Settings


def valid_parquet(path: Path) -> bool:
    """Check the standard Parquet envelope, rejecting empty/HTML/partial downloads."""
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with path.open("rb") as source:
        start = source.read(4)
        source.seek(-4, 2)
        return start == source.read(4) == b"PAR1"


def ensure_embeddings(settings: Settings) -> tuple[Path, ...]:
    if not settings.download_embeddings:
        return ()
    directory = settings.embedding_directory
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in (
        ("topic_id_embedding_mapping.parquet", settings.topic_embedding_url),
        ("data_id_embedding_mapping.parquet", settings.data_embedding_url),
    ):
        destination = directory / filename
        if not valid_parquet(destination):
            # Unique temporary files and atomic replacement keep partial files out of the cache.
            temporary = None
            try:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.load_default_certs()
                with (
                    urlopen(url, timeout=60, context=context) as response,
                    tempfile.NamedTemporaryFile(
                        dir=directory, suffix=".part", delete=False
                    ) as output,
                ):
                    temporary = Path(output.name)
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                if not valid_parquet(temporary):
                    raise ValueError(
                        f"{filename}: download is not a Parquet file; check Google Drive sharing permissions."
                    )
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        paths.append(destination)
    return tuple(paths)
