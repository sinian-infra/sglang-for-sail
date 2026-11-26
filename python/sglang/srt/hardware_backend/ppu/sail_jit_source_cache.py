"""Content-addressed atomic JIT generated-source cache with local fallback.

Prevents rank/layer/stream-specific paths from entering ``load_jit`` build keys
(which would force redundant N-rank recompilation storms) by publishing complete
source bytes atomically under ``<cache_dir>/sail_generated_sources/<sha256>/``.
When a shared cache directory is unavailable (e.g. EROFS/EACCES/ESTALE), falls
back once to a PID-scoped local temp directory without hiding compiler errors.
"""

import errno
import hashlib
import logging
import os
import tempfile
from functools import cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR_ERRNOS = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
        errno.ENOSPC,
        errno.EDQUOT,
        errno.ESTALE,
        errno.EIO,
        errno.ETIMEDOUT,
        errno.ENOTDIR,
        errno.ENOENT,
        errno.ENOLCK,
        errno.EOPNOTSUPP,
        errno.ENOSYS,
    }
)


@cache
def _local_cache_root(pid: int) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"sglang-sail-jit-local-{pid}-"))


def local_cache_root() -> Path:
    """Return a PID-scoped local fallback cache root (fork-safe)."""
    return _local_cache_root(os.getpid())


def is_cache_directory_error(error: OSError, root: Path) -> bool:
    """Distinguish cache-directory I/O failures from missing compilers or build errors."""
    if error.filename is not None:
        path = Path(os.path.abspath(error.filename))
        if not path.is_relative_to(Path(os.path.abspath(root))):
            return False
    return error.errno in _CACHE_DIR_ERRNOS


def publish_source(
    filename: str,
    source: str,
    *,
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Atomically publish generated CUDA/C++ source under its SHA256 content address."""
    if not filename or filename in (".", "..") or Path(filename).name != filename:
        raise ValueError(f"Generated source requires a plain filename, got {filename!r}")
    data = source.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    root = Path(
        cache_dir or os.environ.get("SGLANG_JIT_CACHE_DIR") or "~/.cache/sglang/jit"
    ).expanduser()
    directory = root / "sail_generated_sources" / digest
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / filename
    try:
        if output.read_bytes() == data:
            return output
    except FileNotFoundError:
        pass

    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=f".{filename}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def load_generated_jit(load_jit_fn, filename: str, source: str, *args, **kwargs):
    """Publish generated source atomically and invoke ``load_jit_fn`` with single local fallback."""
    root = Path(
        os.environ.get("SGLANG_JIT_CACHE_DIR") or "~/.cache/sglang/jit"
    ).expanduser()

    def _load(directory: Path, *, override_env: bool):
        header = publish_source(filename, source, cache_dir=directory)
        if not override_env:
            return load_jit_fn(*args, cuda_files=[str(header)], **kwargs)
        prev = os.environ.get("SGLANG_JIT_CACHE_DIR")
        os.environ["SGLANG_JIT_CACHE_DIR"] = str(directory)
        try:
            return load_jit_fn(*args, cuda_files=[str(header)], **kwargs)
        finally:
            if prev is None:
                os.environ.pop("SGLANG_JIT_CACHE_DIR", None)
            else:
                os.environ["SGLANG_JIT_CACHE_DIR"] = prev

    try:
        return _load(root, override_env=False)
    except OSError as error:
        if not is_cache_directory_error(error, root):
            raise
        local = local_cache_root()
        logger.warning(
            "Sail JIT cache %s unavailable; compiling at local fallback %s: %s",
            root,
            local,
            error,
        )
        return _load(local, override_env=True)
