"""Where burst images live.

The website and the predictor can use different directories. On the current
server that split is `paths.py` (gitignored). Environment variables override
it, which is what the dev and prod containers use.

Bare process (systemd), in order:

- `FLARESENSE_IMAGE_DIR` — directory the website reads
- `FLARESENSE_PREDICT_DIR` — directory the predictor writes
- otherwise `paths.py` (`SERVER_BURST_IMAGES_PATH`, `NAS_BURST_IMAGES_PATH`)
- otherwise `./data/burst_images`

Docker bind-mounts the host directory to `/data/burst_images` and sets
`FLARESENSE_IMAGE_DIR` to that container path. The host path is
`FLARESENSE_HOST_IMAGE_DIR` in Compose, not this variable.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_IMAGE_DIR = os.path.join(_REPO_ROOT, "data", "burst_images")


def _legacy(name: str):
    try:
        import paths
    except ImportError:
        return None
    return getattr(paths, name, None)


def image_dir() -> str:
    """Directory the website reads burst PNGs from."""
    env = os.environ.get("FLARESENSE_IMAGE_DIR")
    if env:
        return env
    legacy = _legacy("SERVER_BURST_IMAGES_PATH")
    if legacy:
        return legacy
    os.makedirs(_LOCAL_IMAGE_DIR, exist_ok=True)
    return _LOCAL_IMAGE_DIR


def predict_image_dir() -> str:
    """Directory the predictor writes burst PNGs to."""
    env = os.environ.get("FLARESENSE_PREDICT_DIR")
    if env:
        return env
    legacy = _legacy("NAS_BURST_IMAGES_PATH")
    if legacy:
        return legacy
    return image_dir()
