# SPDX-License-Identifier: MIT
import io
import logging
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Literal

from .config import env

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if env.ACTION_DEBUG else logging.INFO)


@dataclass
class WorkerRequest:
    """ Inputs for the worker process """
    image: bytes


@dataclass
class WorkerResponse:
    """ Worker outputs on success """
    binary: bytes
    normalized: bytes


@dataclass
class WorkerStatus:
    """ Status message from the worker """
    type: Literal['loaded', 'error']
    message: str | None = None


def _send(connection: Connection, message: object) -> bool:
    """ Send a message to the parent. Returns False if the parent is gone. """
    try:
        connection.send(message)
    except OSError:
        logger.warning(f'Could not send {type(message).__name__} to parent')
        return False
    return True


def worker(connection: Connection, *args) -> None:
    """
    Process-isolated action worker.

    Protocol:
    1. Send exactly one `WorkerStatus` after startup: 'loaded', or 'error' followed by exit if loading failed.
    2. For every request (`request = connection.recv()`), send exactly one `WorkerResponse`
       or `WorkerStatus('error', ...)` (`connection.send(obj)`).
    3. Exit when `None` is received or the parent closes the pipe.
    """
    logger.debug('Initialize worker')

    with connection:
        try:
            from nlbin import Nlbin
            from PIL import Image
            
            nlbin = Nlbin(env.NLBIN_DEVICE)
        except Exception as exc:
            logger.exception('Failed to load worker')
            _send(connection, WorkerStatus('error', f'{type(exc).__name__} {exc}'))
            return

        if not _send(connection, WorkerStatus('loaded')):
            return
        
        def _fmt(suffix: str) -> str:
            ext = suffix.split('.')[-1]
            return {'jpg': 'JPEG', 'jpeg': 'JPEG', 'tif': 'TIFF'}.get(ext, ext.upper())

        def _to_bytes(img: Image.Image, suffix: str) -> bytes:
            fmt = _fmt(suffix)
            if fmt == 'JPEG' and img.mode not in ('L', 'RGB'):
                img = img.convert('L')
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            return buf.getvalue()

        while True:
            try:
                request = connection.recv()
            except (EOFError, OSError):
                logger.debug('Parent closed pipe; shutting down worker')
                break
            if request is None:
                logger.debug('Parent sent shutdown signal; shutting down worker')
                break

            try:
                if not isinstance(request, WorkerRequest):
                    raise TypeError(f'Unexpected request: {type(request).__name__}')

                image = Image.open(io.BytesIO(request.image))
                binary, normalized = nlbin.process(image, *args)
                
                connection.send(WorkerResponse(
                    _to_bytes(binary, env.NLBIN_BIN_SUFFIX), 
                    _to_bytes(normalized, env.NLBIN_NRM_SUFFIX)
                ))

            except Exception as exc:
                logger.exception('Failed to process element')
                if not _send(connection, WorkerStatus('error', f'{type(exc).__name__} {exc}')):
                    break

    logger.info('Worker stopped successfully')
