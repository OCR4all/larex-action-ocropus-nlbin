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


def worker(
    connection: Connection, 
    threshold: float, 
    zoom: float, 
    scale: float, 
    border: float, 
    percentile: int, 
    range_: int, 
    low: int, 
    high: int
) -> None:
    """
    Process-isolated action worker.
 
    Protocol:
    1. Send exactly one `WorkerStatus` after startup: 'loaded', or 'error' followed by exit if loading failed.
    2. For every request (`request = connection.recv()`), send exactly one `WorkerResponse`
       or `WorkerStatus('error', ...)` (`connection.send(obj)`).
    3. Exit when `None` is received or the parent closes the pipe.
    """
    logger.debug('Initialize worker')
    
    try:
        from nlbin import Nlbin
        from PIL import Image
        
        nlbin = Nlbin(env.NLBIN_DEVICE)
        
        connection.send(WorkerStatus('loaded'))
    except Exception as exc:
        logger.exception('Failed to load worker')
        try:
            connection.send(WorkerStatus('error', f'{type(exc).__name__} {exc}'))
        except (BrokenPipeError, EOFError, OSError):
            logger.warning('Could not send load failure to parent')
        connection.close()
        return
    
    def _fmt(suffix: str) -> str:
        ext = suffix.split('.')[-1]
        return {'jpg': 'JPEG', 'jpeg': 'JPEG', 'tif': 'TIFF'}.get(ext, ext.upper())

    def to_bytes(img: Image.Image, suffix: str) -> bytes:
        fmt = _fmt(suffix)
        # JPEG can't store mode "1" (or "P"/"RGBA"); convert if needed
        if fmt == 'JPEG' and img.mode not in ('L', 'RGB'):
            img = img.convert('L')
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()   # bytes, independent of cursor position
    
    while True:
        try:
            request = connection.recv()
        except EOFError:
            logger.debug('Parent closed pipe; shutting down worker')
            break
        if request is None:
            logger.debug('Parent sent shutdown signal; shutting down worker')
            break   
        if not isinstance(request, WorkerRequest):
            logger.error(f'Received unexpected request: {type(request).__name__}')
            continue

        try:
            image = Image.open(io.BytesIO(request.image))
            binary, normalized = nlbin.process(
                image, 
                threshold, 
                zoom, 
                scale, 
                border, 
                percentile, 
                range_,
                low,
                high
            )
            
            connection.send(WorkerResponse(
                to_bytes(binary, env.NLBIN_BIN_SUFFIX), 
                to_bytes(normalized, env.NLBIN_NRM_SUFFIX)
            ))

        except Exception as exc:
            logger.exception('Failed to process element')
            try:
                connection.send(WorkerStatus('error', f'{type(exc).__name__} {exc}'))
            except (BrokenPipeError, EOFError, OSError):
                logger.warning('Could not send worker failure to parent')
                break

    connection.close()
    logger.info('Worker stopped successfully')
