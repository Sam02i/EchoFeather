"""Run with `python app/app.py`. One small API serves the website and both models."""
import io
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import uvicorn

try:
    from .inference import BirdIdentifier, COMMON_NAMES, ROOT, ModelUnavailable, class_names, model_paths
except ImportError:
    from inference import BirdIdentifier, COMMON_NAMES, ROOT, ModelUnavailable, class_names, model_paths

STATIC = Path(__file__).parent / 'static'
MAX_UPLOAD = 25 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
identifier = BirdIdentifier()
app = FastAPI(title='Samiksha · Bird ID')
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/api/status')
def status():
    return {kind: {'available': all(path.is_file() for path in model_paths(kind)),
                'species_count': len(class_names(kind))}
            for kind in ('vision', 'audio')}


@app.get('/api/species')
def species():
    photo, audio = class_names('vision'), class_names('audio')
    return [{'id': index, 'species': name, 'name': COMMON_NAMES.get(name, name.replace('_', ' ')),
            'scientific_name': name.replace('_', ' '), 'photo': name in photo, 'audio': name in audio,
            'image': f'/api/species/{index}/image' if species_image(name) else None}
            for index, name in enumerate(sorted(set(photo + audio)))]


def species_image(name):
    bundled = next((p for p in (STATIC / 'species').glob(f'{name}.*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), None)
    if bundled:
        return bundled
    folder = ROOT / 'data' / 'images' / 'val' / name
    return next((p for p in sorted(folder.glob('*')) if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), None)


@app.get('/api/species/{index}/image')
def thumbnail(index: int):
    names = sorted(set(class_names('vision') + class_names('audio')))
    if index < 0 or index >= len(names):
        raise HTTPException(404, 'Species not found.')
    image = species_image(names[index])
    if image is None:
        raise HTTPException(404, 'No photograph available.')
    return FileResponse(image)


def read_upload(file):
    data = file.file.read(MAX_UPLOAD + 1)
    if not data:
        raise HTTPException(400, 'Choose a file first.')
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, 'Choose a file smaller than 25 MB.')
    return data


@app.post('/api/predict/image')
def predict_image(file: UploadFile = File(...)):
    try:
        data = read_upload(file)
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise HTTPException(413, 'Choose a photo under 20 megapixels.')
            predictions = identifier.image(image)
        return {'predictions': predictions, 'kind': 'vision'}
    except ModelUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(400, 'This file could not be read as a photo. Try JPEG, PNG or WebP.') from error
    except HTTPException:
        raise
    except Exception as error:
        logging.exception('Photo inference failed')
        raise HTTPException(500, 'Photo identification failed. Please try another image.') from error
    finally:
        file.file.close()


@app.post('/api/predict/audio')
def predict_audio(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
            raise HTTPException(400, 'Use a WAV, MP3, FLAC, OGG or M4A recording.')
        data = read_upload(file)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f'upload{suffix}'
            path.write_bytes(data)
            predictions = identifier.audio(path)
        return {'predictions': predictions, 'kind': 'audio', 'analyzed_seconds_limit': 30}
    except ModelUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logging.exception('Audio inference failed')
        raise HTTPException(500, 'Audio identification failed. Please try another recording.') from error
    finally:
        file.file.close()


@app.get('/')
@app.get('/bird')
@app.get('/audio')
@app.get('/about')
@app.get('/collection')
def page():
    return FileResponse(STATIC / 'index.html')


if __name__ == '__main__':
    uvicorn.run(app, host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '8000')))
