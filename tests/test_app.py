"""API contracts and failure paths; run with python -m unittest discover -s tests."""
import io
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import soundfile as sf

from fastapi.testclient import TestClient
from PIL import Image
from app.app import app, identifier
from app.inference import ModelUnavailable


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_all_pages_and_assets(self):
        for route in ('/', '/bird', '/audio', '/about', '/collection', '/static/app.js', '/static/mosaic.js'):
            self.assertEqual(self.client.get(route).status_code, 200)
        self.assertEqual(self.client.get('/unknown').status_code, 404)

    def test_invalid_photo(self):
        response = self.client.post('/api/predict/image', files={'file': ('bad.jpg', b'not an image', 'image/jpeg')})
        self.assertEqual(response.status_code, 400)

    def test_missing_upload(self):
        self.assertEqual(self.client.post('/api/predict/image').status_code, 422)

    def test_photo_prediction_contract(self):
        image = io.BytesIO()
        Image.new('RGB', (224, 224), '#778833').save(image, format='PNG')
        response = self.client.post('/api/predict/image', files={'file': ('image.png', image.getvalue(), 'image/png')})
        self.assertEqual(response.status_code, 200)
        predictions = response.json()['predictions']
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= row['score'] <= 1 for row in predictions))
        self.assertGreaterEqual(predictions[0]['score'], predictions[-1]['score'])

    def test_unavailable_audio(self):
        with patch.object(identifier, 'audio', side_effect=ModelUnavailable('Audio model unavailable.')):
            response = self.client.post('/api/predict/audio', files={'file': ('bird.wav', b'anything', 'audio/wav')})
        self.assertEqual(response.status_code, 503)

    def test_audio_extension(self):
        response = self.client.post('/api/predict/audio', files={'file': ('script.py', b'print(1)')})
        self.assertEqual(response.status_code, 400)

    def test_upload_limit(self):
        with patch('app.app.MAX_UPLOAD', 4):
            response = self.client.post('/api/predict/image', files={'file': ('image.png', b'12345')})
        self.assertEqual(response.status_code, 413)

    def test_short_audio(self):
        recording = io.BytesIO()
        sf.write(recording, np.zeros(22050), 22050, format='WAV')
        response = self.client.post('/api/predict/audio', files={'file': ('short.wav', recording.getvalue(), 'audio/wav')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('3 seconds', response.json()['detail'])

    def test_real_audio_prediction(self):
        recording = next(Path('data/audio/test/Coracias_benghalensis').glob('*.mp3'))
        with recording.open('rb') as file:
            response = self.client.post('/api/predict/audio', files={'file': (recording.name, file, 'audio/mpeg')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['predictions']), 5)

    def test_audio_split_has_no_shared_recordings(self):
        root = Path('data/audio/spectrograms_split')
        recordings = lambda split: {(p.parent.name, p.stem.rsplit('_', 1)[0]) for p in (root / split).glob('*/*.png')}
        train, validation = recordings('train'), recordings('val')
        self.assertTrue(train and validation)
        self.assertFalse(train & validation)

    def test_collection(self):
        response = self.client.get('/api/species')
        self.assertEqual(response.status_code, 200)
        species = response.json()
        self.assertGreaterEqual(len(species), 20)
        for bird in species:
            self.assertIn('scientific_name', bird)
        self.assertEqual(self.client.get('/api/species/-1/image').status_code, 404)
        self.assertEqual(self.client.get('/api/species/100000/image').status_code, 404)


if __name__ == '__main__':
    unittest.main()
