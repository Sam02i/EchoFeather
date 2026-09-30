"""Load local checkpoints and run the same preprocessing used during training."""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'audio'))

COMMON_NAMES = {
    'Acridotheres_tristis': 'Common myna', 'Anastomus_oscitans': 'Asian openbill',
    'Ardea_cinerea': 'Grey heron', 'Bubulcus_coromandus': 'Eastern cattle egret',
    'Cinnyris_asiaticus': 'Purple sunbird', 'Columba_livia': 'Rock pigeon',
    'Copsychus_saularis': 'Oriental magpie-robin', 'Coracias_benghalensis': 'Indian roller',
    'Corvus_splendens': 'House crow', 'Dicrurus_macrocercus': 'Black drongo',
    'Halcyon_smyrnensis': 'White-throated kingfisher', 'Merops_orientalis': 'Green bee-eater',
    'Milvus_migrans': 'Black kite', 'Orthotomus_sutorius': 'Common tailorbird',
    'Passer_domesticus': 'House sparrow', 'Psittacula_krameri': 'Rose-ringed parakeet',
    'Pycnonotus_cafer': 'Red-vented bulbul', 'Streptopelia_decaocto': 'Eurasian collared dove',
    'Threskiornis_melanocephalus': 'Black-headed ibis', 'Vanellus_indicus': 'Red-wattled lapwing',
}


class ModelUnavailable(Exception):
    pass


def model_paths(kind):
    """Environment overrides, bundled app files, then this repository's training output."""
    default = ROOT / 'runs' / ('vision_v2' if kind == 'vision' else 'audio_v1')
    model = Path(os.environ.get(f'{kind.upper()}_MODEL', default / 'best_model.pth'))
    classes = Path(os.environ.get(f'{kind.upper()}_CLASSES', default / 'classes.json'))
    if not os.environ.get(f'{kind.upper()}_MODEL') and (ROOT / 'app' / f'{kind}_model.pth').exists():
        model = ROOT / 'app' / f'{kind}_model.pth'
        classes = ROOT / 'app' / f'{kind}_classes.json'
    return model, classes


def class_names(kind):
    _, path = model_paths(kind)
    if not path.is_file():
        return []
    names = json.loads(path.read_text())
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError('The model class list is invalid.')
    return names


class BirdIdentifier:
    def __init__(self):
        # CPU inference is portable and leaves the GPU available for training.
        self.models = {}
        self.lock = threading.RLock()
        self.vision_transform = transforms.Compose([
            transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.audio_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
        ])

    def load(self, kind):
        if kind in self.models:
            return self.models[kind]
        checkpoint, classes_path = model_paths(kind)
        if not checkpoint.is_file() or not classes_path.is_file():
            raise ModelUnavailable(f'The {kind} model is not available yet.')
        names = class_names(kind)
        if kind == 'vision':
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(names))
        else:
            from train_audio_cnn import SmallAudioCNN
            model = SmallAudioCNN(len(names))
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        self.models[kind] = model, names
        return model, names

    @staticmethod
    def result(probabilities, names):
        values, indices = probabilities.topk(min(5, len(names)))
        return [{'species': names[index],
                 'name': COMMON_NAMES.get(names[index], names[index].replace('_', ' ')),
                 'scientific_name': names[index].replace('_', ' '),
                 'score': float(score)} for score, index in zip(values, indices.tolist())]

    def image(self, image):
        with self.lock, torch.inference_mode():
            model, names = self.load('vision')
            image = ImageOps.exif_transpose(image).convert('RGB')
            tensor = self.vision_transform(image).unsqueeze(0)
            return self.result(model(tensor).softmax(1)[0], names)

    def audio(self, path):
        # A private folder and lock prevent concurrent requests from mixing clips/plots.
        with self.lock, torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
            model, names = self.load('audio')
            from audio_to_melspec import clip_to_spectrogram_images
            output = Path(directory)
            count = clip_to_spectrogram_images(path, output, 'clip', max_seconds=30)
            if not count:
                raise ValueError('Use a readable recording with at least 3 seconds of audio.')
            scores = []
            for file in sorted(output.glob('clip_*.png')):
                with Image.open(file) as image:
                    tensor = self.audio_transform(image.convert('RGB')).unsqueeze(0)
                scores.append(model(tensor).softmax(1)[0])
            return self.result(torch.stack(scores).mean(0), names)
