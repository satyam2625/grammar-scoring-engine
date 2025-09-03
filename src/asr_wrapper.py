# src/asr_wrapper.py
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_processor = None
_model = None

def init_asr(model_name="facebook/wav2vec2-base-960h"):
    global _processor, _model
    if _processor is None:
        _processor = Wav2Vec2Processor.from_pretrained(model_name)
        _model = Wav2Vec2ForCTC.from_pretrained(model_name).to(_device)

def transcribe(y, sr=16000):
    if _processor is None or _model is None:
        init_asr()
    inputs = _processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(_device)
    with torch.no_grad():
        logits = _model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return _processor.batch_decode(pred_ids)[0]
