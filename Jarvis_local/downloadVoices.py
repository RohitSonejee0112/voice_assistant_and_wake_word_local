import urllib.request
import os

os.makedirs("models", exist_ok=True)

VOICES = {
    # J.A.R.V.I.S. picks
    "en_GB-alan-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
    "en_GB-alan-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    "en_GB-alan-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx",
    "en_GB-alan-low.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/low/en_GB-alan-low.onnx.json",
    
    # Alternatives
    "en_US-ryan-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    "en_US-ryan-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    "en_US-danny-low.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/danny/low/en_US-danny-low.onnx",
    "en_US-danny-low.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/danny/low/en_US-danny-low.onnx.json",
}

for name, url in VOICES.items():
    path = f"models/{name}"
    if os.path.exists(path):
        print(f"✓ {name}")
        continue
    print(f"↓ {name}...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved")

print("\nDone! Update config.py PIPER_MODEL to one of:")
print("  models/en_GB-alan-medium.onnx      ← J.A.R.V.I.S. (recommended)")
print("  models/en_GB-alan-low.onnx         ← Faster J.A.R.V.I.S.")
print("  models/en_US-ryan-medium.onnx      ← Neutral professional")
print("  models/en_US-danny-low.onnx        ← Deep/older")