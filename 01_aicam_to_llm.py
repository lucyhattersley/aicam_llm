import os
import time
from openai import OpenAI
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

# --- Setup ---
client = OpenAI(api_key=<OPENAI_API_KEY>)
CONFIDENCE_THRESHOLD = 0.3
device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

with device as stream:
    for frame in stream:
        frame.display() # Displays preview window
        
        # Filter confident detections
        detections = frame.detections[frame.detections.confidence > CONFIDENCE_THRESHOLD]
        labels = [f"{model.labels[c]} ({s:.2f})" for _, s, c, _ in detections]

        # Build a natural prompt
        prompt = f"At {time.strftime('%H:%M:%S')}, the camera detected: {labels}"
        print("Prompt to LLM:", prompt)

        # Ask the LLM for a summary
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        print("LLM summary:", response.output[0].content[0].text)

        time.sleep(5)
