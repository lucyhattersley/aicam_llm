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
        detections = frame.detections[frame.detections.confidence > CONFIDENCE_THRESHOLD]
        labels = [f"{model.labels[c]} ({s:.2f})" for _, s, c, _ in detections]

        prompt = f"You have access to a smart camera in a warehouse. At {time.strftime('%H:%M:%S')}, the camera detected: {labels} Provide information if people are wearing highvis jackets."
        print("Prompt to LLM:", prompt)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        print("LLM summary:", response.output[0].content[0].text)

        time.sleep(5)
