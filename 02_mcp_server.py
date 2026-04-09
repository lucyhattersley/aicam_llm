import asyncio
from mcp.server.fastmcp import FastMCP
from modlib.devices.ai_camera import AiCamera
from modlib.models.zoo import NanoDetPlus416x416

last_inference_result = None

mcp = FastMCP("RaspberryPi MCP Server", host="0.0.0.0")


# Define a tool the LLM can call
@mcp.tool()
async def get_last_inference() -> str:
    """
    Returns the most recent inference result from the AI Camera.
    """
    return str(last_inference_result or "No inference result yet.")


async def run_inference():
    global last_inference_result
    device = AiCamera(frame_rate=30, image_size=(640, 480))
    model = NanoDetPlus416x416()
    device.deploy(model)

    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.2]
            labels = [f"{model.labels[c]} ({s:.2f})" for _, s, c, _ in detections]
            last_inference_result = {"detections": labels}
            frame.display()
            await asyncio.sleep(0.01)


async def main():
    await asyncio.gather(
        mcp.run_sse_async(),  # MCP server
        run_inference()       # Camera inference loop
    )


if __name__ == "__main__":
    asyncio.run(main())
