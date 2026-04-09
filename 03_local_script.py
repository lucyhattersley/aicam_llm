from openai import OpenAI

# Your server URL (replace with your actual URL)
url = 'https://your-ngrok-url.ngrok-free.dev'

client = OpenAI(api_key='your-api-key-here')

resp = client.responses.create(
    model="gpt-4o-mini",
    tools=[
        {
            "type": "mcp",
            "server_label": "rpi_server",
            "server_url": f"{url}/sse",
            "require_approval": "never",
        },
    ],
    input="What is the last inference result?"
)

print(resp.output_text)
