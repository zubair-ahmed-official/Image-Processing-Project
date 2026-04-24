import asyncio
import websockets

async def handler(websocket):
    async for message in websocket:
        print("Received:", message)

async def main():
    async with websockets.serve(handler, "127.0.0.1", 8080):
        print("WebSocket server running on ws://127.0.0.1:8080")
        await asyncio.Future()

asyncio.run(main())