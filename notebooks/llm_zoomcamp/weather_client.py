"""Simple MCP Client for Weather Service"""
import asyncio

from fastmcp import Client

async def main():
    async with Client("weather_server.py") as mcp_client:
        result = await mcp_client.list_tools()
        print(result)

if __name__ == "__main__":
    test = asyncio.run(main())