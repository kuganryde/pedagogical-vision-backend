import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx

logger = logging.getLogger("edu_vision.events")

class EventDispatcher:
    """Dispatches real-time behavioral alerts and events to registered webhooks."""

    def __init__(self, webhook_urls: Optional[List[str]] = None):
        self.webhook_urls = webhook_urls or []
        self._client = httpx.AsyncClient(timeout=4.0)

    def register_webhook(self, url: str):
        if url not in self.webhook_urls:
            self.webhook_urls.append(url)

    def unregister_webhook(self, url: str):
        if url in self.webhook_urls:
            self.webhook_urls.remove(url)

    async def dispatch_event(self, event_type: str, payload: Dict[str, Any]):
        """Dispatches an event asynchronously without blocking frame processing."""
        if not self.webhook_urls:
            return

        body = {
            "event": event_type,
            "data": payload
        }

        async def send(url: str):
            try:
                await self._client.post(url, json=body)
            except Exception as e:
                logger.warning(f"Failed to post event {event_type} to {url}: {e}")

        tasks = [send(url) for url in self.webhook_urls]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        await self._client.aclose()
