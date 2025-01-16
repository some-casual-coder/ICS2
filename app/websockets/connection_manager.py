from typing import Dict
from uuid import UUID
from fastapi import WebSocket

from app.websockets.room import User

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[UUID, WebSocket] = {}

    async def connect(self, user_id: UUID, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: UUID):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def broadcast_to_room(self, room_users: Dict[UUID, User], message: dict):
        for user_id in room_users:
            if user_id in self.active_connections:
                await self.active_connections[user_id].send_json(message)