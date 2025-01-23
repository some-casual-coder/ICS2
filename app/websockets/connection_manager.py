import logging
import random
import string
import sys
from typing import Dict
from uuid import UUID
from fastapi import WebSocket

from app.websockets.room import User

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler('app.log') 
    ]
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, dict] = {}
        self.user_to_room: Dict[str, str] = {}

    def generate_room_code(self, length: int = 6) -> str:
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        logger.info(f"User {user_id} disconnected")

    async def broadcast_to_room(self, room_code: str, message: dict, exclude_user: str = None):
        if room_code in self.rooms:
            for user_id in self.rooms[room_code]["users"]:
                if user_id != exclude_user and user_id in self.active_connections:
                    await self.active_connections[user_id].send_json(message)

    async def send_personal_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)