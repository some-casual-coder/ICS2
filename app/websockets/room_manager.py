from typing import Dict, List, Optional
from uuid import UUID, uuid4
import random
import string
from .room import Room, User

class RoomManager:
    def __init__(self):
        self.rooms: Dict[UUID, Room] = {}
        self.user_to_room: Dict[UUID, UUID] = {}

    def generate_room_code(self, length: int = 6) -> str:
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
            if not any(room.code == code for room in self.rooms.values()):
                return code

    def create_room(self, host_name: str) -> Room:
        room_id = uuid4()
        host_id = uuid4()
        room = Room(
            id=room_id,
            code=self.generate_room_code(),
            host_id=host_id,
            users={
                host_id: User(id=host_id, name=host_name, is_host=True)
            }
        )
        self.rooms[room_id] = room
        self.user_to_room[host_id] = room_id
        return room

    def get_room_by_code(self, code: str) -> Optional[Room]:
        return next((room for room in self.rooms.values() if room.code == code), None)

    def add_pending_user(self, room_id: UUID, user_name: str) -> tuple[UUID, Room]:
        user_id = uuid4()
        user = User(id=user_id, name=user_name)
        room = self.rooms[room_id]
        room.pending_users[user_id] = user
        return user_id, room

    def approve_user(self, room_id: UUID, user_id: UUID) -> Optional[Room]:
        room = self.rooms[room_id]
        if user_id in room.pending_users:
            user = room.pending_users.pop(user_id)
            room.users[user_id] = user
            self.user_to_room[user_id] = room_id
            return room
        return None

    def remove_user(self, room_id: UUID, user_id: UUID) -> Optional[Room]:
        room = self.rooms[room_id]
        if user_id in room.users:
            del room.users[user_id]
            del self.user_to_room[user_id]
            return room
        return None

    def get_all_rooms(self) -> List[Room]:
        return list(self.rooms.values())
