import logging
from typing import Dict, Set
import json
import uuid
import random
import string
import sys
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler('app.log') 
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websockets"])

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

manager = ConnectionManager()

@router.websocket("/{user_name}")
async def websocket_endpoint(websocket: WebSocket, user_name: str):
    logger.info(f"New WebSocket connection attempt from user: {user_name}")
    user_id = str(uuid.uuid4())
    await manager.connect(user_id, websocket)


    try:
        logger.info(f"WebSocket connection accepted for user: {user_name}")
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            logger.info(f"Received {action} from {user_name}")

            if action == "create_room":
                logger.info(f"User {user_name} is creating a room")
                room_id = str(uuid.uuid4())
                room_code = manager.generate_room_code()
                room_name = data.get("room_name", "Unname Room")
                latitude = data.get("latitude")
                longitude = data.get("longitude")

                manager.rooms[room_code] = {
                    "id": room_id,
                    "code": room_code,
                    "name": room_name,
                    "host": user_id,
                    "latitude": latitude,
                    "longitude": longitude,
                    "users": {
                        user_id: {
                            "name": user_name,
                            "is_host": True,
                            "status": "joined",           
                            "swipe_progress": 0,         
                            "total_movies": 0        
                        }
                    },
                    "pending_users": {}
                }
                manager.user_to_room[user_id] = room_code
                
                await websocket.send_json({
                    "action": "room_created",
                    "room_code": room_code,
                    "user_id": user_id,
                    "room_data": manager.rooms[room_code]
                })
                logger.info(f"Room {room_code} name {room_name}, lat {latitude}, lng {longitude} created by {user_name}")

            elif action == "join_room":
                room_code = data.get("room_code")
                if room_code in manager.rooms:
                    room = manager.rooms[room_code]
                    room["pending_users"][user_id] = {"name": user_name, "is_host": False}
                    
                    # Notify host
                    host_ws = manager.active_connections.get(room["host"])
                    if host_ws:
                        await host_ws.send_json({
                            "action": "join_request",
                            "user_id": user_id,
                            "user_name": user_name
                        })
                    
                    await websocket.send_json({
                        "action": "waiting_approval",
                        "room_code": room_code,
                        "user_id": user_id
                    })
                    logger.info(f"User {user_name} waiting for approval to join room {room_code}")

            elif action == "approve_user":
                room_code = manager.user_to_room.get(user_id)
                target_user_id = data.get("user_id")
                
                if room_code and target_user_id:
                    room = manager.rooms[room_code]
                    if user_id == room["host"] and target_user_id in room["pending_users"]:
                        user_data = room["pending_users"].pop(target_user_id)
                        room["users"][target_user_id] = user_data
                        manager.user_to_room[target_user_id] = room_code
                        
                        # Notify all users in room
                        await manager.broadcast_to_room(room_code, {
                            "action": "user_joined",
                            "room_data": room
                        })
                        logger.info(f"User {target_user_id} approved to join room {room_code}")

            elif action == "remove_user":
                room_code = manager.user_to_room.get(user_id)
                target_user_id = data.get("user_id")
                
                if room_code and target_user_id:
                    room = manager.rooms[room_code]
                    if user_id == room["host"] and target_user_id in room["users"]:
                        del room["users"][target_user_id]
                        if target_user_id in manager.user_to_room:
                            del manager.user_to_room[target_user_id]
                        
                        # Notify removed user
                        if target_user_id in manager.active_connections:
                            await manager.active_connections[target_user_id].send_json({
                                "action": "removed_from_room"
                            })
                        
                        # Notify remaining users
                        await manager.broadcast_to_room(room_code, {
                            "action": "user_removed",
                            "room_data": room
                        })
                        logger.info(f"User {target_user_id} removed from room {room_code}")

            elif action == "scan_rooms":
                logger.info(f"User {user_name} is scanning for room")
                user_lat = data.get("latitude")
                user_long = data.get("longitude")
                logger.info(f"Lat {user_lat}, Lng {user_long} is scanning for room")
                max_distance = data.get("max_distance", 5.0) 
                
                nearby_rooms = [
                    {
                        "code": code,
                        "name": room["name"],
                        "user_count": len(room["users"]),
                        "distance": calculate_distance(
                            user_lat, user_long,
                            room["latitude"], room["longitude"]
                        )
                    }
                    for code, room in manager.rooms.items()
                    if calculate_distance(
                        user_lat, user_long,
                        room["latitude"], room["longitude"]
                    ) <= max_distance
                ]
                
                await websocket.send_json({
                    "action": "room_list",
                    "rooms": nearby_rooms
                })
                logger.info(f"Room scan requested by {user_name}")

            elif action == "get_room_details":
                room_code = data.get("room_code")
                if room_code in manager.rooms:
                    room = manager.rooms[room_code]
                    await websocket.send_json({
                        "action": "room_details",
                        "room_data": room
                    })
                    logger.info(f"Room details requested for room {room_code}")

            elif action == "update_status":
                room_code = manager.user_to_room.get(user_id)
                if room_code:
                    new_status = data.get("status")
                    if new_status in ["joined", "swiping", "completed"]:
                        room = manager.rooms[room_code]
                        if user_id in room["users"]:
                            room["users"][user_id]["status"] = new_status
                            await manager.broadcast_to_room(
                                room_code, 
                                {
                                    "action": "status_updated",
                                    "user_id": str(user_id),
                                    "status": new_status,
                                    "room_data": room
                                }
                            )

            elif action == "update_progress":
                room_code = manager.user_to_room.get(user_id)
                if room_code:
                    room = manager.rooms[room_code]
                    if user_id in room["users"]:
                        current_count = data.get("current_count", 0)
                        total_count = data.get("total_count", 0)
                        room["users"][user_id]["swipe_progress"] = current_count
                        room["users"][user_id]["total_movies"] = total_count
            
                        # Automatically update status based on progress
                        if current_count == 0:
                            room["users"][user_id]["status"] = "joined"
                        elif current_count == total_count:
                            room["users"][user_id]["status"] = "completed"
                        else:
                            room["users"][user_id]["status"] = "swiping"

                        await manager.broadcast_to_room(
                            room_code,
                            {
                                "action": "progress_updated",
                                "user_id": str(user_id),
                                "progress": current_count,
                                "total": total_count,
                                "room_data": room
                            }
                        )

    except WebSocketDisconnect:
        manager.disconnect(user_id)
        room_code = manager.user_to_room.get(user_id)
        
        if room_code:
            room = manager.rooms[room_code]
            if user_id == room["host"]:
                # Host disconnected - close room
                for room_user_id in room["users"]:
                    if room_user_id in manager.user_to_room:
                        del manager.user_to_room[room_user_id]
                del manager.rooms[room_code]
                await manager.broadcast_to_room(room_code, {
                    "action": "room_closed"
                })
                logger.info(f"Room {room_code} closed - host disconnected")
            else:
                # Regular user disconnected
                if user_id in room["users"]:
                    del room["users"][user_id]
                if user_id in manager.user_to_room:
                    del manager.user_to_room[user_id]
                await manager.broadcast_to_room(room_code, {
                    "action": "user_left",
                    "room_data": room
                })
                logger.info(f"User {user_name} left room {room_code}")

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance