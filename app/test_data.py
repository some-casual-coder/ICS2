TEST_USERS = [
    {"id": "user1", "name": "Alice", "is_host": True},
    {"id": "user2", "name": "Bob", "is_host": False},
    {"id": "user3", "name": "Charlie", "is_host": False},
]

TEST_ROOMS = [
    {
        "id": "room1",
        "code": "ABC123",
        "host_id": "user1",
        "users": {
            "user1": TEST_USERS[0],
            "user2": TEST_USERS[1]
        },
        "pending_users": {
            "user3": TEST_USERS[2]
        }
    }
]