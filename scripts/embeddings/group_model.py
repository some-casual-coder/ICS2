from typing import List
from pydantic import BaseModel

class GroupPreferencesModel(BaseModel):
    group_preferences: List[int]