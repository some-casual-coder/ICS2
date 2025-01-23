from firebase_admin import firestore
from ..core.firebase import db
from .models import BatchInteraction, MovieInteraction, RoomStatistics, SwipeAction, MovieStats
from fastapi import HTTPException

class InteractionService:
    def __init__(self):
        self.db = db

    async def record_interaction(self, interaction: MovieInteraction):
        try:
            # Record individual interaction
            interactions_ref = self.db.collection('movie_interactions')
            interactions_ref.add({
                'user_id': interaction.user_id,
                'movie_id': interaction.movie_id,
                'action': interaction.action,
                'room_id': interaction.room_id,
                'timestamp': firestore.SERVER_TIMESTAMP
            })

            # Update movie statistics
            stats_ref = self.db.collection('movie_statistics').document(str(interaction.movie_id))
            stats_doc = stats_ref.get()

            if stats_doc.exists:
                stats = stats_doc.to_dict()
            else:
                stats = {
                    'total_watches': 0,
                    'interested_count': 0,
                    'not_interested_count': 0,
                    'not_sure_count': 0
                }

            # Update counters based on swipe direction
            if interaction.action == SwipeAction.WATCHED_LIKED:
                stats['total_watches'] += 1
            elif interaction.action == SwipeAction.INTERESTED:
                stats['interested_count'] += 1
            elif interaction.action == SwipeAction.NOT_INTERESTED:
                stats['not_interested_count'] += 1
            elif interaction.action == SwipeAction.NOT_SURE:
                stats['not_sure_count'] += 1

            # Calculate engagement ratio
            total_interactions = (stats['total_watches'] + stats['interested_count'] +
                                stats['not_interested_count'] + stats['not_sure_count'])
            
            if total_interactions > 0:
                stats['engagement_ratio'] = (stats['total_watches'] + stats['interested_count']) / total_interactions
            else:
                stats['engagement_ratio'] = 0.0

            # Update or create statistics document
            stats_ref.set(stats, merge=True)

            return True

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_movie_statistics(self, movie_id: int) -> MovieStats:
        try:
            stats_doc = self.db.collection('movie_statistics').document(str(movie_id)).get()
            
            if not stats_doc.exists:
                return MovieStats(
                    movie_id=movie_id,
                    total_watches=0,
                    interested_count=0,
                    not_interested_count=0,
                    not_sure_count=0,
                    engagement_ratio=0.0
                )

            stats = stats_doc.to_dict()
            return MovieStats(
                movie_id=movie_id,
                **stats
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    async def record_batch_interactions(self, batch: BatchInteraction):
        try:
            # Start a batch write
            batch_write = self.db.batch()
        
            # Create references
            interactions_ref = self.db.collection('movie_interactions')
            stats_ref = self.db.collection('movie_statistics')
            room_ref = self.db.collection('room_statistics').document(batch.room_id)

            room_doc = room_ref.get()
            room_stats = RoomStatistics(room_id=batch.room_id)
            if room_doc.exists:
                room_stats = RoomStatistics(**room_doc.to_dict())

            # Group swipes by movie_id for efficient stats updates
            movie_stats = {}
        
            # Process each interaction
            for interaction in batch.swipes:
                # Add interaction document
                doc_ref = interactions_ref.document()
                batch_write.set(doc_ref, {
                    'user_id': batch.user_id,
                    'movie_id': interaction.movie_id,
                    'action': interaction.action,
                    'room_id': batch.room_id,
                    'timestamp': firestore.SERVER_TIMESTAMP
                })

                movie_id = interaction.movie_id
                if interaction.action == SwipeAction.WATCHED_LIKED and movie_id not in room_stats.watched_movie_ids:
                    room_stats.watched_movie_ids.append(movie_id)
                elif interaction.action == SwipeAction.INTERESTED and movie_id not in room_stats.interested_movie_ids:
                    room_stats.interested_movie_ids.append(movie_id)
                elif interaction.action == SwipeAction.NOT_INTERESTED and movie_id not in room_stats.not_interested_movie_ids:
                    room_stats.not_interested_movie_ids.append(movie_id)
                elif interaction.action == SwipeAction.NOT_SURE and movie_id not in room_stats.not_sure_movie_ids:
                    room_stats.not_sure_movie_ids.append(movie_id)
            
                # Aggregate stats updates
                if interaction.movie_id not in movie_stats:
                    movie_stats[interaction.movie_id] = {
                        'total_watches': 0,
                        'interested_count': 0,
                        'not_interested_count': 0,
                        'not_sure_count': 0
                    }
            
                stats = movie_stats[interaction.movie_id]
                if interaction.action == SwipeAction.WATCHED_LIKED:
                    stats['total_watches'] += 1
                elif interaction.action == SwipeAction.INTERESTED:
                    stats['interested_count'] += 1
                elif interaction.action == SwipeAction.NOT_INTERESTED:
                    stats['not_interested_count'] += 1
                elif interaction.action == SwipeAction.NOT_SURE:
                    stats['not_sure_count'] += 1
        
            # Add room stats update to batch
            batch_write.set(room_ref, room_stats.dict())

            # Update stats for each movie
            for movie_id, stats_update in movie_stats.items():
                stats_doc_ref = stats_ref.document(str(movie_id))
                batch_write.set(stats_doc_ref, stats_update, merge=True)
        
            # Commit the batch
            batch_write.commit()
            return True
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        

    async def update_room_stats(self, batch_interaction):
        room_ref = self.db.collection('room_statistics').document(batch_interaction.room_id)
        
        try:
            room_doc = room_ref.get()
            stats = RoomStatistics(room_id=batch_interaction.room_id)
            
            if room_doc.exists:
                data = room_doc.to_dict()
                stats = RoomStatistics(**data)
            
            for interaction in batch_interaction.swipes:
                movie_id = interaction.movie_id
                if interaction.action == SwipeAction.WATCHED_LIKED:
                    if movie_id not in stats.watched_movie_ids:
                        stats.watched_movie_ids.append(movie_id)
                elif interaction.action == SwipeAction.INTERESTED:
                    if movie_id not in stats.interested_movie_ids:
                        stats.interested_movie_ids.append(movie_id)
                elif interaction.action == SwipeAction.NOT_INTERESTED:
                    if movie_id not in stats.not_interested_movie_ids:
                        stats.not_interested_movie_ids.append(movie_id)
                elif interaction.action == SwipeAction.NOT_SURE:
                    if movie_id not in stats.not_sure_movie_ids:
                        stats.not_sure_movie_ids.append(movie_id)
            
            room_ref.set(stats.dict())
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update room stats: {str(e)}")

    async def get_room_stats(self, room_id: str) -> RoomStatistics:
        try:
            room_doc = self.db.collection('room_statistics').document(room_id).get()
            if not room_doc.exists:
                return RoomStatistics(room_id=room_id)
            
            return RoomStatistics(**room_doc.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch room stats: {str(e)}")
