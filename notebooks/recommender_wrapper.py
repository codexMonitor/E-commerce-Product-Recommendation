import numpy as np

class Recommender:

    def __init__(self,
                 model,
                 interaction_matrix,
                 user_encoder,
                 item_encoder):

        self.model = model
        self.interaction_matrix = interaction_matrix
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def recommend(self, visitor_id, top_k=10):
        
        # 1. Safely attempt to encode the user
        try:
            encoded_user = self.user_encoder.transform([visitor_id])[0]
        except ValueError:
            # COLD START: The visitor_id was not in the training data.
            print(f"Notice: Visitor {visitor_id} is new. Returning empty recommendations.")
            return []

        # 2. Generate recommendations for known users
        ids, scores = self.model.recommend(
            userid=encoded_user,
            user_items=self.interaction_matrix[encoded_user],
            N=top_k,
            filter_already_liked_items=True
        )

        # 3. Decode the item IDs back to their original retail dataset IDs
        original_items = self.item_encoder.inverse_transform(ids)

        # 4. Return as a list of dictionaries for the frontend
        return [
            {
                "item_id": int(item),
                "score": float(score)
            }
            for item, score in zip(original_items, scores)
        ]