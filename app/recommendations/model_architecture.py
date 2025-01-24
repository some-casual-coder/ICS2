import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable 
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.losses import Loss

@register_keras_serializable()
class MovieTower(tf.keras.layers.Layer):
    def __init__(self, hidden_dims=[512, 256], dropout_rate=0.3, input_dim=1536, l2_reg=1e-5, **kwargs):
        super(MovieTower, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.l2_reg = l2_reg
        
        self.layer_list = []
        for dim in hidden_dims:
            self.layer_list.append(Dense(dim, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg)))
            self.layer_list.append(BatchNormalization())
            self.layer_list.append(Dropout(dropout_rate))
        
        self.final_norm = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layer_list:
            x = layer(x, training=training)
        return self.final_norm(x)

    def build(self, input_shape):
        assert input_shape[-1] == 1536, f"Expected input shape (-1) to be 1536, got {input_shape[-1]}"
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "input_dim": self.input_dim,
            "l2_reg": self.l2_reg
        })
        return config

@register_keras_serializable()
class GroupTower(tf.keras.layers.Layer):
    def __init__(self, hidden_dims=[512, 256], num_heads=4, dropout_rate=0.3, l2_reg=1e-5, **kwargs):
        super(GroupTower, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dims[0]
        self.num_heads = num_heads
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        self.dense1 = Dense(self.hidden_dim, activation='gelu', kernel_regularizer=regularizers.l2(self.l2_reg))
        self.query = Dense(self.hidden_dim)
        self.key = Dense(self.hidden_dim)
        self.value = Dense(self.hidden_dim)
        
        self.layer_list = []
        for dim in hidden_dims:
            self.layer_list.append(Dense(dim, activation='gelu'))
            self.layer_list.append(BatchNormalization())
            self.layer_list.append(Dropout(dropout_rate))
            
        self.final_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.hidden_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        x = tf.matmul(attention_weights, v)
        
        for layer in self.layer_list:
            x = layer(x, training=training)
            
        return self.final_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg
        })
        return config

@register_keras_serializable()
class TwoTowerModel(tf.keras.Model):
    def __init__(
        self,
        movie_dim=1536,
        group_dim=11,
        hidden_dims=[512, 256],
        num_heads=4,
        dropout_rate=0.3,
        temperature=0.5,
        l2_reg=1e-5,
        **kwargs
    ):
        super(TwoTowerModel, self).__init__(**kwargs)
        self.movie_tower = MovieTower(hidden_dims, dropout_rate, input_dim=movie_dim, l2_reg=l2_reg)
        self.group_tower = GroupTower(hidden_dims, num_heads, dropout_rate, l2_reg=l2_reg)
        self.temperature = temperature
        
        # Save for config
        self.movie_dim = movie_dim
        self.group_dim = group_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def call(self, inputs, training=False):
        movie_input, group_input = inputs
    
        movie_embedding = self.movie_tower(movie_input, training=training)
        group_embedding = self.group_tower(group_input, training=training)
    
        # Add logging
        tf.print("Movie embedding stats:", 
             tf.reduce_mean(movie_embedding), 
             tf.reduce_min(movie_embedding), 
             tf.reduce_max(movie_embedding))
        tf.print("Group embedding stats:", 
             tf.reduce_mean(group_embedding), 
             tf.reduce_min(group_embedding), 
             tf.reduce_max(group_embedding))
    
        normalized_movie_emb = tf.math.l2_normalize(movie_embedding, axis=1)
        normalized_group_emb = tf.math.l2_normalize(group_embedding, axis=1)
    
        similarity = tf.matmul(normalized_movie_emb, normalized_group_emb, transpose_b=True)
    
        # Log raw similarity
        tf.print("Raw similarity stats:", 
             tf.reduce_mean(similarity), 
             tf.reduce_min(similarity), 
             tf.reduce_max(similarity))
    
        similarity = similarity / self.temperature
    
        # Log after temperature
        tf.print("After temperature stats:", 
             tf.reduce_mean(similarity), 
             tf.reduce_min(similarity), 
             tf.reduce_max(similarity))
    
        # similarity = similarity - tf.reduce_max(similarity, axis=1, keepdims=True)
        similarity = tf.nn.sigmoid(similarity)
    
        return similarity

    def get_config(self):
        config = super().get_config()
        config.update({
            "movie_dim": self.movie_dim,
            "group_dim": self.group_dim,
            "hidden_dims": self.hidden_dims,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "temperature": self.temperature,
            "l2_reg": self.l2_reg
        })
        return config
    
@register_keras_serializable()
class SimplifiedContrastiveLoss(Loss):
    def __init__(self, temperature=0.1, margin=0.1, name='contrastive_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.margin = margin

    def call(self, y_true, y_pred):
        # Compute similarities
        similarities = y_pred

        # Apply temperature scaling
        scaled_similarities = similarities / self.temperature

        # Compute log probabilities
        log_probabilities = tf.nn.log_softmax(scaled_similarities, axis=-1)

        # Mask for positive pairs
        positive_mask = tf.cast(y_true > 0, tf.float32)

        # Negative log-likelihood for positive pairs
        loss = -tf.reduce_mean(
            tf.reduce_sum(log_probabilities * positive_mask, axis=-1)
        )

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "margin": self.margin
        })
        return config


@register_keras_serializable()
class PrecisionMetric(tf.keras.metrics.Metric):
    def __init__(self, k=10, name=None, dtype=None, **kwargs):
        # Only set default name if name is not provided
        name = name or f'precision_at_{k}'
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.k = k
        self.total = self.add_weight(
            name='total', shape=(), initializer='zeros')
        self.count = self.add_weight(
            name='count', shape=(), initializer='zeros')

    def update_state(self, y_true, similarity_matrix, sample_weight=None):
        # Flatten y_true to [batch_size]
        y_true_flat = tf.reshape(y_true, [-1])

        # Get top-k indices based on similarity matrix
        _, top_k_indices = tf.nn.top_k(similarity_matrix, k=self.k)

        # Expand true labels to match top_k_indices shape
        y_true_expanded = tf.cast(
            tf.expand_dims(y_true_flat, axis=1), tf.int32)

        # Precision: count of relevant items in top-k / k
        is_relevant = tf.reduce_any(
            tf.equal(top_k_indices, y_true_expanded),
            axis=1
        )

        # Calculate precision as number of relevant items in top-k divided by k
        precision_per_sample = tf.cast(is_relevant, tf.float32)

        # Compute precision
        precision = tf.reduce_mean(precision_per_sample)

        # Update total and count
        self.total.assign_add(precision)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({
            "k": self.k
        })
        return config


@register_keras_serializable()
class RecallMetric(tf.keras.metrics.Metric):
    def __init__(self, k=10, name=None, dtype=None, **kwargs):
        # Only set default name if name is not provided
        name = name or f'recall_at_{k}'
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.k = k
        self.total = self.add_weight(
            name='total', shape=(), initializer='zeros')
        self.count = self.add_weight(
            name='count', shape=(), initializer='zeros')

    def update_state(self, y_true, similarity_matrix, sample_weight=None):
        # Flatten y_true to [batch_size]
        y_true_flat = tf.reshape(y_true, [-1])

        # Get top-k indices based on similarity matrix
        _, top_k_indices = tf.nn.top_k(similarity_matrix, k=self.k)

        # Calculate the total number of positive samples
        total_positives = tf.reduce_sum(
            tf.cast(tf.equal(y_true_flat, 1), tf.float32))

        # Ensure we don't divide by zero
        total_positives = tf.maximum(total_positives, 1.0)

        # Expand true labels to match top_k_indices shape
        y_true_expanded = tf.cast(
            tf.expand_dims(y_true_flat, axis=1), tf.int32)

        # Find correct predictions in top-k for each sample
        is_correct = tf.reduce_any(
            tf.equal(top_k_indices, y_true_expanded),
            axis=1
        )

        # Calculate number of correctly retrieved relevant items
        correct_retrievals = tf.reduce_sum(tf.cast(is_correct, tf.float32))

        # Calculate recall as (correctly retrieved relevant items) / (total relevant items)
        recall = correct_retrievals / total_positives

        # Clip recall to a reasonable range
        recall = tf.minimum(recall, 1.0)

        # Update total and count
        self.total.assign_add(recall)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({
            "k": self.k
        })
        return config


@register_keras_serializable()
class NDCGMetric(tf.keras.metrics.Metric):
    def __init__(self, k=10, name=None, dtype=None, **kwargs):
        # Only set default name if name is not provided
        name = name or f'ndcg_at_{k}'
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.k = k
        self.total = self.add_weight(
            name='total', shape=(), initializer='zeros')
        self.count = self.add_weight(
            name='count', shape=(), initializer='zeros')

    def update_state(self, y_true, similarity_matrix, sample_weight=None):
        # Reshape y_true to ensure it's 2D (batch_size, 1)
        y_true_flat = tf.reshape(y_true, [-1, 1])

        # Get top-k indices based on similarity matrix
        _, top_k_indices = tf.nn.top_k(similarity_matrix, k=self.k)

        # Create rank discount factors
        rank_range = tf.range(1, self.k + 1, dtype=tf.float32)
        discounts = tf.math.log(rank_range + 1)

        # Broadcast discounts to match batch size
        discounts_broadcast = tf.tile(tf.expand_dims(discounts, 0), [
                                      tf.shape(top_k_indices)[0], 1])

        # Compute DCG
        def compute_gain_and_dcg(indices, labels):
            # Gather true labels for top-k indices
            top_k_true = tf.gather(labels, indices, batch_dims=0)

            # Cast and clip labels to float32
            top_k_true_casted = tf.cast(
                tf.minimum(top_k_true, 1.0), tf.float32)

            # Squeeze top_k_true_casted to make it of shape [batch_size, k]
            top_k_true_casted = tf.squeeze(top_k_true_casted, axis=-1)

            # Compute gains divided by discounts
            gains_discounted = top_k_true_casted / discounts_broadcast

            # Compute DCG
            return tf.reduce_sum(gains_discounted, axis=1)

        # Compute DCG
        dcg = compute_gain_and_dcg(top_k_indices, y_true_flat)

        # Compute IDCG (ideal DCG) based on similarity_matrix (not y_true_flat)
        _, ideal_indices = tf.nn.top_k(similarity_matrix, k=self.k)
        idcg = compute_gain_and_dcg(ideal_indices, y_true_flat)

        # Compute NDCG
        ndcg = dcg / (idcg + tf.keras.backend.epsilon())

        # Clip and take mean
        ndcg = tf.reduce_mean(tf.minimum(ndcg, 1.0))

        # Update state
        self.total.assign_add(ndcg)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)

    def get_config(self):
        config = super().get_config()
        config.update({
            "k": self.k
        })
        return config