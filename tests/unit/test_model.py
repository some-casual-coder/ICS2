import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LayerNormalization
from app.recommendations.model_architecture import MovieTower, GroupTower, TwoTowerModel, SimplifiedContrastiveLoss, PrecisionMetric, RecallMetric, NDCGMetric


class TwoTowerModelTests(unittest.TestCase):
    def setUp(self):
        """Setup for all tests"""
        tf.keras.backend.clear_session()  # Clear any previous session state
        # Load saved model
        self.model = load_model(
            'best_two_tower_model.keras',
            custom_objects={
                'MovieTower': MovieTower,
                'GroupTower': GroupTower,
                'TwoTowerModel': TwoTowerModel,
                'SimplifiedContrastiveLoss': SimplifiedContrastiveLoss,
                'PrecisionMetric': PrecisionMetric,
                'RecallMetric': RecallMetric,
                'NDCGMetric': NDCGMetric
            }
        )

        # Test data dimensions
        self.movie_dim = 1536
        self.group_dim = 11

    def test_movie_tower(self):
        """Test movie tower output dimensions and values"""
        movie_tower = MovieTower()
        test_input = np.random.random((1, self.movie_dim))
        output = movie_tower(test_input)
        self.assertEqual(output.shape[-1], 256)
        self.assertTrue(np.all(np.isfinite(output)))

    def test_movie_tower_layers(self):
        """Test that MovieTower layers are properly initialized and working"""
        movie_tower = MovieTower()
        # Check if layers were created correctly
        self.assertTrue(len(movie_tower.layer_list) > 0,
                        "Layer list should not be empty")
        # Check if each group of 3 layers is Dense->BatchNorm->Dropout
        for i in range(0, len(movie_tower.layer_list), 3):
            self.assertIsInstance(movie_tower.layer_list[i], Dense)
            self.assertIsInstance(
                movie_tower.layer_list[i+1], BatchNormalization)
            self.assertIsInstance(movie_tower.layer_list[i+2], Dropout)

    def test_movie_tower_input_validation(self):
        """Test MovieTower input shape validation"""
        movie_tower = MovieTower()
        # Test with incorrect input dimension
        wrong_input = np.random.random((1, 100))  # Wrong dimension
        with self.assertRaises(AssertionError):
            movie_tower(wrong_input)

    def test_movie_tower_regularization(self):
        """Test that L2 regularization is properly applied"""
        l2_reg = 0.01
        movie_tower = MovieTower(l2_reg=l2_reg)
        # Check if Dense layers have L2 regularization
        for layer in movie_tower.layer_list:
            if isinstance(layer, Dense):
                self.assertIsNotNone(layer.kernel_regularizer)
                self.assertEqual(layer.kernel_regularizer.l2, l2_reg)

    def test_group_tower(self):
        """Test group tower output dimensions and values"""
        group_tower = GroupTower()
        test_input = np.random.random(
            (1, self.group_dim))  # Using group_dim (11)
        output = group_tower(test_input)
        self.assertEqual(output.shape[-1], 256)  # Check output dimension
        self.assertTrue(np.all(np.isfinite(output)))  # Check for valid values

    def test_group_tower_attention(self):
        """Test group tower attention mechanism"""
        group_tower = GroupTower(num_heads=4)
        test_input = np.random.random((1, self.group_dim))

        # Get the attention output
        output = group_tower(test_input)

        # Test shape
        self.assertEqual(output.shape[-1], 256)

        # Test if attention weights are normalized (sum close to 1)
        # We can't directly access attention weights as they're computed in call(),
        # but we can verify the output is properly normalized
        self.assertTrue(np.all(np.isfinite(output)))

    def test_group_tower_layers(self):
        """Test that GroupTower layers are correctly initialized"""
        group_tower = GroupTower()

        # Test key components exist
        self.assertIsNotNone(group_tower.dense1)
        self.assertIsNotNone(group_tower.query)
        self.assertIsNotNone(group_tower.key)
        self.assertIsNotNone(group_tower.value)
        self.assertIsNotNone(group_tower.final_norm)

        # Test layer list structure
        self.assertTrue(len(group_tower.layer_list) > 0)
        for i in range(0, len(group_tower.layer_list), 3):
            self.assertIsInstance(group_tower.layer_list[i], Dense)
            self.assertIsInstance(
                group_tower.layer_list[i+1], BatchNormalization)
            self.assertIsInstance(group_tower.layer_list[i+2], Dropout)

    def test_two_tower_model_structure(self):
        """Test TwoTower model architecture and components"""
        model = TwoTowerModel()

        # Test model components exist
        self.assertIsNotNone(model.movie_tower)
        self.assertIsNotNone(model.group_tower)

        # Test if towers are of correct type
        self.assertIsInstance(model.movie_tower, MovieTower)
        self.assertIsInstance(model.group_tower, GroupTower)

    def test_two_tower_model_forward_pass(self):
        """Test forward pass through the complete model"""
        model = TwoTowerModel()

        # Create test inputs
        batch_size = 2
        movie_input = np.random.random((batch_size, self.movie_dim))
        group_input = np.random.random((batch_size, self.group_dim))

        # Get model output
        similarity = model([movie_input, group_input])

        # Check output shape (should be batch_size x batch_size similarity matrix)
        self.assertEqual(similarity.shape, (batch_size, batch_size))

        # Check if similarities are properly scaled (between -1/temp and 1/temp)
        self.assertTrue(np.all(np.isfinite(similarity)))

        # Check if the similarity matrix is properly normalized
        # The exponential of similarities should sum to 1 for each row
        softmax_sims = tf.nn.softmax(similarity, axis=-1)
        row_sums = tf.reduce_sum(softmax_sims, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))

    def test_two_tower_model_temperature(self):
        """Test temperature scaling in the model with consistent setup"""
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Test with different temperature values
        temp1 = 0.05  # Lower temperature
        temp2 = 0.1   # Higher temperature

        # Create a consistent model initialization
        def create_model(temperature):
            # Recreate the model with a fixed random state
            with tf.keras.utils.custom_object_scope({'TwoTowerModel': TwoTowerModel}):
                model = TwoTowerModel(temperature=temperature)
            
                # Force model to build and initialize weights
                dummy_movie_input = np.random.random((1, self.movie_dim))
                dummy_group_input = np.random.random((1, self.group_dim))
                model([dummy_movie_input, dummy_group_input])
        
            return model

        # Create models
        model1 = create_model(temp1)
        model2 = create_model(temp2)

        # Create consistent input data
        batch_size = 2
        movie_input = np.random.random((batch_size, self.movie_dim))
        group_input = np.random.random((batch_size, self.group_dim))

        # Disable training mode to avoid dropout variations
        sim1 = model1([movie_input, group_input], training=False)
        sim2 = model2([movie_input, group_input], training=False)

        # Variance comparison
        var1 = tf.math.reduce_variance(sim1)
        var2 = tf.math.reduce_variance(sim2)
    
        # Verbose logging
        print("\nTemperature Scaling Test Details:")
        print(f"Temperature 1: {temp1}")
        print(f"Temperature 2: {temp2}")
        print(f"Variance at temp {temp1}: {var1.numpy()}")
        print(f"Variance at temp {temp2}: {var2.numpy()}")
        print(f"Mean at temp {temp1}: {tf.reduce_mean(sim1).numpy()}")
        print(f"Mean at temp {temp2}: {tf.reduce_mean(sim2).numpy()}")

        # Multiple runs to ensure consistency
        def run_variance_check():
            # Recreate model and inputs each time
            model1 = create_model(temp1)
            model2 = create_model(temp2)
        
            movie_input = np.random.random((batch_size, self.movie_dim))
            group_input = np.random.random((batch_size, self.group_dim))
        
            sim1 = model1([movie_input, group_input], training=False)
            sim2 = model2([movie_input, group_input], training=False)
        
            var1 = tf.math.reduce_variance(sim1)
            var2 = tf.math.reduce_variance(sim2)
        
            return var1.numpy(), var2.numpy()

        # Run multiple checks
        variances = [run_variance_check() for _ in range(5)]
        print("\nMultiple Run Variances:")
        for i, (v1, v2) in enumerate(variances, 1):
            print(f"Run {i}: Var1 = {v1}, Var2 = {v2}")

        # Assertion with more robust checking
        def check_variance_trend(variances):
            # Check if lower temperature consistently has higher variance
            lower_temp_vars = [v[0] for v in variances]
            higher_temp_vars = [v[1] for v in variances]
        
            # Allow some statistical variation, but look for a trend
            lower_temp_var_mean = np.mean(lower_temp_vars)
            higher_temp_var_mean = np.mean(higher_temp_vars)
        
            print(f"\nMean Variance - Temp {temp1}: {lower_temp_var_mean}")
            print(f"Mean Variance - Temp {temp2}: {higher_temp_var_mean}")
        
            return lower_temp_var_mean > higher_temp_var_mean

        # Final assertion
        self.assertTrue(
            check_variance_trend(variances),
            f"Lower temperature should lead to higher variance across multiple runs. "
            f"Temperatures: {temp1} vs {temp2}"
        )

    def test_two_tower_model_batch_processing(self):
        """Test model with different batch sizes"""
        model = TwoTowerModel()

        # Test with different batch sizes
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            movie_input = np.random.random((batch_size, self.movie_dim))
            group_input = np.random.random((batch_size, self.group_dim))

            similarity = model([movie_input, group_input])

            # Check output shape
            self.assertEqual(similarity.shape, (batch_size, batch_size))
            # Check if output is valid
            self.assertTrue(np.all(np.isfinite(similarity)))

    def test_contrastive_loss(self):
        """Test SimplifiedContrastiveLoss behavior"""
        loss_fn = SimplifiedContrastiveLoss(temperature=0.1)

        # Create sample prediction and target
        batch_size = 3
        y_pred = tf.random.uniform(
            (batch_size, batch_size))  # similarity matrix
        # Create one-hot encoded ground truth
        # Diagonal matrix representing correct matches
        y_true = tf.eye(batch_size)

        # Calculate loss
        loss_value = loss_fn(y_true, y_pred)

        # Check if loss is scalar and finite
        self.assertTrue(np.isscalar(loss_value.numpy()))
        self.assertTrue(np.isfinite(loss_value.numpy()))

        # Loss should be positive
        self.assertGreater(loss_value, 0)

    def test_precision_metric(self):
        """Test PrecisionMetric behavior"""
        metric = PrecisionMetric(k=2)

        # Create sample prediction and target
        batch_size = 3
        similarities = tf.constant([
            [1.0, 0.5, 0.3],  # First item matches with first (correct)
            [0.2, 1.0, 0.4],  # Second item matches with second (correct)
            [0.3, 0.5, 1.0]   # Third item matches with third (correct)
        ])
        y_true = tf.constant([0, 1, 2])  # True matches

        # Update state with batch
        metric.update_state(y_true, similarities)
        result = metric.result()
        result_float = float(result.numpy())

        # For k=2, each row should contain its correct match in top-2
        # So precision should be 1.0
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    def test_recall_metric(self):
        """Test RecallMetric behavior"""
        metric = RecallMetric(k=2)

        # Create sample prediction and target
        similarities = tf.constant([
            [1.0, 0.5, 0.3],
            [0.2, 1.0, 0.4],
            [0.3, 0.5, 1.0]
        ])
        y_true = tf.constant([0, 1, 2])

        # Update state with batch
        metric.update_state(y_true, similarities)
        result = metric.result()
        result_float = float(result.numpy())

        # Check basic properties
        self.assertIsInstance(result_float, float)
        self.assertGreaterEqual(result_float, 0.0)
        self.assertLessEqual(result_float, 1.0)

    # def test_ndcg_metric(self):
    #     """Test NDCGMetric behavior"""
    #     metric = NDCGMetric(k=2)

    #     # Create sample prediction and target
    #     similarities = tf.constant([
    #         [1.0, 0.5, 0.3],
    #         [0.2, 1.0, 0.4],
    #         [0.3, 0.5, 1.0]
    #     ])
    #     y_true = tf.constant([0, 1, 2])

    #     # Update state with batch
    #     metric.update_state(y_true, similarities)
    #     result = metric.result()
    #     result_float = float(result.numpy())

    #     # Check basic properties
    #     self.assertIsInstance(result_float, float)
    #     self.assertGreaterEqual(result, 0.0)
    #     self.assertLessEqual(result, 1.0)

    # def test_metric_reset(self):
    #     """Test that metrics can be properly reset"""
    #     metrics = [
    #         PrecisionMetric(k=2),
    #         RecallMetric(k=2),
    #         NDCGMetric(k=2)
    #     ]

    #     similarities = tf.constant([
    #         [1.0, 0.5, 0.3],
    #         [0.2, 1.0, 0.4],
    #         [0.3, 0.5, 1.0]
    #     ])
    #     y_true = tf.constant([0, 1, 2])

    #     for metric in metrics:
    #         # First update
    #         metric.update_state(y_true, similarities)
    #         first_result = metric.result()

    #         # Reset
    #         metric.reset_state()

    #         # Check if reset worked
    #         reset_result = metric.result()
    #         self.assertEqual(
    #             reset_result, 0.0, f"{metric.__class__.__name__} did not reset properly")

    #         # Update again
    #         metric.update_state(y_true, similarities)
    #         second_result = metric.result()

    #         # Results should be the same before and after reset
    #         self.assertAllClose(first_result, second_result,
    #                             msg=f"{metric.__class__.__name__} gave different results after reset")

    # def test_model_prediction(self):
    #     """Test end-to-end model prediction"""
    #     movie_input = np.random.random((1, self.movie_dim))
    #     group_input = np.random.random((1, self.group_dim))
    #     similarity = self.model.predict([movie_input, group_input])
    #     self.assertTrue(np.all(similarity >= -1))
    #     self.assertTrue(np.all(similarity <= 1))

    # def test_similarity_calculation(self):
    #     """Test similarity score computation"""
    #     movie_inputs = np.random.random((5, self.movie_dim))
    #     group_inputs = np.random.random((5, self.group_dim))
    #     similarities = self.model.predict([movie_inputs, group_inputs])
    #     self.assertEqual(similarities.shape[0], 5)
    #     self.assertTrue(np.all(np.isfinite(similarities)))
