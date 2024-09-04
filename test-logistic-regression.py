import torch
from logistic_regression import LogisticRegressionModel

def test_logistic_regression():
    # Define input dimensions
    input_dim = 4  # Number of features in the Iris dataset
    output_dim = 3  # Number of classes

    # Initialize the model
    model = LogisticRegressionModel(input_dim, output_dim)

    # Create a dummy input tensor
    X_dummy = torch.randn(1, input_dim)  # 1 sample with 4 features

    # Run the model on the dummy input
    output = model(X_dummy)

    # Check the output dimensions
    assert output.shape == (1, output_dim), "Output shape is incorrect"

if __name__ == "__main__":
    pytest.main()
