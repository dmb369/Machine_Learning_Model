import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def data_preprocessing(task_1a_dataframe):
    df = task_1a_dataframe.copy()

    current_year = 2023
    df['JoiningYear'] = current_year - df['JoiningYear']
    
    education_mapping = {
        'Bachelors': 0,
        'Masters': 1,
        'PHD': 2,
    }

    location_mapping = {
        'Bangalore': 0,
        'Pune': 1,
        'New Delhi': 2,
    }

    gender_mapping = {
        'Male': 0,
        'Female': 1,
    }

    benched_mapping = {
        'No': 0,
        'Yes': 1,
    }

    df['Education'] = df['Education'].map(education_mapping)
    df['City'] = df['City'].map(location_mapping)
    df['Gender'] = df['Gender'].map(gender_mapping)
    df['EverBenched'] = df['EverBenched'].map(benched_mapping)

    # categorical_features = ['Education', 'City', 'Gender', 'EverBenched']
    # df = pd.get_dummies(df, columns=categorical_features)

    return df

# Identify Features and Targets
def identify_features_and_targets(encoded_dataframe):
    target_column = 'LeaveOrNot'
    
    features = [col for col in encoded_dataframe.columns if col != target_column]

    features_and_targets = [features, target_column]

    return features_and_targets

# Load Data as Tensors
def load_as_tensors(features_and_targets):
    batch_size = 256
    validation_split = 0.25
    
    df = encoded_dataframe

    feature_columns = features_and_targets[0]
    target_column = features_and_targets[1]

    X = df[feature_columns].values
    y = df[target_column].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    total_samples = len(X_tensor)
    split_idx = int(total_samples * (1 - validation_split))

    X_train_tensor = X_tensor[:split_idx]
    y_train_tensor = y_tensor[:split_idx]
    X_val_tensor = X_tensor[split_idx:]
    y_val_tensor = y_tensor[split_idx:]

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    tensors_and_iterable_training_data = [X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, train_loader, val_loader]

    return tensors_and_iterable_training_data

# Define the Neural Network Model
class Salary_Predictor(nn.Module):
    def __init__(self, input_size):
        super(Salary_Predictor, self).__init__()
        
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, 64)  # Input size to 64 hidden units
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 32)          # 64 hidden units to 32 hidden units
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(32, 1)           # 32 hidden units to 1 output unit (binary classification)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Define the activation functions (e.g., ReLU) between layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))  # Using sigmoid activation for binary classification
        
        return x

# Define Loss Function
def model_loss_function():
    return nn.BCELoss()  # Binary Cross-Entropy Loss

# Define Optimizer
def model_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001

# Define Number of Epochs
def model_number_of_epochs():
    return 500# You can adjust the number of epochs as needed

# Training Function
def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    X_train_tensor, _, y_train_tensor, _, train_loader, _ = tensors_and_iterable_training_data
    
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape the target tensor to match the shape of predicted
            targets = targets.view(-1, 1)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
    return model

# Validation Function
def validation_function(trained_model, tensors_and_iterable_training_data):
    _, X_val_tensor, _, y_val_tensor, _, val_loader = tensors_and_iterable_training_data
    trained_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = trained_model(inputs)
            predicted = (outputs > 0.5).float()
            
            # Reshape the target tensor to match the shape of predicted
            targets = targets.view(-1, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return accuracy

# Main Function
if __name__ == "__main__":
    # Load your dataset as 'task_1a_dataframe'
    # ...
    task_1a_dataframe= pd.read_csv('/Users/devmbandhiya/Desktop/Task_1A/task_1a_dataset.csv')

    # Data preprocessing
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # Identify features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # Load data as tensors
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

    # Initialize the model
    input_size = len(features_and_targets[0])
    model = Salary_Predictor(input_size)

    # Define loss function, optimizer, and number of epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # Train the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

    # Validate the model
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    print(f"Validation Accuracy: {model_accuracy * 100:.2f}%")