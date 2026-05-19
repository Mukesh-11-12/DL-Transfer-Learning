# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Developing a Neural Network Classification Model using Transfer Learning

Training deep neural networks from scratch requires large datasets, high computational power, and significant training time. In many practical scenarios, such resources are limited.

The goal of this project is to develop an image classification model using Transfer Learning, where a pre-trained neural network is reused and fine-tuned to classify new data efficiently.

<img width="415" height="114" alt="image" src="https://github.com/user-attachments/assets/31251361-c423-4bb3-a769-876f1831d1c3" />




## Neural Network Model
A Neural Network Model is a type of machine learning model inspired by the structure and functioning of the human brain. It is used to learn patterns from data and make predictions or decisions.
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d43e5227-d911-4186-b071-3b68b96e5d3f" />

## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

### Name: D.MUKESH

### Register Number: 212224040204

```python
# Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
# write your code here
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
# Write your code here
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


# Train the Model
def train_model(model, train_loader,test_loader,num_epochs=100):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for images,labels in test_loader:
                images,labels=images.to(device),labels.to(device)
                outputs=model(images)
                loss=criterion(outputs,labels.unsqueeze(1).float())
                val_loss=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:D.MUKESH")
    print("Register Number: 212224040204")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="818" height="770" alt="image" src="https://github.com/user-attachments/assets/d5c96534-6b40-43af-be9e-e3b4356a5ad5" />




## Confusion Matrix
<img width="786" height="683" alt="image" src="https://github.com/user-attachments/assets/6cd597d9-f460-44d3-9666-e11a0677cf21" />






## Classification Report
<img width="528" height="239" alt="image" src="https://github.com/user-attachments/assets/3be805a2-71de-4800-937e-cbf2bfc6d0c4" />




### New Sample Data Prediction
<img width="309" height="597" alt="image" src="https://github.com/user-attachments/assets/98217d59-a217-44f0-997a-abaf40dd4dc0" />



## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
