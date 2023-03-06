import torch
torch.manual_seed(42)

import numpy as np
import shutil
import os
from os.path import join
import json
import warnings
warnings.filterwarnings("ignore")


# Train model
def train(model, 
          input_size,
          classes,
          train_dataloader, 
          test_dataloader, 
          n_epochs,
          batch_size,
          criterion,
          optimizer,
          learning_rate,
          device, 
          project_dir,
          models_dir,
          n_epochs_trained=0):
    
    history = {"train_losses": [],
            "val_losses": [],
            "train_accuracy": [],
            "val_accuracy": []}

    for epoch in range(n_epochs):
        print(f"\nStarting training phase for epoch {epoch + 1}/{n_epochs}")
        model.train()
        running_loss = 0.0
        n_correct = 0
        n_predictions = 0

        for i, data in enumerate(train_dataloader, 0):
            # get the inputs- data is a batch of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None

            # forward pass
            outputs = model(inputs) # predict label for each image
            loss = criterion(outputs, labels) # calculate batch loss
            running_loss += loss.item()

            # backward pass
            loss.backward()
            optimizer.step()

            # calculate batch accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            n_correct += batch_correct
            n_predictions += labels.size(0)

            # print statistics
            n = 1
            if i % n == n-1:    # print every n mini-batches
                print(f'[Epoch: {epoch + 1}/{n_epochs} Batch: {i+1}/{len(train_dataloader)}] loss: {running_loss / (n*(i+1)):.4f} accuracy: {n_correct / n_predictions * 100:.4f} batch_accuracy: {batch_correct / labels.size(0) * 100:.4f}', end='\r')
            if i == len(train_dataloader) - 1:
                print(f'[Epoch: {epoch + 1}/{n_epochs} Batch: {i+1}/{len(train_dataloader)}] loss: {running_loss / (n*(i+1)):.4f} accuracy: {n_correct / n_predictions * 100:.4f} batch_accuracy: {batch_correct / labels.size(0) * 100:.4f}')

        
        history["train_losses"].append(running_loss / len(train_dataloader))
        history["train_accuracy"].append(n_correct / n_predictions * 100)
        running_loss = 0.0
        n_correct = 0
        n_predictions = 0

        # Validate model
        print(f"Starting validation phase for epoch {epoch + 1}/{n_epochs}")
        with torch.no_grad():
            model.eval() 

            for i, data in enumerate(test_dataloader, 0):
                # get the inputs- data is a batch of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # calculate accuracy for batch
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                n_correct += batch_correct
                n_predictions += labels.size(0)

                print(f'[Epoch: {epoch + 1}/{n_epochs} Batch: {i+1}/{len(test_dataloader)}] accuracy: {n_correct / n_predictions * 100:.4f}' , end='\r')

        # calculate val loss for epoch
        history["val_losses"].append(running_loss / len(test_dataloader))
        
        # calculate accuracy for epoch
        accuracy = n_correct / n_predictions * 100
        accuracy = round(accuracy, 4)
        
        print("[Train loss: %.4f],  [Test loss: %.4f],  [Validation Accuracy: %.2f]" \
            %(history["train_losses"][-1], history["val_losses"][-1], accuracy))
        
        # save model if it is the first one OR has higher accuracy
        if len(history["val_accuracy"]) == 0 or accuracy > max(history["val_accuracy"]):
            if not os.path.isdir(join(project_dir, models_dir)): 
                os.mkdir(join(project_dir, models_dir))

            # if model folder exists, delete it
            models_folder = join(project_dir, models_dir, "model")

            if os.path.isdir(models_folder):
                shutil.rmtree(models_folder, ignore_errors=True)

            os.mkdir(models_folder)

            model_save_path = join(models_folder, "model.pt")

            state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    }

            torch.save(state, model_save_path)
            print(f"Saved model with accuracy {accuracy}")

        history["val_accuracy"].append(accuracy)
        
    print("Finished training")

    # add model accuracy to folder's name
    best_epoch_ind = int(np.argmax(history["val_accuracy"]))
    new_foldername = join(project_dir, models_dir, f"model_{history['val_accuracy'][best_epoch_ind]}")

    # change model name if there is a naming conflict
    while os.path.isdir(join(project_dir, models_dir, new_foldername)):
        split_name = new_foldername.split('_')
        if len(split_name) == 2:
            new_foldername += "_1"
        elif len(split_name) == 3:
            new_foldername = f"{split_name[0]}_{split_name[1]}_{int(split_name[2]) + 1}"

    os.rename(join(project_dir, models_dir, "model"), join(project_dir, models_dir, new_foldername))

    # save hyperparameters json file in model folder

    hp_dict = {"size" : input_size, 
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "epochs": n_epochs,
            "totalEpochs": best_epoch_ind + 1 + n_epochs_trained}
    
    with open(join(new_foldername, "hyperparameters.json"), 'w', encoding='utf-8') as f:
        json.dump(hp_dict, f, ensure_ascii=False, indent=4)
        
    print("\nWrote hyperparameters to:")
    print(join(new_foldername, "hyperparameters.json"))

    # write classes to classes.txt file in model folder
    with open(join(new_foldername, "classes.txt"), 'w') as f:
        for c in classes:
            f.write(c + '\n')

    print("\nWrote class labels to:")
    print(join(new_foldername, "class_labels.txt"))

    return history

