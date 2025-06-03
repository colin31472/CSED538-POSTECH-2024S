import os
import matplotlib.pyplot as plt
from datetime import datetime


def save_results(result_dict, config):
    current_time = datetime.now().strftime("%d_%H-%M")

    top1_acc = result_dict['top1_test_acc']
    top5_acc = result_dict['top5_test_acc']
    train_losses = result_dict['train_losses']
    valid_losses = result_dict['valid_losses']
    train_acc = result_dict['train_acc']
    valid_acc = result_dict['val_acc']
    lrs = result_dict['lrs']

    folder_path = f"results/{config.model_name}_{config.lr_scheduler_name}_{current_time}"
    os.makedirs(folder_path, exist_ok=True)
    
    # folder_path = f"/content/drive/MyDrive/results/{config.model_name}_{config.lr_scheduler_name}_{current_time}"
    # os.makedirs(folder_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f"train_losses_{current_time}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f"valid_losses_{current_time}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train Accuracy')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f"train_accuracy_plot_{current_time}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f"valid_accuracy_plot_{current_time}.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label='learning rate')
    plt.title('learning rate')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.legend()
    plt.savefig(os.path.join(folder_path, f"lr_plot_{current_time}.png"))
    plt.close()

    with open(os.path.join(folder_path, f"results_{current_time}.txt"), 'w') as f:
        f.write(f"Train Loss: {train_losses}\n")
        f.write(f"Validation Loss: {valid_losses}\n")
        f.write(f"Train Accuracy: {train_acc}\n")
        f.write(f"Validation Accuracy: {valid_acc}\n")
        f.write(f"Top-1 Accuracy: {top1_acc}\n")
        f.write(f"Top-5 Accuracy: {top5_acc}\n")
        f.write(f"lrs: {lrs}\n")
    return