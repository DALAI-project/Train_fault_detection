import torch
import onnx
import onnxruntime
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def set_seed(random_seed):
    """Function for setting random seed for the relevant libraries."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    print(f"Random seed set as {random_seed}")

def save_model(model, input_size, save_model_format, save_model_path, date):
    """Function for saving the model in .pth or .onnx format."""
    if save_model_format == 'onnx':
        onnx_model_path = os.path.join(save_model_path, 'densenet_' + date + '.onnx')
        # Random batch size
        batch_size = 1
        # Random input to the model (with correct dimensions)
        x = torch.randn(batch_size, 3, input_size, input_size, requires_grad=True)
        torch_out = model(x)

        # Export the model
        torch.onnx.export(model,                   # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        onnx_model_path,           # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})

        print('ONNX model saved to ', onnx_model_path)
        # Test transformed model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print('ONNX model checked.')

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        onnx_session = onnxruntime.InferenceSession(onnx_model_path)
        # compute ONNX Runtime output prediction
        onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(x)}
        onnx_out = onnx_session.run(None, onnx_inputs)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), onnx_out[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    else:
        pytorch_model_path = os.path.join(save_model_path, 'densenet_' + date + '.pth')
        torch.save(model, pytorch_model_path)
        print('Pytorch model saved to ', pytorch_model_path)


def plot_metrics(hist_dict, results_folder, date):
    """Function for plotting the training and validation results."""
    epochs = range(1, len(hist_dict['tr_loss'])+1)
    plt.plot(epochs, hist_dict['tr_loss'], 'g', label='Training loss')
    plt.plot(epochs, hist_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_folder + date + '_tr_val_loss.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_acc'], 'g', label='Training accuracy')
    plt.plot(epochs, hist_dict['val_acc'], 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(results_folder + date +  '_tr_val_acc.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_f1'], 'g', label='Training F1 score')
    plt.plot(epochs, hist_dict['val_f1'], 'b', label='Validation F1 score')
    plt.title('Training and Validation F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig(results_folder + date +  '_tr_val_f1.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['lr1'], 'g', label='Backbone learning rate')
    plt.plot(epochs, hist_dict['lr2'], 'b', label='Classifier learning rate')
    plt.title('Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.savefig(results_folder + date +  '_learning_rate.jpg', bbox_inches='tight')
    plt.close()
