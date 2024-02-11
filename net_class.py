import torch.nn as nn
import torchvision.models as models


class PretrainedModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(PretrainedModel, self).__init__() # used to call the constructor of the parent class, which is nn.Module

        # Check if the specified model is available
        if model_name not in models.__dict__:
            raise ValueError(f"Model '{model_name}' not found in torchvision.models")

        # Load the model
        self.model = models.__dict__[model_name](pretrained=pretrained)

        # Modify the classifier to match the number of classes
        # Find the last fully connected layer (classifier) and modify it
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                self.model.__setattr__(name, nn.Linear(in_features, num_classes))
                break

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    shufflenetv2_model = PretrainedModel("vgg16", num_classes=10, pretrained=True)
    print(shufflenetv2_model)