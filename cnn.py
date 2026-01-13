import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        """
        Instantiates the CNN model.
        You should use nn.Sequential to contain multiple useful components to make the forward application simpler.

        HINT: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:

        super(CNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Convolutional layers
            ...
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # Linear layers
            ...
        )
        """
        super(CNN, self).__init__()

        negativeSlope = 0.01

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=negativeSlope, inplace=True),
            # [8 x 84 x 84]
            nn.Conv2d(
                in_channels=8,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=negativeSlope, inplace=True),
            # [32 x 84 x 84]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32 x 42 x 42]
            nn.Dropout(p=0.2),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=negativeSlope, inplace=True),
            # [32 x 42 x 42]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=negativeSlope, inplace=True),
            # [64 x 42 x 42]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [64 x 21 x 21]
            nn.Dropout(p=0.2),
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.LeakyReLU(negative_slope=negativeSlope, inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=negativeSlope, inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 4),
        )
    # end __init__

    def forward(self, x):
        """
        Runs the forward method for the CNN model

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: output classification tensor of the model
        """
        featureMaps = self.feature_extractor(x)
        pooledFeatures = self.avg_pooling(featureMaps)
        outputScores = self.classifier(pooledFeatures)
        return outputScores
    # end forward
