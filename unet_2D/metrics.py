import os
import torchmetrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch import Tensor

class UNetMetrics:
    """
    Class used to track, compute, and manage metrics throughout UNet training
    and validation

    Arguments:
        - num_classes (int): number of classes
    """
    def __init__(self, num_classes: int, device: str):
        self.num_classes = num_classes
        self._train = True
        self.device = device

        # parameters to be passed to torchmetric classes
        params = {
            "num_classes": self.num_classes,
            "mdmc_average": "global",
            "average": "micro",
        }
        if self.num_classes > 1:
            params["ignore_index"] = 0

        # holder for torchmetric instances
        self.metrics: dict[str, dict[str, object]] = {
            stage: {
                "acc": torchmetrics.Accuracy(**params, multiclass=False).to(device=self.device),
                "dice": torchmetrics.Dice(**params, multiclass=False).to(device=self.device),
              # "iou": torchmetrics.JaccardIndex(**params).to(device=DEVICE),
            } for stage in ["train", "val"]
        }

        self.plots: dict[str, dict[str, list]] = {
            stage: {
                "step": [],
                "loss": [],
                "acc": [],
                "dice": [],
            } for stage in ["train", "val"]
        }

    def train(self):
        """
        Puts instance into "train" mode and updates the training metrics and plot
        """
        self._train = True

    def eval(self):
        """
        Puts instance into "eval" mode and updates the validation metrics and plot
        """
        self._train = False

    def update_metrics(
        self,
        preds:
        Tensor,
        targets:
        Tensor
    ) -> tuple[float]:
        """
        Calculates metrics with new batch of data

        Arguments:
            - preds (torch.Tensor): predicted labels with shape (N, H, W)
            - targets (torch.Tensor): targets with shape (N, H, W)

        Returns:
            - tuple of accuracy and dice score floats
        """
        stage = "train" if self._train else "val"
        result = {}
        for metric_name, metric in self.metrics[stage].items():
            result[metric_name] = metric(preds, targets).item()

        return result

    def update_plot(
        self,
        step: int,
        loss: float,
        acc: float,
        dice: float,
    ):
        """
        Update metrics with new batch of data

        Arguments:
            - step (int): batch index corresponding with the provided data
            - loss (float): mean batch cross-entropy loss
            - preds (torch.Tensor): predicted labels with shape (N, H, W)
            - targets (torch.Tensor): targets with shape (N, H, W)
        """
        stage = "train" if self._train else "val"

        self.plots[stage]["step"].append(step)
        self.plots[stage]["loss"].append(loss)
        self.plots[stage]["acc"].append(acc)
        self.plots[stage]["dice"].append(dice)

    def compute(self) -> dict[str, float]:
        """
        Returns dictionary containing final average metrics
        """
        stage = "train" if self._train else "val"
        result = {}
        for metric_name, metric in self.metrics[stage].items():
            # Calculate final average for each metric
            result[metric_name] = metric.compute().item()

            # Resets metric running internal values
            metric.reset()

        return result

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Rearrange metric data into "long-form" pandas dataframe. This includes
        both training and validation data, regardless of class instance state
        https://seaborn.pydata.org/tutorial/data_structure.html
        """
        # Key to better formatted strings for plot
        alt_names = {
            "loss": "Loss",
            "acc": "Accuracy",
            "dice": "Dice Score",
            "train": "Training",
            "val": "Validation",
        }
        df_list = []
        for stage_name, stage in self.plots.items():
            for metric_name, values in stage.items():
                # Exclude batch (step) number from metrics
                if metric_name != "step":
                    stage_col = [alt_names[stage_name]] * len(values)
                    metric_col = [alt_names[metric_name]] * len(values)

                    # Concatenate rows into df_list
                    df_list += list(zip(
                        self.plots[stage_name]["step"],
                        stage_col,
                        metric_col,
                        values
                    ))

        # Generate dataframe
        df = pd.DataFrame(df_list, columns=[
            "Batch",
            "Stage",
            "Metrics",
            "Value",
        ])
        return df

    def save_plots(self, output_dir: str):
        """
        Plots training and validation curves using seaborn and saves the images
        to the "output" path

        Arguments:
            - output_dir (str): path to directory in which plot will be saved
        """

        # Format data into "long-form" dataframe for seaborn
        df = self._to_dataframe()

        # Set seaborn image size and font scaling
        sns.set(
            rc={"figure.figsize": (16, 9)},
            font_scale=1.5,
        )

        # Generate seaborn lineplot
        # Colors different based on metric
        # Style (solid v. dashed) based on training or validation
        # Dashes specified with list of tuples: (segment, gap)
        lplot = sns.lineplot(
            x="Batch",
            y="Value",
            hue="Metrics",
            style="Stage",
            data=df,
            dashes=[(1,0),(6,3)],
        )

        # Remove y-axis label
        lplot.set(ylabel=None)

        # Set plot title
        plt.title("Training Loss and Accuracy Curves")

        # Move legend to outside upper-right corner of image
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

        # Save image to disk
        lplot.figure.savefig(os.path.join(output_dir, "training_curves.png"))

    def write_to_file(self, output_dir: str):
        """
        Writes data to csv in "long form"

        Arguments:
            - output_dir (str): path to .csv output directory
        """
        df = self._to_dataframe()
 
