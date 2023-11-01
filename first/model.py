from first.imports import *
class Model(L.LightningModule):
  def __init__(self):
    super().__init__()
    self.learning_rate = 0.02

    self.save_hyperparameters()
  
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    #self.val_accuracy = accuracy()
    #self.val_accuracy =accuracy(torch.tensor([1,1,1,1,1,1,1,1,1,1]),torch.tensor([0,0,0,0,0,0,0,0,0,0]),'multiclass',num_classes = 10)
    self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> torch.Tensor:
        """Defines a single training step for the model.

        Args:
            batch: A tuple containing the input and target tensors for the batch.
            batch_nb: The batch number.

        Returns:
            torch.Tensor: The loss value for the current batch.

        Examples:
            >>> model = MNISTModel()
            >>> x = torch.randn(1, 1, 28, 28)
            >>> y = torch.tensor([1])
            >>> loss = model.training_step((x, y), 0)
        """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', self.val_accuracy, on_step=True, on_epoch=True, logger=True)
        return loss
  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> None:
        """Defines a single validation step for the MLP.

        Args:
            batch : A tuple containing the input data and target labels.
            batch_idx : The index of the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
  def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer to use during training.

        Returns:
            torch.optim.Optimizer: The optimizer to use during training.

        Examples:
            >>> model = MNISTModel()
            >>> optimizer = model.configure_optimizers()
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

