from first.imports import *
class Data(L.LightningDataModule):

  def __init__(self, data_dir = 'new_data'):
    super().__init__()
    self.data_dir = data_dir
    #self.hidden_size = hidden_size
    self.learning_rate = 0.02
    self.transform = transforms.Compose(
            [
               transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    ####################
    # DATA RELATED HOOKS
    ####################

  def prepare_data(self):
        # download
    CIFAR10(self.data_dir, train=True, download=True)
    CIFAR10(self.data_dir, train=False, download=True)

  def setup(self, stage=None):
     # Assign train/val datasets for use in dataloaders
     if stage == "fit" or stage is None:
        mnist_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [0.7, 0.3])

        # Assign test dataset for use in dataloader(s)
     if stage == "test" or stage is None:
        self.mnist_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

  def train_dataloader(self):
      return torch.utils.data.DataLoader(self.mnist_train, batch_size=4)

  def val_dataloader(self):
      return torch.utils.data.DataLoader(self.mnist_val, batch_size=4)

  def test_dataloader(self):
      return torch.utils.data.DataLoader(self.mnist_test, batch_size=4)
