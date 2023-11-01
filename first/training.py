from first.imports import *
from first.model import *
from first.dataset_prepare import *

d = Data()
m = Model()
wandb_logger = WandbLogger(project='first', job_type='train')

#os.system("pip install lightning")
trainer = L.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
    logger=wandb_logger
)

trainer.fit(m,d)        
