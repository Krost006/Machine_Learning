from first.imports import *
from first.model import *
from first.dataset_prepare import *

d = Data()
m = Model()


os.system("pip install lightning")
trainer = L.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
)

trainer.fit(m,d)        
