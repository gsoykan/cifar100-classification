import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

CHECKPOINT_PATH = "../saved_models/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train_cifar100(module_class, **kwargs):
    """
          :param kwargs: additional kwargs => model_name,  train_loader, val_loader, test_loader
          :return: model, result
          """
    model_kwargs = {}
    for key in kwargs.keys():
        if key not in ["model_name", "train_loader", "val_loader", "test_loader"]:
            model_kwargs[key] = kwargs[key]
    train_loader = kwargs['train_loader']
    val_loader = kwargs['val_loader']
    test_loader = kwargs['test_loader']
    model_name = kwargs['model_name']
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "CIFAR100")
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True,
                                                    mode="max",
                                                    monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=100,
                         gradient_clip_val=2,
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = module_class.load_from_checkpoint(pretrained_filename)
    else:
        model = module_class(**model_kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = module_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"],
              "val_acc": val_result[0]["test_acc"],
              "train_acc": train_result[0]["test_acc"]}

    model = model.to(device)
    return model, result
