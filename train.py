import torch

from settings import args

from defenses.loader import get_loader
from defenses.model import get_model
from defenses.trainer import get_trainer


if __name__ == "__main__":
    run(args)

def run(args):

    torch.cuda.set_device(args.gpu)

    # Set data_loader and Trainer
    data_loader = get_loader(args.loader)
    Trainer = get_trainer(args.trainer)

    # Set Train, Test loader
    train_loader, test_loader = data_loader(data_name=args.data,
                                            shuffle_train=True)
    train_loader_ns, _ = data_loader(data_name=args.data,
                                     shuffle_train=False) # w/o Suffle

    # Get First batch
    train_set = iter(train_loader_ns).next()
    test_set = iter(test_loader).next()

    # Set Model
    model = get_model(name=args.model, num_classes=args.num_classes).cuda()

    # Set Trainer
    trainer = Trainer(model, **args.trainer_args)
    trainer.record_rob(train_set, test_set,
                       eps=args.test_eps, alpha=args.test_alpha,
                       steps=args.test_steps)

    # Train Model
    trainer.train(train_loader=train_loader, max_epoch=args.max_epoch,
                  optimizer=args.optimizer, scheduler=args.scheduler,
                  scheduler_type=args.scheduler_type,
                  save_type="Epoch", save_path=args.save_path+args.name,
                  save_overwrite=False, record_type="Epoch")

    trainer.save_all(args.save_path+args.name)