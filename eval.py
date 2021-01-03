from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from torchattacks import VANILA, FGSM, PGD, MultiAttack

from defenses.loader import base_loader
from defenses.model import get_model

def run(gpu, model_path, model, data, num_classes, data_path,
        method, eps, alpha, steps, restart):

    torch.cuda.set_device(gpu)

    # Set Model
    model = get_model(name=model, num_classes=num_classes).cuda()

    model.load_state_dict(torch.load(model_path))
    model = model.cuda().eval()

    # Get Test Loader
    _, test_loader = base_loader(data_name=data,
                                 shuffle_train=False)

    if method == "Standard":
        atk = VANILA(model)
    elif method == "FGSM":
        atk = FGSM(model, eps=eps)
    elif method == "PGD":
        pgd = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
        atk = MultiAttack([pgd]*restart)
    else:
        raise ValueError("Not valid method.")

    atk.set_return_type('int')
    atk.save(data_loader=test_loader,
             save_path=data_path, verbose=True)

    print("Test Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', default=0, type=int, help='GPU number to be used')
    parser.add_argument('--model-path', type=str, help='Path of saved model')
    parser.add_argument('--model', type=str, default='PRN18', choices=['PRN18', 'WRN28'], help='Model Structure')
    parser.add_argument('--data', type=str, default='CIFAR10', help='Data name')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--data-path', type=str, default=None, help='Path for saving adversarial images')
    parser.add_argument('--method', type=str, default='Standard', choices=['Standard', 'FGSM', 'PGD'], help='Training method')
    parser.add_argument('--eps', default=8, type=float, help='Maximum perturbation (ex.8)')
    parser.add_argument('--alpha', default=2, type=float, help='Stepsize (ex.12)')
    parser.add_argument('--steps', default=1, type=int, help='Number of steps of PGD')
    parser.add_argument('--restart', default=1, type=int, help='Number of restart of PGD')
    args = parser.parse_args()

    run(args.gpu,
        args.model_path,
        args.model,
        args.data,
        args.num_classes,
        args.data_path,
        args.method,
        args.eps/255,
        args.alpha/255,
        args.steps,
        args.restart)