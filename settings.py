from utils import Arguments

args = Arguments()

args.gpu = 0
args.name = "FastAdv_CIFAR10_PRN18_Cyclic"
args.save_path = "./_checkpoint/"
args.model = "PRN18"
args.loader = "Base"
args.trainer = "Base"

args.data = "CIFAR10"
args.num_classes = 10

args.init_lr = 0.1
args.optimizer = "SGD(lr="+str(args.init_lr)+", momentum=0.9, weight_decay=5e-4)"

# Cosine
# args.max_epoch = 100
# args.scheduler = "Cosine"
# args.scheduler_type = "Epoch"

# Cyclic
args.max_epoch = 30
args.scheduler = "Cyclic(0, 0.3)"
args.scheduler_type = "Iter"

# Stepwise
# args.max_epoch = 100
# args.scheduler = "Step([50, 75], 0.1)"
# args.scheduler_type = "Epoch"

args.trainer_args = {
    "eps":8/255,
    "alpha":10/255,
}

args.test_eps = args.trainer["eps"] #8/255
args.test_alpha = 2/255
args.test_steps = 10

