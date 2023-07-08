import torch, math, os, logging
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet50, ResNet50_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights

from fixmatch.dataset import get_cifar10, get_cifar100, get_svhn, get_stl10
from fixmatch.ema import ModelEMA
from fixmatch.lr_scheduler import WarmupCosineLrScheduler


def get_dataloader(args):
    get_dataset = {
        "cifar10": get_cifar10,
        "cifar100": get_cifar100,
        "svhn": get_svhn,
        "stl10": get_stl10
    }
    assert args.dataset in get_dataset.keys(), "Dataset must be in {}".format(
        get_dataset.keys())
    train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset = get_dataset[
        args.dataset](args)

    labeled_sampler = RandomSampler(train_labeled_dataset,
                                    replacement=True,
                                    num_samples=len(train_unlabeled_dataset) //
                                    args.uratio)
    unlabeled_sampler = RandomSampler(train_unlabeled_dataset)
    labeled_batch_sampler = BatchSampler(labeled_sampler,
                                         args.batch_size,
                                         drop_last=True)
    unlabeled_batch_sampler = BatchSampler(unlabeled_sampler,
                                           args.batch_size * args.uratio,
                                           drop_last=True)
    labeled_loader = DataLoader(train_labeled_dataset,
                                batch_sampler=labeled_batch_sampler,
                                num_workers=args.num_workers)
    unlabeled_loader = DataLoader(train_unlabeled_dataset,
                                  batch_sampler=unlabeled_batch_sampler,
                                  num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size * args.uratio,
                              shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size * args.uratio,
                             shuffle=False,
                             num_workers=args.num_workers)

    return labeled_loader, unlabeled_loader, valid_loader, test_loader


def get_model(args):
    out_features = {
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "stl10": 10,
    }
    if args.arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, out_features[args.dataset])
    return model


# def get_cosine_schedule_with_warmup(optimizer,
#                                     num_warmup_steps,
#                                     num_training_steps,
#                                     num_cycles=7. / 16.,
#                                     last_epoch=-1):

#     def _lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         no_progress = float(current_step - num_warmup_steps) / \
#             float(max(1, num_training_steps - num_warmup_steps))
#         return max(0., math.cos(math.pi * num_cycles * no_progress))

#     return LambdaLR(optimizer, _lr_lambda, last_epoch)


class FixMatch:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_label_dataloader, self.train_unlabel_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloader(
            args)
        assert len(self.train_label_dataloader) == len(
            self.train_unlabel_dataloader
        ), f"Number of labeled and unlabeled dataloader must be equal. Got {len(self.train_label_dataloader)} {len(self.train_unlabel_dataloader)}"
        self.model = get_model(args).to(self.device)
        self.optimizer = SGD(self.model.parameters(),
                             lr=args.lr,
                             momentum=args.momentum,
                             weight_decay=args.wd,
                             nesterov=args.nesterov)
        self.scheduler = WarmupCosineLrScheduler(
            self.optimizer,
            args.epochs * len(self.train_label_dataloader),
            args.warmup * len(self.train_label_dataloader),
            warmup_ratio=0.1,
            warmup='linear',
            last_epoch=-1)
        self.ema_model = ModelEMA(self.device, self.model, args.ema_decay)
        self.criterion = CrossEntropyLoss()
        self.best_acc = 0.0
        self.best_model = None
        self.epoch = 0
        self.save_path = args.save
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self):
        logging.basicConfig(filename=f'save/{self.args.dataset}_training.log',
                            level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("Loading checkpoint")
        checkpoints = [
            x for x in os.listdir(self.save_path) if "checkpoint" in x
            and self.args.dataset in x and x.startswith(self.args.arch)
        ]
        logging.info(f"Found {len(checkpoints)} checkpoint {checkpoints}")
        if len(checkpoints) > 0:
            checkpoints = sorted(
                checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint = checkpoints[-1]
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint))
            self.model.load_state_dict(checkpoint["model"])
            self.ema_model.ema.load_state_dict(checkpoint["ema"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_acc = checkpoint["acc"]
            # self.best_model = checkpoint["model"]
            self.epoch = checkpoint["epoch"]
            self.train_loss = checkpoint["train_loss"]
            self.valid_loss = checkpoint["valid_loss"]
        size = len(self.train_label_dataloader)
        logging.info("Start training")
        self.train_loss = []
        self.valid_loss = []
        for epoch in range(self.epoch, self.args.epochs):
            logging.info(f"Epoch {epoch+1}/{self.args.epochs}")
            total_loss = []
            label_loss = []
            pseudo_loss = []
            for idx, ((img, label), (w_img, s_img)) in enumerate(
                    zip(self.train_label_dataloader,
                        self.train_unlabel_dataloader)):
                img, label, w_img, s_img = img.to(self.device), label.to(
                    self.device), w_img.to(self.device), s_img.to(self.device)
                logit = self.model(img)
                loss_l = self.criterion(logit, label)
                w_logit = self.model(w_img)
                w_prob = torch.softmax(w_logit / self.args.T, dim=1)
                pseudo_label = w_prob.ge(self.args.threshold).float()
                s_logit = self.model(s_img)
                loss_u = self.criterion(s_logit, pseudo_label)
                loss = loss_l + self.args.wu * loss_u

                self.optimizer.zero_grad()
                loss.backward()
                total_loss.append(loss.item())
                label_loss.append(loss_l.item())
                pseudo_loss.append(loss_u.item())
                #print loss
                if idx % 30 == 0:
                    logging.info(
                        f"\tBatch: {idx}/{size} - Loss: {loss.item():.6f} - Label Loss: {loss_l.item():.6f} - Pseudo Loss: {loss_u.item():.6f}"
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.ema_model.update(self.model)
            self.train_loss.append(sum(total_loss) / len(total_loss))
            self.valid_loss.append(self.validate())
            if (epoch + 1) % 30 == 0 or epoch == self.args.epochs - 1:
                self.test()
                save = {
                    "model": self.model.state_dict(),
                    "ema": self.ema_model.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "best_model": self.best_model,
                    "train_loss": self.train_loss,
                    "valid_loss": self.valid_loss,
                }
                torch.save(
                    save, "{}/{}_{}_checkpoint_{}.pth".format(
                        self.save_path, self.args.arch, self.args.dataset,
                        epoch))
            if epoch == self.args.epochs - 1:
                self.test()
        logging.info('Training completed.')
        logging.shutdown()

    def validate(self):
        logging.info("Validating")
        self.ema_model.ema.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for img, label in self.valid_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                logit = self.ema_model.ema(img)
                _, pred = torch.max(logit, dim=1)
                loss = self.criterion(logit, label)
                total_loss += loss.item()
                correct += (pred == torch.max(label, dim=1)[1]).sum().item()
                total += len(label)
        acc = correct / total
        total_loss /= len(self.test_dataloader)
        logging.info(f"Accuracy: {acc:.6f} - Loss: {total_loss:.6f}")
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = self.model.state_dict()
        return total_loss, acc

    def test(self):
        logging.info("Testing")
        self.ema_model.ema.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for img, label in self.test_dataloader:
                img, label = img.to(self.device), label.to(self.device)
                logit = self.ema_model.ema(img)
                _, pred = torch.max(logit, dim=1)
                loss = self.criterion(logit, label)
                total_loss += loss.item()
                correct += (pred == torch.max(label, dim=1)[1]).sum().item()
                total += len(label)
        acc = correct / total
        total_loss /= len(self.test_dataloader)
        logging.info(f"Accuracy: {acc:.6f} - Loss: {total_loss:.6f}")

    def save(self, path):
        torch.save(self.best_model, path)