import torch, math, os, logging
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision.models import resnet50, ResNet50_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights, wide_resnet101_2, Wide_ResNet101_2_Weights

from fixmatch.dataset import get_cifar10, get_cifar100, get_svhn, get_stl10
from fixmatch.ema import ModelEMA
from fixmatch.lr_scheduler import WarmupCosineLrScheduler
from model.resnet import ResNet50
from model.widen_resnet import WideResNet


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
    models = {
        "pretrained": {
            "resnet50": [resnet50, {
                "weights": ResNet50_Weights.DEFAULT
            }],
            "wide_resnet50_2":
            [wide_resnet50_2, {
                "weights": Wide_ResNet50_2_Weights.DEFAULT
            }],
            "wide_resnet101_2":
            [wide_resnet101_2, {
                "weights": Wide_ResNet101_2_Weights.DEFAULT
            }],
        },
        "defined": {
            "resnet50":
            [ResNet50, {
                "num_classes": out_features[args.dataset]
            }],
            "wide_resnet28_2": [
                WideResNet, {
                    "depth": 28,
                    "widen_factor": 2,
                    "num_classes": out_features[args.dataset]
                }
            ],
            "wide_resnet28_4": [
                WideResNet, {
                    "depth": 28,
                    "widen_factor": 4,
                    "num_classes": out_features[args.dataset]
                }
            ],
            "wide_resnet34_2": [
                WideResNet, {
                    "depth": 34,
                    "widen_factor": 2,
                    "num_classes": out_features[args.dataset]
                }
            ],
        }
    }
    load = models[args.pretrained]
    model_name, model_params = load[args.arch]
    model = model_name(**model_params)
    if args.pretrained == "pretrained":
        in_feat = model.fc.in_features
        out_feat = out_features[args.dataset]
        model.fc = torch.nn.Linear(in_feat, out_feat)
    return model


class FixMatch:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.train_labeled_dataloader, self.train_unlabeled_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloader(
            args)
        # assert len(self.train_label_dataloader) == len(
        #     self.train_unlabel_dataloader
        # ), f"Number of labeled and unlabeled dataloader must be equal. Got {len(self.train_label_dataloader)} {len(self.train_unlabel_dataloader)}"
        self.model = get_model(args).to(self.device)
        self.optimizer = SGD(self.model.parameters(),
                             lr=args.lr,
                             momentum=args.momentum,
                             weight_decay=args.wd,
                             nesterov=args.nesterov)
        self.scheduler = WarmupCosineLrScheduler(self.optimizer,
                                                 args.total_steps,
                                                 args.warmup * args.eval_steps,
                                                 warmup_ratio=0.1,
                                                 warmup='linear',
                                                 last_epoch=-1)
        self.ema_model = ModelEMA(self.device, self.model, args.ema_decay)
        self.criterion = CrossEntropyLoss()
        self.best_acc = 0.0
        self.best_model = None
        self.epoch = 0
        self.save_path = args.save
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

    def train(self):
        logging.info("Loading checkpoint")
        checkpoints = [
            x for x in os.listdir(self.save_path)
            if f"{self.args.dataset}_{self.args.num_labels}_checkpoint" in x
        ]
        logging.info(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        if len(checkpoints) > 0:
            checkpoints = sorted(
                checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint = checkpoints[-1]
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint))
            self.model.load_state_dict(checkpoint["model"])
            self.ema_model.ema.load_state_dict(checkpoint["ema"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_acc = checkpoint["best_acc"]
            self.best_model = checkpoint["model"]
            self.epoch = checkpoint["epoch"]
            self.train_loss = checkpoint["train_loss"]
            self.valid_loss = checkpoint["valid_loss"]
        logging.info("Start training" if self.epoch ==
                     0 else "Continue training")
        print("Start training" if self.epoch ==
              0 else "Continue training") if self.args.debug else None
        self.train_loss = []
        self.valid_loss = []

        labeled_iter = iter(self.train_labeled_dataloader)
        unlabeled_iter = iter(self.train_unlabeled_dataloader)
        for epoch in range(self.epoch, self.args.epochs):
            logging.info(f"Epoch {epoch+1}/{self.args.epochs}")
            logging.info(
                f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            print(
                f"Epochs: {epoch+1}/{self.args.epochs}\nLearning rate: {self.optimizer.param_groups[0]['lr']}"
            ) if self.args.debug else None
            total_loss = []
            label_loss = []
            unlabel_loss = []
            for batch_idx in range(self.args.eval_steps):
                try:
                    l_img, label = next(labeled_iter)
                except:
                    labeled_iter = iter(self.train_labeled_dataloader)
                    l_img, label = next(labeled_iter)

                try:
                    u_w_img, u_s_img = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(self.train_unlabeled_dataloader)
                    u_w_img, u_s_img = next(unlabeled_iter)
            # for idx, ((l_img, label), (u_w_img, u_s_img)) in enumerate(
            #         zip(self.train_label_dataloader,
            #             self.train_unlabel_dataloader)):
                l_img, label, u_w_img, u_s_img = l_img.to(
                    self.device), label.to(self.device), u_w_img.to(
                        self.device), u_s_img.to(self.device)
                l_logit = self.model(l_img)
                l_loss = F.cross_entropy(l_logit, label, reduction="mean")
                u_w_logit = self.model(u_w_img)
                pseudo_label = torch.softmax(u_w_logit / self.args.T, dim=-1)
                max_probs, pred_u_w = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.args.threshold).float()
                u_s_logit = self.model(u_s_img)
                u_loss = F.cross_entropy(u_s_logit, pred_u_w,
                                         reduction="none").mul(mask).mean()
                loss = l_loss + self.args.wu * u_loss

                self.optimizer.zero_grad()
                loss.backward()
                total_loss.append(loss.item())
                label_loss.append(l_loss.item())
                unlabel_loss.append(u_loss.item())
                #print loss
                if batch_idx % int(self.args.eval_steps / 5) == 0:
                    logging.info(
                        f"\tBatch: {batch_idx}/{self.args.eval_steps} - Loss: {loss.item():.6f} - Label Loss: {l_loss.item():.6f} - Pseudo Loss: {u_loss.item():.6f}"
                    )
                    print(
                        f"\tBatch: {batch_idx}/{self.args.eval_steps} - Loss: {loss.item():.6f} - Label Loss: {l_loss.item():.6f} - Pseudo Loss: {u_loss.item():.6f}"
                    ) if self.args.debug else None
                self.optimizer.step()
                self.scheduler.step()
                self.ema_model.update(self.model)
            logging.info(f"Train loss: {sum(total_loss) / len(total_loss)}")
            print(f"Train loss: {sum(total_loss) / len(total_loss)}"
                  ) if self.args.debug else None
            self.train_loss.append(sum(total_loss) / len(total_loss))
            self.valid_loss.append(self.validate())
            if (epoch + 1) % 50 == 0:
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
                    save,
                    "{}/{}_{}_checkpoint_{}".format(self.save_path,
                                                    self.args.dataset,
                                                    self.args.num_labels,
                                                    epoch + 1))
            if epoch == self.args.epochs - 1:
                self.test()
        logging.info('Training completed.')
        print("Training completed") if self.args.debug else None

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
        print(f"Validating: Accuracy: {acc:.6f} - Loss: {total_loss:.6f}"
              ) if self.args.debug else None
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
        print(f"Testing: Accuracy: {acc:.6f} - Loss: {total_loss:.6f}"
              ) if self.args.debug else None
