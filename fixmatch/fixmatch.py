import torch, math, os, logging
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet50, ResNet50_Weights

from fixmatch.dataset import get_cifar10, get_cifar100, get_svhn, get_stl10
from model.resnet import ResNet50
from model.widen_resnet import WideResNet
from model.ema import ModelEMA

def get_dataloader(args):
    get_dataset = {
        "cifar10": get_cifar10,
        "cifar100": get_cifar100,
        "svhn": get_svhn,
        "stl10": get_stl10
    }
    assert args.dataset in get_dataset.keys(), "Dataset must be in {}".format(get_dataset.keys())
    if args.num_labels:
        train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset = get_dataset[args.dataset](args.num_labels)
    elif args.fold:
        train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset = get_dataset[args.dataset](args.fold)
        
    labeled_sampler = RandomSampler(train_labeled_dataset, replacement=True, num_samples=len(train_unlabeled_dataset)//args.uratio)
    unlabeled_sampler = RandomSampler(train_unlabeled_dataset)
    labeled_batch_sampler = BatchSampler(labeled_sampler, args.batch_size, drop_last=True)
    unlabeled_batch_sampler = BatchSampler(unlabeled_sampler, args.batch_size*args.uratio, drop_last=True)
    labeled_loader = DataLoader(train_labeled_dataset, batch_sampler=labeled_batch_sampler, num_workers=args.num_workers)
    unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_sampler=unlabeled_batch_sampler, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size*args.uratio, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*args.uratio, shuffle=False, num_workers=args.num_workers)
    
    return labeled_loader, unlabeled_loader, valid_loader, test_loader

def get_model(args):
    out_features = {
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "stl10": 10,
    }
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, out_features)
    return model

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
    
class FixMatch:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_label_dataloader, self.train_unlabel_dataloader, self.valid_dataloader, self.test_dataloader = get_dataloader(args)
        assert len(self.train_label_dataloader) == len(self.train_unlabel_dataloader), f"Number of labeled and unlabeled dataloader must be equal. Got {len(self.train_label_dataloader)} {len(self.train_unlabel_dataloader)}"
        self.model = get_model(args).to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, args.warmup, args.epochs)
        self.ema_model = ModelEMA(self.device, self.model, args.ema_decay)
        self.criterion = CrossEntropyLoss()
        self.best_acc = 0.0
        self.best_model = None
        self.epoch = 0
        self.save_path = "save/checkpoint_{}.pth"
        if not os.path.isdir("save"):
            os.mkdir("save")
        
    def train(self):
        logging.basicConfig(filename='save/training.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
        logging.info("Start training")
        self.train_loss = []
        self.valid_loss = []
        checkpoints = [x for x in os.listdir("save") if "checkpoint" in x]
        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint = checkpoints[-1]
            checkpoint = torch.load(os.path.join("save", checkpoint))
            self.model.load_state_dict(checkpoint["model"])
            self.ema_model.ema.load_state_dict(checkpoint["ema"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.best_acc = checkpoint["acc"]
            self.best_model = checkpoint["model"]
            self.epoch = checkpoint["epoch"]
            self.train_loss = checkpoint["train_loss"]
            self.valid_loss = checkpoint["valid_loss"]
        size = len(self.train_label_dataloader)
        for epoch in range(self.epoch, self.args.epochs):
            logging.info(f"Epoch {epoch+1}/{self.args.epochs}")
            total_loss = []
            label_loss = []
            pseudo_loss = []
            for idx, ((img, label), (w_img, s_img)) in enumerate(zip(self.train_label_dataloader, self.train_unlabel_dataloader)):
                img, label, w_img, s_img = img.to(self.device), label.to(self.device), w_img.to(self.device), s_img.to(self.device)
                logit = self.model(img)
                loss_l = self.criterion(logit, label)
                w_logit = self.model(w_img)
                w_prob = torch.softmax(w_logit/self.args.T, dim=1)
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
                    logging.info(f"\tBatch: {idx}/{size} - Loss: {loss.item():.6f} - Label Loss: {loss_l.item():.6f} - Pseudo Loss: {loss_u.item():.6f}")
                self.optimizer.step()
                self.ema_model.update(self.model)
                self.scheduler.step()
            if (epoch+1) % 1 == 0 or epoch == self.args.epochs-1:
                self.train_loss.append(sum(total_loss)/len(total_loss))
                self.valid_loss.append(self.validate())
            if (epoch+1) % 50 == 0 or epoch == self.args.epochs-1:
                save = {
                    "model": self.model.state_dict(),
                    "ema": self.ema_model.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": epoch,
                    "best_acc": self.best_acc,
                    "best_model": self.best_model.state_dict(),
                    "train_loss": self.train_loss,
                    "valid_loss": self.valid_loss,
                }
                torch.save(save, self.save_path.format(epoch))
            if epoch == self.args.epochs-1:
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