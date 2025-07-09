import os
import pandas as pd
import datetime 
from models import TransferSegmentation, save_model, load_model
import torch 
import torch.nn as nn 
from torchvision.transforms import v2 
from utils_jeff import Load_Dataset
import torch.utils.tensorboard as tb 
import numpy as np
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Ruta principal donde están las carpetas como 'abyss', 'gran_paradiso_island', etc.
base_path = 'dense_data'

# Inicializamos una lista para guardar los datos
data = []

# Recorremos cada subcarpeta (nivel de mapa)
for level_name in os.listdir(base_path):
    level_path = os.path.join(base_path, level_name)

    # Nos aseguramos de que sea una carpeta
    if not os.path.isdir(level_path):
        continue

    # Rutas de interés: frame y combined_visual
    frame_path = os.path.join(level_path, 'frame')
    combined_visual_path = os.path.join(level_path, 'combined')

    # Leemos los nombres de archivos de cada carpeta (si existen)
    frame_images = os.listdir(frame_path) if os.path.isdir(frame_path) else []
    combined_visual_images = os.listdir(combined_visual_path) if os.path.isdir(combined_visual_path) else []

    # Ajustamos la cantidad para que estén pareados (opcional: puedes usar el largo máximo)
    max_len = max(len(frame_images), len(combined_visual_images))

    # Completamos con None si las listas no son del mismo tamaño
    frame_images += [None] * (max_len - len(frame_images))
    combined_visual_images += [None] * (max_len - len(combined_visual_images))

    # Guardamos en la lista de datos
    for f_img, cv_img in zip(frame_images, combined_visual_images):
        data.append({
            'name': level_name,
            'frame': f_img,
            'combined': cv_img
        })

# Creamos el DataFrame
df = pd.DataFrame(data)


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
datase = Load_Dataset(df, base_path, batch_size=64, aumentation=5, num_workers=4)
=======
datase = Load_Dataset(df, base_path, batch_size=256, aumentation=5, num_workers=16)
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
datase = Load_Dataset(df, base_path, batch_size=256, aumentation=5, num_workers=16)
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
datase = Load_Dataset(df, base_path, batch_size=256, aumentation=5, num_workers=16)
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
datase = Load_Dataset(df, base_path, batch_size=256, aumentation=5, num_workers=16)
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
random_state=42
torch.manual_seed(random_state)


def post_proces(predicted_logits):
    predicted_mask = torch.argmax(predicted_logits, dim=1)  # [B, H, W]
    return predicted_mask

def calculate_multiclass_iou_f1(predicted_logits, target_mask, num_classes, epsilon=1e-6):
    """
    Calcula el IoU y el F1-score por clase y sus promedios (mIoU, macro-F1).

    Args:
        predicted_logits: Tensor [B, C, H, W] (output sin softmax)
        target_mask: Tensor [B, H, W] con valores enteros de clase
        num_classes: Total de clases
    Returns:
        mean_iou: Promedio de IoU entre clases
        mean_f1: Promedio de F1 entre clases
    """
    predicted_mask= post_proces(predicted_logits)

    ious = []
    f1s = []

    for cls in range(num_classes):
        pred_cls = (predicted_mask == cls)
        target_cls = (target_mask == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        # IoU
        iou = (intersection + epsilon) / (union + epsilon)
        ious.append(iou)

        # F1-score
        tp = intersection
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        f1 = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        f1s.append(f1)

    mean_iou = torch.mean(torch.tensor(ious))
    mean_f1 = torch.mean(torch.tensor(f1s))

    return mean_iou.item(), mean_f1.item()

model = TransferSegmentation(n_classes=7)

transform = model.weights.transforms() # resulta mejor con la transformacion propia


train_loader = datase.load_train(transform=transform)
val_loader = datase.load_val(transform=transform)
test_loader = datase.load_test(transform=transform)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

criterion = nn.CrossEntropyLoss() # aplica sigmoid directamente sobre los logits del modelo
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)
=======
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
model.to(device)

criterion = nn.CrossEntropyLoss() # aplica sigmoid directamente sobre los logits del modelo
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        log_dir=None,
        num_classes=7,
        max_epochs=200,
        max_patience=25,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.max_patience = max_patience

        self.best_iou = 0
        self.patience = 0
        self.global_step = 0
        self.writer = SummaryWriter(log_dir, flush_secs=1) if log_dir else None
        self.val_stoping=0
        self.train_stoping=0

    def train(self):
        print(f"{'Epoch':<7}{'Phase':<8}{'Loss':<10}{'mIoU (%)':<10}{'F1 (%)':<10}")
        print("=" * 45)
        for epoch in range(self.max_epochs):
            epoch_loss, epoch_iou, epoch_f1 = self.train_one_epoch(epoch)
    
            self.scheduler.step(epoch_iou)
    
            # Evaluación después de entrenamiento
            val_iou, val_f1 = self.evaluate(epoch)
    
            # Mostrar resumen en columna
            print(f"{epoch+1:<7}{'Train':<8}{epoch_loss:<10.4f}{epoch_iou:<10.2f}{epoch_f1:<10.2f}")
            print(f"{'':<7}{'Val':<8}{'-':<10}{val_iou:<10.2f}{val_f1:<10.2f}")
    
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("epoch_loss", epoch_loss, epoch)
                self.writer.add_scalar("epoch_mean_iou", epoch_iou, epoch)
                self.writer.add_scalar("epoch_mean_f1", epoch_f1, epoch)
                self.writer.add_scalar("val_mean_iou", val_iou, epoch)
                self.writer.add_scalar("val_mean_f1", val_f1, epoch)
                self.writer.add_scalar("epoch_lr", self.optimizer.param_groups[0]["lr"], epoch)
    
            # Guardado del mejor modelo
            
            if val_iou > self.best_iou:
                self.save_model("TransferSegmentation_Jeff.th")
                self.best_iou = val_iou
                self.patience = 0

            if val_iou < self.val_stoping:
               if epoch_iou > self.train_stoping:
                   self.train_stoping=epoch_iou
                   self.patience=0
               else:
                   self.patience+=1 
            else: 
                self.val_stoping=val_iou
                self.patience=0
    
            if self.patience >= self.max_patience:
                print("Early stopping.")
                break


    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0
        ious, f1 = [], []
        total = len(self.train_loader)

        for i, (inputs, targets) in enumerate(self.train_loader):
            print(f"Train Epoch {epoch+1}  {(i+1)*100/total:.2f}%", end="\r")
            inputs = inputs.to(self.device,non_blocking=True)
            targets = targets.long().squeeze().to(self.device,non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_iou, f1_score = calculate_multiclass_iou_f1(
                outputs, targets, self.num_classes
            )
            ious.append(batch_iou)
            f1.append(f1_score)
            running_loss += loss.item()

        mean_iou = np.mean(ious) * 100
        mean_f1 = np.mean(f1) * 100
        avg_loss = running_loss / total
        return avg_loss, mean_iou, mean_f1


    def evaluate(self, epoch):
        self.model.eval()
        ious, f1 = [], []
        total = len(self.val_loader)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                print(f"Val   Epoch {epoch+1}  {(i+1)*100/total:.2f}%", end="\r")
                inputs = inputs.to(self.device,non_blocking=True)
                targets = targets.long().squeeze().to(self.device,non_blocking=True)

                outputs = self.model(inputs)
                batch_iou, f1_score = calculate_multiclass_iou_f1(
                    outputs, targets, self.num_classes
                )
                ious.append(batch_iou)
                f1.append(f1_score)
            print()

        mean_iou = np.mean(ious) * 100
        mean_f1 = np.mean(f1) * 100

        if self.writer:
            self.writer.add_scalar("val_mean_iou", mean_iou, epoch)
            self.writer.add_scalar("val_mean_f1", mean_f1, epoch)
            vis_targets = targets.unsqueeze(1) * 32 / 255
            pred_mask = post_proces(outputs)
            vis_outputs = pred_mask.unsqueeze(1) * 32 / 255
            self.writer.add_images("val/image", inputs[:6].cpu(), epoch)
            self.writer.add_images("val/label", vis_targets[:6].cpu(), epoch)
            self.writer.add_images("val/pred", vis_outputs[:6].cpu(), epoch)

            if self.global_step == 0:
                self.writer.add_graph(self.model, inputs)
        self.global_step += 1
        return mean_iou, mean_f1


    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Modelo guardado como {filename}")

train_logger = None 
log_dir = f'semantic_segmentation/runs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}' 

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD



=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    log_dir=log_dir,
    num_classes=7,
    max_epochs=100,
    max_patience= 100
)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
trainer.train()
=======


trainer.train()
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======


trainer.train()
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
=======


trainer.train()
>>>>>>> f6cb91ebbe6bb75a52033d1f4d3210be2bf53889
