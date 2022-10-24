import os
import shutil
import pandas as pd
from transformers import Wav2Vec2Processor, Data2VecAudioModel, get_cosine_schedule_with_warmup, \
    feature_extraction_utils, Wav2Vec2FeatureExtractor
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import soundfile
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import GradScaler, autocast

device = 'cuda' if torch.cuda.is_available() else 'cpu'

c2i = {c: i for i, c in enumerate(['China', 'South Africa', 'United States', 'Venezuela'])}
i2c = {i: c for c, i in c2i.items()}

t2i = {t: i for i, t in enumerate(['Cry', 'Gasp', 'Groan', 'Grunt', 'Laugh', 'Other', 'Pant', 'Scream'])}
i2t = {i: t for t, i in t2i.items()}

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-base-960h")

class CCC(nn.Module):
    #TODO: use bessel correction term?
    #cf. (yay) https://audtorch.readthedocs.io/en/latest/api-metrics-functional.html
    # and (nay) https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    def __init__(self):
        super(CCC, self).__init__()

    def forward(self, pred, target):
        mu_pred = pred.mean(dim=-1, keepdim=True)
        mu_target = target.mean(dim=-1, keepdim=True)
        var_pred = pred.var(dim=-1, keepdim=True)
        var_target = target.var(dim=-1, keepdim=True)
        cov = ((pred-mu_pred)*(target-mu_target)).mean(dim=-1, keepdim=True)
        ccc = (2 * cov) / (var_pred + var_target + (mu_pred - mu_target) ** 2)
        return ccc.squeeze()


class MTLoss(nn.Module):
    # cf. https://arxiv.org/pdf/1705.07115.pdf
    # the loss module learns the weigths in the linear combination of each task's loss
    # store the weights as log(std_i) -> 1/std_i**2 == exp(-2*weight_i)
    def __init__(self, label_smoothing=0.0):
        super(MTLoss, self).__init__()
        self.requires_grad_(True)
        self.log_std = nn.Parameter(torch.zeros(4))
        self.ccc = CCC()
        self.label_smoothing = label_smoothing
        # weights have to be set, else load_state_dict throws error
        self.ce_country = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(tuple([1/4]*4), device=device))
        self.ce_type = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(tuple([1/8]*8), device=device))

    def forward(self, predict: dict, target: dict):
        loss_c = self.ce_country(predict['country'], target['country'])
        loss_t = self.ce_type(predict['type'], target['type'])
        loss_hi = 1.0 - self.ccc(predict['high'], target['high'])
        loss_cu = 1.0 - self.ccc(predict['culture'], target['culture'])
        loss = torch.stack((loss_c, loss_t, loss_hi, loss_cu))
        loss = loss.transpose(0, 1)
        weight = torch.exp(-2.0*self.log_std)
        ret = torch.matmul(loss, weight)
        ret += self.log_std.sum()
        ret = ret.mean()
        return {'loss': ret, 'task_loss': loss.mean(dim=0), 'task_weight': weight,
                'task_uncertainty': torch.exp(self.log_std),
                'ccc_hi': (1-loss_hi).mean(), 'ccc_cu': (1-loss_cu).mean()}

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        # only during train: inversely proportional weights to counter class imbalance combined with label smoothing
        if mode:
            self.ce_country.weight = torch.tensor((0.21442458972180697, 0.21568838887537362, 0.15371799207163983,
                                                   0.41616902933117955), device=device)
            self.ce_country.label_smoothing = self.label_smoothing
            self.ce_type.weight = torch.tensor((0.09220052044246965, 0.023945658814239372, 0.12453144964594179,
                                                0.12628801797799294, 0.03443521461869565, 0.12619433250471548,
                                                0.36426115678020665, 0.1081436492157384), device=device)
            self.ce_type.label_smoothing = self.label_smoothing
        else:
            self.ce_country.weight = torch.tensor(tuple([1/4]*4), device=device)
            self.ce_country.label_smoothing = 0.0
            self.ce_type.weight = torch.tensor(tuple([1/8]*8), device=device)
            self.ce_type.label_smoothing = 0.0

        return self


class AvbDataset(Dataset):
    def __init__(self, split, scaling=False):
        super(AvbDataset, self).__init__()
        self.df = pd.read_csv('dataset.csv', dtype={'File_ID': str})
        if scaling:
            # train and validation have both the same min-max values which are not always 0-1 (weird)
            # TODO: for test prediction inverse-transform
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.df.loc[self.df.Split == 'Train', self.df.columns[3:-1]])
            self.df = self.df[self.df.Split == split]
            self.df.loc[:, self.df.columns[3:-1]] = self.scaler.transform(self.df.loc[:, self.df.columns[3:-1]])
        else:
            self.df = self.df[self.df.Split == split]
        self.processor = processor

    def __getitem__(self, idx):
        dfi = self.df.iloc[idx]
        data, sampling_rate = soundfile.read(f'audio/wav/{dfi["File_ID"]}.wav')
        # run find_longest_sr.py for padding value, multiple of 128 for cuda cores
        sample = self.processor(data, sampling_rate=sampling_rate, return_tensors="pt",
                                padding='max_length', max_length=159360)
        sample.data['input_values'] = sample.data['input_values'].squeeze()
        sample.data['attention_mask'] = sample.data['attention_mask'].squeeze()
        c = nn.functional.one_hot(torch.tensor(c2i[dfi[2]]), num_classes=len(c2i))
        t = nn.functional.one_hot(torch.tensor(t2i[dfi[55]]), num_classes=len(t2i))
        hi = torch.Tensor(dfi[3:13])
        cu = torch.Tensor(dfi[13:53])
        ret = {'country': c.float(), 'type': t.float(), 'high': hi, 'culture': cu}
        return sample, ret

    def __len__(self):
        return len(self.df)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.requires_grad_(True)

        self.base_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")

        self.country = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )

        self.typ = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 8),
            nn.Softmax(dim=1)
        )

        self.high = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 10)
        )

        self.culture = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 40)
        )

    def forward(self, input_values, attention_mask):
        last_hidden_state = self.base_model(input_values, attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state.mean(dim=1)

        ret = {'country': self.country(last_hidden_state), 'type': self.typ(last_hidden_state),
               'high': torch.clamp(self.high(last_hidden_state), min=0.0, max=1.0),
               'culture': torch.clamp(self.culture(last_hidden_state), min=0.0, max=1.0)}

        return ret

    def freeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.base_model.parameters():
            param.requires_grad = True


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class ConfusionMatrixMeter:
    def __init__(self):
        self.conf_mat = None

    def reset(self):
        if not self.conf_mat is None:
            self.conf_mat = torch.zeros(self.conf_mat.shape)

    def update(self, pred, target):
        if self.conf_mat is None:
            self.conf_mat = torch.zeros((pred.shape[-1], pred.shape[-1]))
        for i, j in zip(pred.argmax(dim=-1), target.argmax(dim=-1)):
            self.conf_mat[i, j] += 1

    def uar(self):
        if self.conf_mat is None:
            return None
        ret = torch.zeros(self.conf_mat.shape[-1])
        for j in range(self.conf_mat.shape[-1]):
            col_sum = self.conf_mat[:, j].sum()
            if col_sum == 0:
                ret[j] = 0
            else:
                ret[j] = self.conf_mat[j, j]/col_sum
        return ret.mean().item()


def move_to(obj, device=device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, feature_extraction_utils.BatchFeature):
        return obj.to(device)
    else:
        raise TypeError(f"Invalid type for move_to: {type(obj)}")


def main():
    lr = 4e-5
    batch_size = 32
    epochs = 30
    patience = 2
    epochs_warmup = 1
    num_workers = 0 if os.name == 'nt' else 2
    torch.backends.cudnn_benchmark_enabled = True

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model = Model()
    model = model.to(device)

    #print('Model summary:')
    #print(model)


    train_loader = DataLoader(dataset=AvbDataset(split='Train'),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True
                              )

    val_loader = DataLoader(dataset=AvbDataset(split='Val'),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True
                            )

    loss_module = MTLoss(label_smoothing=0.1)
    loss_module.to(device)
    optimiser = torch.optim.AdamW(list(model.parameters()) + list(loss_module.parameters()))#, lr=lr)
    scaler = GradScaler()

    loss_eval_best = float('inf')
    out_best = None
    score_best = None
    patience_cur = patience
    loss_train = AverageMeter()
    loss_eval = AverageMeter()
    ccc_hi, ccc_cu = AverageMeter(), AverageMeter()
    uar_type = ConfusionMatrixMeter()
    uar_country = ConfusionMatrixMeter()
    model_save_path = model_path+'/'+os.path.basename(__file__).replace('py', 'pth')
    model_loss_save_path = model_path+'/'+os.path.basename(__file__).replace('.py', '')+'_loss.pth'

    print(f'File {os.path.basename(__file__)}')
    print('Stage 1')
    if os.path.exists(model_save_path) and os.path.exists(model_loss_save_path):
        print('Found existing model, continuing')
    else:
        model.freeze()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable parameters: {total_params}')
        for i in range(1, epochs+1):
            print(f'\nepoch {i}/{epochs}')
            # train
            model.train()
            loss_module.train()
            loss_train.reset()
            bar = tqdm(train_loader)
            for data, label in bar:
                optimiser.zero_grad()
                data, label = move_to(data), move_to(label)
                with autocast():
                    pred = model(**data)
                    out = loss_module(pred, label)

                #out['loss'].backward()
                #optimiser.step()
                scaler.scale(out['loss']).backward()
                scaler.step(optimiser)
                scaler.update()

                loss_train.update(out['loss'].item())
                bar.set_postfix({'loss': loss_train.avg})
            bar.close()

            # eval
            model.eval()
            loss_module.eval()
            loss_eval.reset()
            ccc_hi.reset()
            ccc_cu.reset()
            uar_type.reset()
            uar_country.reset()
            bar = tqdm(val_loader)
            for data, label in bar:
                data, label = move_to(data), move_to(label)
                with torch.no_grad(), autocast():
                    pred = model(**data)
                    out = loss_module(pred, label)

                loss_eval.update(out['loss'].item())
                ccc_hi.update(out['ccc_hi'].item())
                ccc_cu.update(out['ccc_cu'].item())
                uar_type.update(pred['type'], label['type'])
                uar_country.update(pred['country'], label['country'])
                score = {'loss': loss_eval.avg, 'ccc_hi': ccc_hi.avg, 'ccc_cu': ccc_cu.avg,
                         'uar_type': uar_type.uar(), 'uar_country': uar_country.uar()}
                bar.set_postfix(score)
            bar.close()
            patience_cur -= 1
            if loss_eval.avg < loss_eval_best:
                print('model improved from {:.3f} to {:.3f} ({:.3f}%) - saving'.format(loss_eval_best, loss_eval.avg,
                                                                              (1-loss_eval.avg/loss_eval_best)*100))
                loss_eval_best = loss_eval.avg
                out_best = out.copy()
                score_best = score.copy()
                patience_cur = patience
                torch.save(model.state_dict(), model_save_path)
                torch.save(loss_module.state_dict(), model_loss_save_path)
            elif patience_cur < 0:
                print(f'no improvement during patience - stopping early at epoch {i}')
                break
            else:
                print(f'no improvement - patience remaining {patience_cur}')

        i_best = i if i == epochs else i - patience - 1
        shutil.copyfile(model_save_path, model_save_path.replace('.pth', f'_s1e{i_best}.pth'))
        shutil.copyfile(model_loss_save_path, model_loss_save_path.replace('.pth', f'_s1e{i_best}.pth'))

        if out_best is None:
            out_best = out
            score_best = score
        print(f'from best epoch {i_best}:')
        print(f'scores on validation:')
        print('{}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}'.format(*sum(score_best.items(), ())))
        print(f'last batch stats')
        print('loss:             {:.3f}'.format(out_best['loss']))
        print('                  c      t      hi     cu')
        print('task_loss:        {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*list(out_best['task_loss'])))
        print('task_weight:      {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*list(out_best['task_weight'])))
        print('task_uncertainty: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*list(out_best['task_uncertainty'])))

    print('Stage 2')
    model.load_state_dict(torch.load(model_save_path))
    loss_module.load_state_dict(torch.load(model_loss_save_path))
    model.unfreeze()
    optimiser = torch.optim.AdamW(list(model.parameters()) + list(loss_module.parameters()), lr=lr)
    scaler = GradScaler()
    steps_training = len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimiser, num_warmup_steps=epochs_warmup * steps_training,
                                                num_training_steps=epochs * steps_training)
    loss_eval_best = float('inf')
    out_best = None
    score_best = None
    patience_cur = patience

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params}')

    for i in range(1, epochs + 1):
        print(f'\nepoch {i}/{epochs}')
        # train
        model.train()
        loss_module.train()
        loss_train.reset()
        bar = tqdm(train_loader)
        for data, label in bar:
            optimiser.zero_grad()
            data, label = move_to(data), move_to(label)
            with autocast():
                pred = model(**data)
                out = loss_module(pred, label)

            # out['loss'].backward()
            # optimiser.step()
            scaler.scale(out['loss']).backward()
            scaler.step(optimiser)
            scaler.update()

            scheduler.step()
            loss_train.update(out['loss'].item())
            bar.set_postfix({'loss': loss_train.avg})
        bar.close()

        # eval
        model.eval()
        loss_module.eval()
        loss_eval.reset()
        ccc_hi.reset()
        ccc_cu.reset()
        uar_type.reset()
        uar_country.reset()
        bar = tqdm(val_loader)
        for data, label in bar:
            data, label = move_to(data), move_to(label)
            with torch.no_grad(), autocast():
                pred = model(**data)
                out = loss_module(pred, label)

            loss_eval.update(out['loss'].item())
            ccc_hi.update(out['ccc_hi'].item())
            ccc_cu.update(out['ccc_cu'].item())
            uar_type.update(pred['type'], label['type'])
            uar_country.update(pred['country'], label['country'])
            score = {'loss': loss_eval.avg, 'ccc_hi': ccc_hi.avg, 'ccc_cu': ccc_cu.avg,
                     'uar_type': uar_type.uar(), 'uar_country': uar_country.uar()}
            bar.set_postfix(score)
        bar.close()
        patience_cur -= 1
        if loss_eval.avg < loss_eval_best:
            print('model improved from {:.3f} to {:.3f} ({:.3f}%) - saving'.format(loss_eval_best, loss_eval.avg,
                                                                                   abs(1 - loss_eval.avg / loss_eval_best) * 100))
            loss_eval_best = loss_eval.avg
            out_best = out.copy()
            score_best = score.copy()
            patience_cur = patience
            torch.save(model.state_dict(), model_save_path)
            torch.save(loss_module.state_dict(), model_loss_save_path)
        elif patience_cur < 0:
            print(f'no improvement during patience - stopping early at epoch {i}')
            break
        else:
            print(f'no improvement - patience remaining {patience_cur}')

    i_best = i if i == epochs else i - patience - 1
    shutil.copyfile(model_save_path, model_save_path.replace('.pth', f'_s2e{i_best}.pth'))
    shutil.copyfile(model_loss_save_path, model_loss_save_path.replace('.pth', f'_s2e{i_best}.pth'))

    if out_best is None:
        out_best = out
        score_best = score
    print(f'from best epoch {i_best}:')
    print(f'scores on validation:')
    print('{}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {}: {:.3f}'.format(*sum(score_best.items(), ())))
    print(f'last batch stats')
    print('loss:             {:.3f}'.format(out_best['loss']))
    print('                  c      t      hi     cu')
    print('task_loss:        {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*list(out_best['task_loss'])))
    print('task_weight:      {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*list(out_best['task_weight'])))
    print('task_uncertainty: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(*list(out_best['task_uncertainty'])))


if __name__ == '__main__':
    main()
