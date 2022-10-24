import os
import shutil

import numpy as np
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

processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
#processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-base")


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
        ret = {'File_ID': '['+dfi['File_ID']+']'}
        return sample, ret

    def __len__(self):
        return len(self.df)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.requires_grad_(True)

        self.base_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
        #self.base_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")

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
    batch_size = 32
    num_workers = 0 if os.name == 'nt' else 2
    torch.backends.cudnn_benchmark_enabled = True

    test_loader = DataLoader(dataset=AvbDataset(split='Test'),
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True
                              )

    model = Model()
    model = model.to(device)
    model.load_state_dict(torch.load('model/mtl4d2vb_s2e16.pth', map_location=torch.device(device)))
    #model.load_state_dict(torch.load('model/mtl4rmsgmd_s2e23.pth', map_location=torch.device(device)))

    print('Inference on test set')
    model.eval()

    df = pd.read_csv('dataset.csv', dtype={'File_ID': str}).iloc[:0, :].copy()
    df = df.drop(columns=['Split', 'Valence', 'Arousal'])

    bar = tqdm(test_loader)
    for data, label in bar:
        data = move_to(data)
        with torch.no_grad():
            pred = model(**data)
        pred['country'] = torch.argmax(pred['country'], dim=-1, keepdim=True)
        pred['type'] = torch.argmax(pred['type'], dim=-1, keepdim=True)
        ret = torch.concat((pred['country'], pred['high'], pred['culture'], pred['type']), dim=-1)
        ret = ret.cpu()
        ret = ret.numpy()
        ret = np.concatenate((np.expand_dims(label['File_ID'], axis=1), ret), axis=1)
        df = pd.concat([df, pd.DataFrame(ret, columns=df.columns)], ignore_index=True)

    bar.close()
    df.Country = df.Country.apply(lambda x: i2c[int(float(x))])
    df.Voc_Type = df.Voc_Type.apply(lambda x: i2t[int(float(x))])
    df.to_csv(os.path.basename(__file__).replace('py', 'csv'), index=False)


if __name__ == '__main__':
    main()
