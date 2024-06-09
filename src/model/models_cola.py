import pytorch_lightning as pl
import torch
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from sklearn import metrics
import torch.nn as nn
import numpy as np
from torchmetrics import AUROC

from src.model.htsat.htsat import HTSATWrapper

import random 
import torchaudio



class Encoder(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn1(x)
        x = self.efficientnet(x)

        y = x.squeeze(3).squeeze(2)

        return y


class EncoderHTSAT(torch.nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(EncoderHTSAT, self).__init__()
        self.encoder = HTSATWrapper()
        self.out_emb = 768
        
    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.encoder(x)

        return y


class Cola(pl.LightningModule):
    def __init__(self, p=0.1, dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet", max_len=251, out_emb=2048):
        super().__init__()
        self.save_hyperparameters()

        self.p = p
        self.dim_fea, self.dim_hidden, self.dim_out = dim_fea, dim_hidden, dim_out
        self.do = torch.nn.Dropout(p=self.p)
        self.input_length = max_len

        if encoder == "efficientnet":
            self.encoder = Encoder(drop_connect_rate=p)
        elif encoder == "htsat":
            self.encoder = EncoderHTSAT()
            self.dim_fea = self.encoder.out_emb
            if dim_hidden > self.dim_fea : self.dim_hidden = self.dim_fea
        self.encoder_model = encoder

        self.middle_enabled = (self.dim_fea != self.dim_hidden)
        if self.middle_enabled:
            self.middle = torch.nn.Linear(self.dim_fea, self.dim_hidden)

        self.g = torch.nn.Linear(self.dim_hidden, self.dim_out)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.dim_out)
        self.linear = torch.nn.Linear(self.dim_out, self.dim_out, bias=False)

    def forward(self, x):
        # print(x)
        x1, x2 = x

        if self.middle_enabled:
            x1 = self.do(self.middle(self.encoder(x1)))
        else:
            x1 = self.do(self.encoder(x1))
        x1 = self.do(self.g(x1))
        x1 = self.do(torch.tanh(self.layer_norm(x1)))

        if self.middle_enabled:
            x2 = self.do(self.middle(self.encoder(x2)))
        else:
            x2 = self.do(self.encoder(x2))
        x2 = self.do(self.g(x2))
        x2 = self.do(torch.tanh(self.layer_norm(x2)))

        x1 = self.linear(x1)

        return x1, x2

    def extract_feature(self, x, dim=1280):
        if self.encoder_model == "vit":
            return self.extract_feature_vit(x, dim)
        x = self.encoder(x)
        if dim == self.dim_fea:
            return x
        if self.middle_enabled:
            x = self.middle(x)
        if dim == self.dim_hidden:
            return x
        x = self.g(x)
        if dim == self.dim_out:
            return x
        raise NotImplementedError

    
    def training_step(self, x, batch_idx):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def test_step(self, x, batch_idx):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class ColaMD(pl.LightningModule):
    def __init__(self, p=0.1, dim_fea=1280, dim_hidden=1280, dim_out=512, encoder="efficientnet", batch_size=128, num_batch=[258.0, 288, 4, 51, 75, 146, 138], out_emb=2048, max_len=251):
        super().__init__()
        self.save_hyperparameters()

        self.p = p
        self.dim_fea, self.dim_hidden, self.dim_out = dim_fea, dim_hidden, dim_out
        self.do = torch.nn.Dropout(p=self.p)
        self.input_length = max_len

        # self.encoder = Encoder(drop_connect_rate=p)
        if encoder == "efficientnet":
            self.encoder = Encoder(drop_connect_rate=p)
        elif encoder == "htsat":
            self.encoder = EncoderHTSAT()
            self.dim_fea = self.encoder.out_emb
            if dim_hidden > self.dim_fea : self.dim_hidden = self.dim_fea
        self.encoder_model = encoder
        print(num_batch)
        self.num_batch = [b/np.sum(num_batch) for b in num_batch]
        print(self.num_batch)

        self.middle_enabled = (self.dim_fea != self.dim_hidden)
        if self.middle_enabled:
            self.middle = torch.nn.Linear(self.dim_fea, self.dim_hidden)

        self.g = torch.nn.Linear(self.dim_hidden, self.dim_out)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=dim_out)
        self.linear = torch.nn.Linear(dim_out, dim_out, bias=False)
        self.batch_size = batch_size

    def forward(self, x):
        x1, x2 = x

        if self.middle_enabled:
            x1 = self.do(self.middle(self.encoder(x1)))
        else:
            x1 = self.do(self.encoder(x1))
        x1 = self.do(self.g(x1))
        x1 = self.do(torch.tanh(self.layer_norm(x1)))

        if self.middle_enabled:
            x2 = self.do(self.middle(self.encoder(x2)))
        else:
            x2 = self.do(self.encoder(x2))
        x2 = self.do(self.g(x2))
        x2 = self.do(torch.tanh(self.layer_norm(x2)))

        x1 = self.linear(x1)

        return x1, x2

    def extract_feature(self, x, dim=1280):
        x = self.encoder(x)
        if dim == self.dim_fea:
            return x
        if self.middle_enabled:
            x = self.middle(x)
        if dim == self.dim_hidden:
            return x
        x = self.g(x)
        if dim == self.dim_out:
            return x
        raise NotImplementedError
    
    def _calculate_loss(self, x, batch_idx, mode):
        x1, x2 = self(x)

        y = torch.arange(x1.size(0), device=x1.device)

        y_hat = torch.mm(x1, x2.t())

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("{}_loss".format(mode), loss, batch_size=self.batch_size)
        self.log("{}_acc".format(mode), acc, batch_size=self.batch_size)
        return loss
    
    def training_step(self, x, batch_idx):
        """
        covidbreath Length of Training, Validation, Testing: 258 29 29
        covidcough Length of Training, Validation, Testing: 288 32 32
        icbhi Length of Training, Validation, Testing: 4 1 1
        coughvid Length of Training, Validation, Testing: 51 6 6
        hf_lung Length of Training, Validation, Testing: 75 9 9
        covidUKexhalation Length of Training, Validation, Testing: 146 17 17
        covidUKcough Length of Training, Validation, Testing: 138 16 16
        """

        batch, batch_idx, dataloader_idx = x
        lst = range(len(batch))
        
        s = random.choices(lst, weights=self.num_batch, k=1)[0]
        loss = self._calculate_loss(batch[s], batch_idx, "train" + str(s))
        return loss

    def validation_step(self, x, batch_idx, dataloader_idx=0):
        batch, batch_idx, dataloader_idx = x
        
        self._calculate_loss(batch, batch_idx, "valid")

    def test_step(self, x, batch_idx):
        batch, batch_idx, dataloader_idx = x
        self._calculate_loss(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def weights_init(network, seed=None):
    # if seed is not None:
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)

    for m in network:
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Linear') != -1:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()


class AudioClassifier(pl.LightningModule):
    def __init__(self, net, head="mlp", feat_dim=1280, classes=4, lr=1e-4, loss_func=None, freeze_encoder="none", l2_strength=0.0005, random_seed=1):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder
        # self.l2_strength = l2_strength
        # print(self.net)
        if freeze_encoder == "all":
            for param in self.net.parameters():
                param.requires_grad = False
        # elif freeze_encoder == "last":
        #     # print(self.net)
        #     # Selective freezing (fine-tuning only the last few layers), name not matching yet
        #     for name, param in self.net.named_parameters():
        #         if  'efficientnet._blocks.15.' in name or 'efficientnet._blocks.11.' in name or "efficientnet._blocks.12." in name or "efficientnet._blocks.13." in name or "efficientnet._blocks.14." in name: 
        #             print(name)
        #             param.requires_grad = True  # Unfreeze for fine-tuning
        #         elif 'layers.3' in name or "htsat.norm" in name or "htsat.head" in name or "htsat.tscam_conv" in name:
        #             print(name)
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False  # Freeze the rest
        elif freeze_encoder == "early":
            # print(self.net)
            # Selective freezing (fine-tuning only the last few layers), name not matching yet
            for name, param in self.net.named_parameters():
                # print(name)
                if 'cnn1' in name or 'efficientnet._blocks.0.' in name or 'efficientnet._blocks.1.' in name or "efficientnet._blocks.2." in name or "efficientnet._blocks.3." in name or "efficientnet._blocks.4." in name: 
                    # for efficientnet
                    param.requires_grad = True
                    print(name)
                elif 'patch_embed' in name or 'layers.0' in name or 'layers.1' in name or 'layers.2' in name or "htsat.norm" in name or "htsat.head" in name or "htsat.tscam_conv" in name:
                    # for htsat
                    param.requires_grad = True
                    print(name)
                else:
                    param.requires_grad = False
                    # print(name)

        if head == 'linear':
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        

        weights_init(self.head, seed=random_seed)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

    def forward_feature(self, x):
        return self.net(x)

    def forward(self, x):
        x = self.net(x)
        return self.head(x)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        probabilities = F.softmax(y_hat, dim=1)  
        return probabilities

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

         # Apply L2 regularization on head
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_head", l2_regularization)
        loss += self.l2_strength_new_layers * l2_regularization

        # Apply L2 regularization on encoder
        l2_regularization = 0
        for param in self.net.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_encoder", l2_regularization)
        loss += self.l2_strength_encoder * l2_regularization

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))
    
    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        print("test_auc", auc)
        self.log("test_auc", auc)
        
        self.test_step_outputs.clear() 
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class AudioClassifierAudioMAE(pl.LightningModule):
    def __init__(self, net, head="mlp", feat_dim=1280, classes=4, lr=1e-4, loss_func=None, freeze_encoder="none", l2_strength=0.0005, random_seed=1):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder

        # print(self.net)
        
        if head == 'linear':
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        

        weights_init(self.head, seed=random_seed)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

    def forward_feature(self, x):
        return self.net.forward_feature(x)

    def forward(self, x):
        x = self.net.forward_feature(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

         # Apply L2 regularization on head
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_head", l2_regularization)
        loss += self.l2_strength_new_layers * l2_regularization

        # Apply L2 regularization on encoder
        l2_regularization = 0
        for param in self.net.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_encoder", l2_regularization)
        loss += self.l2_strength_encoder * l2_regularization

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))
    
    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        print("test_auc", auc)
        self.log("test_auc", auc)
        
        self.test_step_outputs.clear() 
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



class AudioClassifierCLAP(pl.LightningModule):
    def __init__(self, net, head="mlp", feat_dim=1280, classes=4, lr=1e-4, loss_func=None, freeze_encoder="none", l2_strength=0.0005, random_seed=1):
        super().__init__()
        self.net = net
        self.freeze_encoder = freeze_encoder
        # self.l2_strength = l2_strength
        # print(self.net)
        self.net.train()
    
        
        if freeze_encoder == "all":
            for param in self.encoder.parameters():
                param.requires_grad = False
        # elif freeze_encoder == "selective":
        #     # Selective freezing (fine-tuning only the last few layers)
        #     for name, param in self.encoder.named_parameters():
        #         if 'layer3' in name or 'layer4' in name or 'fc' in name: 
        #             param.requires_grad = True  # Unfreeze for fine-tuning
        #         else:
        #             param.requires_grad = False  # Freeze the rest
        else:
            for param in self.net.parameters():
                param.requires_grad = True
        
        if head == 'linear':
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        

        weights_init(self.head, seed=random_seed)
        self.lr = lr
        # self.l2_strength = l2_strength
        self.l2_strength_new_layers = l2_strength
        self.l2_strength_encoder = l2_strength * 0.2
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))
    
    def read_audio(self, audio_path, resample=True):
        r"""Loads audio file or array and returns a torch tensor"""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        
        resample_rate = 16000
        # print(sample_rate)
        if resample and resample_rate != sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        return audio_time_series, resample_rate

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=False):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = self.read_audio(audio_path, resample=resample)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(
                audio_file, 5, resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def forward_feature(self, x, resample=True):
        preprocessed_audio = self.preprocess_audio(x, resample)
        preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
        audio_embed, _ = self.net(preprocessed_audio)
        return audio_embed
    
    def forward(self, x, resample=True):
        preprocessed_audio = self.preprocess_audio(x, resample)
        preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
        # print(preprocessed_audio.shape)
        # return self._get_audio_embeddings(preprocessed_audio)
        # x = self.net.get_audio_embeddings(x)
        audio_embed, _ = self.net(preprocessed_audio)
        # print(audio_embed.shape)
        return self.head(audio_embed)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)

         # Apply L2 regularization on head
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_head", l2_regularization)
        loss += self.l2_strength_new_layers * l2_regularization

        # Apply L2 regularization on encoder
        l2_regularization = 0
        # for param in self.net.clap.audio_encoder.parameters():
        for param in self.net.parameters():
            l2_regularization += param.pow(2).sum()

        self.log("train_l2_encoder", l2_regularization)
        loss += self.l2_strength_encoder * l2_regularization

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))
    
    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        print("test_auc", auc)
        self.log("test_auc", auc)
        
        self.test_step_outputs.clear() 
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LinearHead(pl.LightningModule):
    def __init__(self, net=None, head="mlp", feat_dim=1280, classes=4, from_feature=True, lr=1e-4, loss_func=None, l2_strength=0.0005, random_seed=1):
        super().__init__()
        self.from_feature = from_feature

        if not from_feature:
            if net is None:
                raise ValueError(
                'no encoder given and not from feature input')
            self.net = net
            for param in self.net.parameters():
                param.requires_grad = False
        
        if head == 'linear':
            print(feat_dim, classes)
            self.head = nn.Sequential(nn.Linear(feat_dim, classes))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, classes)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        # self.head = nn.Linear(dim_in, dim_out)

        weights_init(self.head, seed=random_seed)
        self.lr = lr
        self.l2_strength = l2_strength
        self.loss = loss_func if loss_func else nn.CrossEntropyLoss()
        self.classes = classes
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()
            
    def forward(self, x):
        if self.from_feature:
            return self.head(x)
        
        with torch.no_grad():
            x = self.net(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x) + 1e-10
        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
         # Apply L2 regularization
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()
        self.log("train_l2", l2_regularization)
        loss += self.l2_strength * l2_regularization

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()


        self.log("train_acc", acc)
        # print("train_loss", loss)
        # print("train_acc", acc)f

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_hat = self(x)

        # print(y_hat, y)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

        self.validation_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)

        probabilities = F.softmax(y_hat, dim=1)  

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).double().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.test_step_outputs.append((y.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy() ))
    
    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        # print("valid_auc", auc)
        self.log("valid_auc", auc)

        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        predicted = np.concatenate([output[1] for output in all_outputs])
        probs = np.concatenate([output[2] for output in all_outputs])

        auroc = AUROC(task="multiclass", num_classes=self.classes)
        auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))
        
        print("test_auc", auc)
        self.log("test_auc", auc)
        
        self.test_step_outputs.clear() 
        return auc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LinearHeadR(pl.LightningModule):
    def __init__(self, net=None, head="mlp", feat_dim=1280, output_dim=1, from_feature=True, lr=1e-4, loss_func=None, l2_strength=0.0005, random_seed=1):
        super().__init__()
        self.from_feature = from_feature

        if not from_feature:
            if net is None:
                raise ValueError(
                'no encoder given and not from feature input')
            self.net = net
            for param in self.net.parameters():
                param.requires_grad = False
        
        if head == 'linear':
            # print(feat_dim, output_dim)
            self.head = nn.Sequential(nn.Linear(feat_dim, output_dim))
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, output_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
        # self.head = nn.Linear(dim_in, dim_out)

        weights_init(self.head, seed=random_seed)
        self.lr = lr
        self.l2_strength = l2_strength
        self.loss = loss_func if loss_func else nn.MSELoss()
        self.classes = output_dim
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        # self.fc.bias.data.zero_()
            
    def forward(self, x):
        
        if self.from_feature:
            return self.head(x)
        
        with torch.no_grad():
            x = self.net(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) + 1e-10
        #print(y_hat, y)
        

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        # print('training loss:', loss.item())
        
         # Apply L2 regularization
        l2_regularization = 0
        for param in self.head.parameters():
            l2_regularization += param.pow(2).sum()
        loss += self.l2_strength * l2_regularization

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
   
        loss = self.loss(y_hat, y)      
        mae = torch.mean(torch.abs(y_hat - y))
        mape = torch.mean(torch.abs((y_hat - y) / y)) * 100
   
        self.log("valid_loss", loss)
        self.log("valid_MAE", mae)
        self.log("valid_MAPE", mape)
 
        self.validation_step_outputs.append((y.cpu().numpy(), y_hat.cpu().numpy() ))

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        mape = torch.mean(torch.abs((y_hat - y) / y)) * 100
   
        self.log("test_loss", loss)
        self.log("test_MAE", mae)
        self.log("test_MAPE", mape)
        self.test_step_outputs.append((y.cpu().numpy(), y_hat.cpu().numpy() ))
    
    def on_validation_epoch_end(self):
        all_outputs = self.validation_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        y_hat = np.concatenate([output[1] for output in all_outputs])
        
        mae = np.mean(np.abs(y_hat - y))
        mape = np.mean(np.abs((y_hat - y) / y)) * 100
        mse = np.mean((y - y_hat) ** 2)
        self.log("valid_MAE", mae)
        self.log("valid_MAPE", mape)
        self.log("valid_loss", mse)
        # print('valid_mae:', mae, 'valid y:', y, 'valid y_hat:', y_hat)

        self.validation_step_outputs.clear() 

    def on_test_epoch_end(self):
        all_outputs = self.test_step_outputs
        y = np.concatenate([output[0] for output in all_outputs])
        y_hat = np.concatenate([output[1] for output in all_outputs])
        
        mae = np.mean(np.abs(y_hat - y))
        mape = np.mean(np.abs((y_hat - y) / y)) * 100
        mse = np.mean((y - y_hat) ** 2)
        self.log("test_MAE", mae)
        self.log("test_MAPE", mape)
        self.log("test_loss", mse)
        print('test_mae:', mae, y_hat, y)
        
        self.test_step_outputs.clear() 
        return mae

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
