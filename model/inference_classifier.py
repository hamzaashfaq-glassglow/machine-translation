import torch
import torch.nn as nn
import pickle


class Vocab:
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.next_idx = 0

    def encode(self, text):
        tokens = text.lower().split()
        return [self.w2i.get(t, self.w2i.get('<UNK>', 3)) for t in tokens]

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx in (1, 2):   # <SOS> / <EOS>
                continue
            word = self.i2w.get(idx, '')
            if word:
                words.append(word)
        return ' '.join(words)

    def __len__(self):
        return self.next_idx


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Vocab':
            return Vocab
        return super().find_class(module, name)

# EXACT SAME CLASSES FROM TRAINING
class Attn(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.attn = nn.Linear(hdim*4, hdim)
        self.v = nn.Linear(hdim, 1, bias=False)
    
    def forward(self, dec_h, enc_out):
        batch = dec_h.shape[0]
        seq_len = enc_out.shape[0]
        dec_h = dec_h.unsqueeze(1).expand(-1, seq_len, -1)
        enc_out = enc_out.transpose(0,1)
        energy = torch.tanh(self.attn(torch.cat([dec_h, enc_out], dim=2)))
        attn = self.v(energy).squeeze(2)
        attn = torch.softmax(attn, dim=1)
        ctx = torch.einsum('bs,bsh->bh', attn, enc_out)
        return ctx, attn

class Enc(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hdim, nl=1, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hdim, nl, dropout=drop, bidirectional=True)
    
    def forward(self, src):
        emb = self.emb(src)
        out, (h, c) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        c = torch.cat([c[-2], c[-1]], dim=1)
        return out, (h, c)

class Dec(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hdim, out_dim, nl=1, drop=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.attn = Attn(hdim)
        self.lstm = nn.LSTM(emb_dim+hdim*2, hdim*2, nl, dropout=drop)
        self.fc = nn.Linear(hdim*4, out_dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, tok, h, c, enc_out):
        tok = tok.unsqueeze(0)
        emb = self.drop(self.emb(tok))
        ctx, attn = self.attn(h, enc_out)
        lstm_in = torch.cat([emb, ctx.unsqueeze(0)], dim=2)
        out, (h, c) = self.lstm(lstm_in, (h.unsqueeze(0), c.unsqueeze(0)))
        out = out.squeeze(0)
        pred = self.fc(torch.cat([out, ctx], dim=1))
        return pred, h.squeeze(0), c.squeeze(0), attn

class S2S(nn.Module):
    def __init__(self, enc, dec, device):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.device = device

# INFERENCE CLASS
class Translator:
    def __init__(self, model_path='seq2seq_model.pth', 
                 vocab_ur_path='vocab_urdu.pkl',
                 vocab_en_path='vocab_english.pkl',
                 device='cpu'):
        
        self.device = torch.device(device)
        
        # Load vocabularies
        with open(vocab_ur_path, 'rb') as f:
            self.v_ur = _VocabUnpickler(f).load()
        with open(vocab_en_path, 'rb') as f:
            self.v_en = _VocabUnpickler(f).load()
        
        print(f'Urdu vocab size: {len(self.v_ur)}')
        print(f'English vocab size: {len(self.v_en)}')
        
        # Initialize model with EXACT same config as training
        enc = Enc(len(self.v_ur), 128, 256, nl=1, drop=0.0)
        dec = Dec(len(self.v_en), 128, 256, len(self.v_en), nl=1, drop=0.0)
        self.model = S2S(enc, dec, self.device).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f'Model loaded from {model_path}')
        print(f'Device: {self.device}')
    
    def translate(self, urdu_text, max_len=50):
        """Translate Roman Urdu to English"""
        with torch.no_grad():
            # Encode
            src_idx = [1] + self.v_ur.encode(urdu_text) + [2]
            src_t = torch.tensor(src_idx).unsqueeze(1).to(self.device)
            
            enc_out, (h, c) = self.model.enc(src_t)
            
            # Decode
            trg_idx = [1]
            dec_in = torch.tensor([1]).to(self.device)
            
            for _ in range(max_len):
                pred, h, c, _ = self.model.dec(dec_in, h, c, enc_out)
                tok = pred.argmax(1).item()
                trg_idx.append(tok)
                
                if tok == 2:  # <EOS>
                    break
                
                dec_in = torch.tensor([tok]).to(self.device)
            
            return self.v_en.decode(trg_idx)


# TEST
if __name__ == '__main__':
    translator = Translator(device='cpu')
    
    test_inputs = [
        'aap khan thai',
        'ktabin prhna pasand hai'
    ]
    
    print('\nTranslations:')
    for urdu in test_inputs:
        english = translator.translate(urdu)
        print(f'  UR: {urdu}')
        print(f'  EN: {english}\n')