import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. SOTA MODEL ARCHITECTURE
# ==========================================
class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, out_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, out_dim, kernel_size=3, padding=1)
        
    def forward(self, smiles_tokens):
        x = self.embed(smiles_tokens).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)

class HGNN_SA(nn.Module):
    def __init__(self, num_drugs, num_ses, smiles_vocab_size):
        super().__init__()
        self.drug_id_emb = nn.Embedding(num_drugs, 64)
        self.smiles_enc = SmilesEncoder(smiles_vocab_size, embed_dim=64, out_dim=64)
        self.node_proj = nn.Linear(128, 128)
        
        self.se_emb = nn.Embedding(num_ses, 128)
        
        self.conv1 = HypergraphConv(128, 128)
        self.conv2 = HypergraphConv(128, 128)
        
        self.attn_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        
        self.mlp = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, hyperedge_index, se_indices, all_smiles_tokens):
        # 1. Chemical + ID Features
        smiles_feats = self.smiles_enc(all_smiles_tokens)
        node_feats = F.relu(self.node_proj(torch.cat([self.drug_id_emb.weight, smiles_feats], dim=1)))
        
        # 2. Graph Message Passing
        x = F.relu(self.conv1(node_feats, hyperedge_index))
        x = F.relu(self.conv2(x, hyperedge_index))
        
        # 3. Multi-Head Attention
        drugs = x[hyperedge_index[0]]
        attn_probs = scatter_softmax(self.attn_layer(drugs), hyperedge_index[1], dim=0)
        combo_vector = scatter_add(drugs * attn_probs, hyperedge_index[1], dim=0)
        
        # 4. Predict
        out = torch.cat([combo_vector, self.se_emb(se_indices)], dim=1)
        return self.mlp(out).squeeze(-1)

# ==========================================
# 2. DATA PREPARATION (Strict Quarter Split)
# ==========================================
def prepare_sota_data():
    print("Loading Dataset and SMILES...")
    df = pd.read_csv("data/hoddi_merged.csv")
    df['DrugBankID'] = df['DrugBankID'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['target'] = df['hyperedge_label'].apply(lambda x: 0 if x == -1 else 1)
    
    smiles_df = pd.read_csv("data/Drugbank_ID_SMILE_all_structure links.csv", usecols=['DrugBank ID', 'SMILES'])
    smiles_dict = dict(zip(smiles_df['DrugBank ID'], smiles_df['SMILES']))
    
    all_drugs = sorted(list(set([drug for sublist in df['DrugBankID'] for drug in sublist])))
    drug2idx = {d: i for i, d in enumerate(all_drugs)}
    all_ses = df['SE_label'].unique()
    se2idx = {s: i for i, s in enumerate(all_ses)}
    
    char_vocab = set()
    for s in smiles_dict.values():
        if isinstance(s, str): char_vocab.update(list(s))
    
    char2idx = {c: i+1 for i, c in enumerate(sorted(list(char_vocab)))}
    char2idx['<UNK>'] = len(char2idx) + 1
    
    MAX_SMILES_LEN = 256
    all_smiles_tokens = np.zeros((len(drug2idx), MAX_SMILES_LEN), dtype=int)
    for drug, idx in drug2idx.items():
        smile = smiles_dict.get(drug, "C")
        if not isinstance(smile, str): smile = "C"
        tokens = [char2idx.get(c, char2idx['<UNK>']) for c in list(smile)[:MAX_SMILES_LEN]]
        all_smiles_tokens[idx, :len(tokens)] = tokens
        
    all_smiles_tensor = torch.tensor(all_smiles_tokens, dtype=torch.long)
    
    # STRICT CHRONOLOGICAL SPLIT (70 / 15 / 15)
    unique_quarters = sorted(df['time'].unique())
    num_q = len(unique_quarters)
    train_end = int(0.70 * num_q)
    val_end = train_end + int(0.15 * num_q)
    
    train_q = unique_quarters[:train_end]
    val_q = unique_quarters[train_end:val_end]
    test_q = unique_quarters[val_end:]
    
    print(f"Split -> Train: {len(train_q)} quarters | Val: {len(val_q)} quarters | Test: {len(test_q)} quarters")
    
    splits = {
        'train': df[df['time'].isin(train_q)].reset_index(drop=True),
        'val': df[df['time'].isin(val_q)].reset_index(drop=True),
        'test': df[df['time'].isin(test_q)].reset_index(drop=True)
    }
    
    def build_tensors(split_df):
        d_idx, r_idx, s_idx, labels = [], [], [], []
        for i, row in split_df.iterrows():
            for d in row['DrugBankID']:
                d_idx.append(drug2idx[d])
                r_idx.append(i)
            s_idx.append(se2idx[row['SE_label']])
            labels.append(row['target'])
        return (torch.tensor([d_idx, r_idx], dtype=torch.long), 
                torch.tensor(s_idx, dtype=torch.long), 
                torch.tensor(labels, dtype=torch.float))

    data = {k: build_tensors(v) for k, v in splits.items()}
    data.update({'num_drugs': len(drug2idx), 'num_ses': len(se2idx), 
                 'drug2idx': drug2idx, 'se2idx': se2idx, 
                 'char2idx': char2idx, 'smiles_tensor': all_smiles_tensor})
    return data

# ==========================================
# 3. TRAINING AND EVALUATION LOOP
# ==========================================
def train_and_evaluate(max_epochs=500):
    data = prepare_sota_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining SOTA HGNN-SA on {device}...")
    
    model = HGNN_SA(data['num_drugs'], data['num_ses'], len(data['char2idx'])+1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    
    train_h, train_s, train_y = [t.to(device) for t in data['train']]
    val_h, val_s, val_y = [t.to(device) for t in data['val']]
    smiles_t = data['smiles_tensor'].to(device)
    
    best_auc = 0
    patience_counter = 0
    early_stop_patience = 25 
    
    # --- TRAINING PHASE ---
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_h, train_s, smiles_t)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out_v = model(val_h, val_s, smiles_t)
            probs = torch.sigmoid(out_v).cpu().numpy()
            y_true = val_y.cpu().numpy()
            
            auc = roc_auc_score(y_true, probs)
            scheduler.step(auc)
            
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'drug2idx': data['drug2idx'],
                    'se2idx': data['se2idx'],
                    'char2idx': data['char2idx'],
                    'smiles_tensor': data['smiles_tensor']
                }, 'models/HGNN_model.pt')
            else:
                patience_counter += 1
                
        print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val AUC: {auc:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val AUC: {best_auc:.4f}")
            break

    print("\nTraining Complete. Moving to Final Test Set Evaluation...\n")
    
    # --- EVALUATION PHASE ---
    # Load the best model weights
    checkpoint = torch.load('models/HGNN_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_h, test_s, test_y = [t.to(device) for t in data['test']]
    
    print("Running Inference on unseen Future Quarters...")
    with torch.no_grad():
        out_test = model(test_h, test_s, smiles_t)
        probs = torch.sigmoid(out_test).cpu().numpy()
        y_true = test_y.cpu().numpy()
        y_pred = (probs > 0.5).astype(int)
        
    # Calculate Metrics
    test_auc = roc_auc_score(y_true, probs)
    test_pr_auc = average_precision_score(y_true, probs)
    test_f1 = f1_score(y_true, y_pred)
    test_acc = accuracy_score(y_true, y_pred)
    test_prec = precision_score(y_true, y_pred)
    test_rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # --- GENERATE EVALUATION REPORT ---
    report = f"""
======================================================================
EVALUATION REPORT: POLYPHARMACY HGNN-SA
======================================================================
Model Architecture: Hypergraph Neural Network + SMILES CNN + Attention
Data Split strategy: Strict Chronological Quarter-Level (70/15/15)

FINAL TEST SET METRICS (Unseen Future Data):
----------------------------------------------------------------------
ROC-AUC (Main Metric) : {test_auc:.4f}
PR-AUC                : {test_pr_auc:.4f}
F1-Score              : {test_f1:.4f}
Accuracy              : {test_acc:.4f}
Precision             : {test_prec:.4f}
Recall                : {test_rec:.4f}

CONFUSION MATRIX:
                     Predicted Safe (0) | Predicted Toxic (1)
Actual Safe (0)    : {cm[0][0]:<18} | {cm[0][1]}
Actual Toxic (1)   : {cm[1][0]:<18} | {cm[1][1]}
======================================================================
"""
    print(report)
    
    # Save to a text file
    with open("results/evaluate_HGNN.txt", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Saved evaluation results to 'results/evaluate_HGNN.txt'.")

if __name__ == "__main__":
    train_and_evaluate(max_epochs=500)