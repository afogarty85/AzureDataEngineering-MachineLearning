import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention  
import math  
  

  

class LearnablePositionalEncoding(nn.Module):  
    def __init__(self, d_model, max_len, dropout=0.1):  
        super(LearnablePositionalEncoding, self).__init__()  
        self.dropout = nn.Dropout(p=dropout)  
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))  # Learnable positional encoding  
  
    def forward(self, x):  
        x = x + self.pos_embedding[:, :x.size(1), :]  
        return self.dropout(x)  


class TabularTransformerEncoder(nn.Module):  
    def __init__(self, embedding_table_shapes, num_cont_features, num_bin_features, num_text_features, d_model, nhead, num_layers, dim_feedforward, noise_std, dropout=0.1, num_classes=3, max_len=251):  
        super(TabularTransformerEncoder, self).__init__()  
  
        self.d_model = d_model  
  
        # Continuous features MLP  
        self.cont_mlp = nn.Sequential(  
            nn.Linear(num_cont_features, d_model),  
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )  
  
        # Categorical features embeddings  
        self.cat_embeddings = nn.ModuleDict({  
            feature: nn.Embedding(cardinality, d_model, padding_idx=0)  
            for feature, (cardinality, _) in embedding_table_shapes.items()  
        })  
  
        # Text features linear transformation  
        self.text_fc = nn.Linear(num_text_features, d_model)  
  
        # Binary features linear transformation  
        self.bin_fc = nn.Linear(num_bin_features, d_model)  
  
        # Combining layer  
        total_d_model = (1 + len(embedding_table_shapes) + 1 + 1) * d_model  
        self.combining_fc = nn.Linear(total_d_model, d_model)  
  
        # Positional Encoding  
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)   
  
        # Transformer encoder layer  
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)  
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)  
  
        # Pooling layer  
        self.pooling_fc = nn.Linear(d_model, d_model)  
  
        # Noise  
        self.noise_std = noise_std  

        # Classification head  
        self.classification_head = nn.Linear(d_model, num_classes)  

    def forward(self, x_cat, x_bin, x_cont, x_text, mask_cat, mask_bin, mask_cont):  
        # Debug: Check input tensors for NaNs  
        for name, tensor in {'x_bin': x_bin, 'x_cont': x_cont, 'x_text': x_text}.items():  
            if torch.isnan(tensor).any():  
                print(f"NaNs detected in input tensor: {name}")  
  
        for feature, tensor in x_cat.items():  
            if torch.isnan(tensor).any():  
                print(f"NaNs detected in input tensor: x_cat[{feature}]")  
  
        # x_cat: dict of tensors with keys as categorical feature names  
        # x_bin: [batch_size, seq_len, num_bin_features]  
        # x_cont: [batch_size, seq_len, num_cont_features]  
        # x_text: [batch_size, seq_len, num_text_features]  
        # mask_*: [batch_size, seq_len, 1]  
  
        batch_size, seq_len = x_cont.shape[0], x_cont.shape[1]  
  
        # Process continuous features  
        x_cont = self.cont_mlp(x_cont)  # [batch_size, seq_len, d_model]  
        x_cont = x_cont * mask_cont  # Apply mask to continuous features  
  
        # Process categorical features  
        cat_embeddings = [self.cat_embeddings[feature](x_cat[feature]) * mask_cat for feature in x_cat]  
        x_cat = torch.cat(cat_embeddings, dim=-1)  # [batch_size, seq_len, len(embedding_table_shapes) * d_model]  
  
        # Process binary features  
        x_bin = self.bin_fc(x_bin)  # [batch_size, seq_len, d_model]  
        x_bin = x_bin * mask_bin  # Apply mask to binary features  
  
        # Process text features  
        x_text = self.text_fc(x_text)  # [batch_size, seq_len, d_model]  
        # no mask to text features  
  
        # Concatenate features  
        x = torch.cat((x_cont, x_cat, x_bin, x_text), dim=-1)  # [batch_size, seq_len, total_d_model]  
  
        # Adding Gaussian noise  
        if self.training and self.noise_std > 0.0:  
            noise = torch.normal(mean=0.0, std=self.noise_std, size=x.size()).to(x.device)  
            x = x + noise  
  
        # Linear transformation to match transformer d_model  
        x = self.combining_fc(x)  # [batch_size, seq_len, d_model]  
  
        # Add positional encodings  
        x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]  
    
        # Combine masks to create a unified mask for the transformer encoder  
        # mask_*: [batch_size, seq_len, 1]  
        combined_mask = mask_cont.squeeze(-1) & mask_bin.squeeze(-1) & mask_cat.squeeze(-1)  # [batch_size, seq_len]  
    
        # Apply transformer encoder  
        x = self.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=~combined_mask).transpose(0, 1)  # [batch_size, seq_len, d_model]  
    
        # Get the last hidden state for each sequence  
        # Find the index of the last valid token for each sequence  
        last_valid_indices = combined_mask.sum(dim=1) - 1  # [batch_size]  
        last_valid_indices = last_valid_indices.clamp(min=0)  # Ensure no negative indices  
    
        # Gather the last hidden state for each sequence  
        pooled_output = x[torch.arange(batch_size), last_valid_indices]  # [batch_size, d_model]  
    
        # Final transformation  
        pooled_output = self.pooling_fc(pooled_output)  # [batch_size, d_model]  
    
        # Debug: Check output tensors for NaNs  
        if torch.isnan(pooled_output).any():  
            print("NaNs detected in the output tensor.")  
    
        # Classification logits  
        logits = self.classification_head(pooled_output)  # [batch_size, num_classes]  
  
        return pooled_output, logits  



      
  
