# loss.py  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  


class InfoNCELoss(torch.nn.Module):  
    def __init__(self, temperature=0.07):  
        super(InfoNCELoss, self).__init__()  
        self.temperature = temperature  
  
    def forward(self, query, key, negatives):  
        """  
        Args:  
            query: A tensor of shape (batch_size, d_model)  
            key: A tensor of shape (batch_size, d_model)  
            negatives: A tensor of shape (batch_size, num_negatives, d_model)  

        High Temperature: The model is more relaxed and treats the differences between positive (similar) and negative (dissimilar)
        samples more evenly. This can be useful when you're 
        starting to train your model and want it to explore different possibilities without making strong decisions too early 

        Low Temperature: The model becomes very confident and focuses almost entirely on the most similar (positive) samples,
        while ignoring the rest. This can be useful when you want
        your model to make very clear distinctions, but if you do it too early or too much, it might ignore important details
        """  
        batch_size = query.shape[0]  
        d_model = query.shape[1]  
        num_negatives = negatives.shape[1]  
  
        # Normalize the feature vectors  
        query = F.normalize(query, dim=1)  
        key = F.normalize(key, dim=1)  
        negatives = F.normalize(negatives, dim=2)  
  
        # Compute positive logits: (batch_size,)  
        positive_logits = torch.sum(query * key, dim=1) / self.temperature  
  
        # Compute negative logits: (batch_size, num_negatives)  
        # We need to reshape the negatives tensor for the matrix multiplication  
        negative_logits = torch.bmm(negatives, query.unsqueeze(2)).squeeze(2) / self.temperature  
  
        # Combine logits: (batch_size, 1 + num_negatives)  
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)  
  
        # Create labels: (batch_size,)  
        labels = torch.zeros(batch_size, dtype=torch.long).to(query.device)  
  
        # Compute cross-entropy loss  
        loss = F.cross_entropy(logits, labels)  
          
        return loss  