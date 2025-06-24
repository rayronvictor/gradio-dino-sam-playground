from transformers import SamModel
import torch

print("Baixando modelo do Hugging Face...")
model = SamModel.from_pretrained(
    "nasskall/vitiligo",
    from_tf=True,
    ignore_mismatched_sizes=True
)

print("Salvando modelo convertido...")
torch.save(model.state_dict(), "checkpoints/vitiligo_sam_finetuned.pth")
print("Modelo salvo com sucesso!")
