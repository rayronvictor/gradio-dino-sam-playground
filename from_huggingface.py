import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Tente usar as bibliotecas, senão peça ao usuário para instalar
try:
    from transformers import SamModel, SamProcessor
    import tensorflow
except ImportError:
    print("Algumas bibliotecas necessárias não foram encontradas.")
    print("Por favor, instale-as com: pip install transformers torch torchvision Pillow matplotlib tensorflow")
    exit()


def segment_from_local_image(image_path: str, prompt_points: list):
    """
    Carrega o modelo nasskall/vitiligo, segmenta uma imagem local com base em pontos de prompt e exibe o resultado.

    Args:
        image_path (str): O caminho para a imagem local a ser segmentada.
        prompt_points (list): Uma lista de listas contendo as coordenadas [x, y] dos prompts.
                              Ex: [[[x1, y1]], [[x2, y2]]]
    """
    # 1. Carregar a imagem local
    print(f"Carregando a imagem local de: {image_path}")
    if not os.path.exists(image_path):
        print(f"Erro: O arquivo de imagem não foi encontrado no caminho especificado: {image_path}")
        print("Verifique se o caminho está correto e se a estrutura de pastas (ex: 'assets/sua_imagem.jpeg') existe.")
        return

    try:
        raw_image = Image.open(image_path).convert("RGB")
        print("Imagem carregada com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return

    print("Carregando o modelo 'nasskall/vitiligo' do Hugging Face...")

    # 2. Carregar o modelo e o processador
    try:
        model = SamModel.from_pretrained("nasskall/vitiligo", from_tf=True)
        processor = SamProcessor.from_pretrained("nasskall/vitiligo")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    # 3. Preparar a imagem e os prompts para o modelo
    print("Processando a imagem e os prompts...")
    inputs = processor(raw_image, input_points=prompt_points, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Executando o modelo no dispositivo: {device}")

    # 4. Executar o modelo
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Pós-processar as máscaras
    print("Gerando máscara de segmentação...")
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    segmentation_mask = masks[0][0][0].cpu().numpy()

    # 6. Visualizar o resultado
    print("Exibindo o resultado...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(np.array(raw_image))
    # Extrai os pontos para plotagem, garantindo que o formato esteja correto
    points_np = np.array(prompt_points).squeeze(axis=1)
    ax[0].scatter(points_np[:, 0], points_np[:, 1], color='green', marker='*', s=150, edgecolor='white')
    ax[0].set_title("Imagem Original com Prompt")
    ax[0].axis('off')

    ax[1].imshow(np.array(raw_image))
    mask_color = np.array([0, 1, 0, 0.6])  # RGBA: Verde com 60% de opacidade
    mask_image = np.ma.masked_where(segmentation_mask == 0, np.tile(mask_color, (*segmentation_mask.shape, 1)))
    ax[1].imshow(mask_image)
    ax[1].set_title("Resultado da Segmentação")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- Parâmetros ---
    # 1. Defina o caminho para a sua imagem local
    IMAGE_PATH = "assets/vitiligo_face01.jpeg"

    # 2. !! IMPORTANTE !! Ajuste as coordenadas do ponto para a sua imagem.
    # O ponto [x, y] deve estar sobre a área que você quer segmentar.
    # Você pode usar um editor de imagens (Paint, GIMP, etc.) para encontrar as coordenadas.
    INPUT_POINTS = [[[450, 350]]]  # <-- AJUSTE ESTE PONTO PARA SUA IMAGEM

    # Chama a função principal com o caminho da imagem local
    segment_from_local_image(IMAGE_PATH, INPUT_POINTS)