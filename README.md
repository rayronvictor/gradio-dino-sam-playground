```bash
# cria a env
conda env create -f environment.yaml

# ativa o ambiente
conda activate gradio-playground

# instala as dependências
pip install -r requirements.txt

# executa o projeto
python main.py # use --cpu para rodar na CPU e não na GPU
``` 