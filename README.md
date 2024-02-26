# LLM_Modelagem_Deploy_Gradio

Algumas orientações:
Faça a modelagem do algoritmo, no Google Colab, pois é necessário GPU. O arquivo está em .ipynb, justamente para isso e não deve ser levado para o ambiente Spaces do Hugging Face. 

Com o modelo treinado e feito o Fine Tuning, faça o download do modelo. Crie um diretório chamado 'modelos' e outra pasta 'modelo_final'. No ambiente Spaces, arraste e cole a pasta modelos na area de files. 

Importe para o ambiente:
pasta: modelos
arquivos: requirements.txt
          app.py

Estará pronto: README.md
               .gitattributes

Este é o aplicativo pronto. 
https://huggingface.co/spaces/camilaaeromoca/LLM_gradio



title: LLM Gradio

emoji: 🏢

colorFrom: indigo

colorTo: yellow

sdk: gradio

sdk_version: 4.19.2

app_file: app.py

pinned: false
