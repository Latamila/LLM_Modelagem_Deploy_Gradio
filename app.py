import gradio as gr
import torch
import transformers
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Tokenizer
from transformers import GPT2Tokenizer

modelo_llm = AutoModelForCausalLM.from_pretrained("modelos/modelo_final")

# Definindo uma classe chamada NumberTokenizer, que é usada para tokenizar os números
class DSATokenizer:
    
    # Método construtor da classe, que é executado quando um objeto dessa classe é criado
    def __init__(self, numbers_qty = 10):
        
        # Lista de tokens possíveis que o tokenizador pode encontrar
        vocab = ['+', '=', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Definindo a quantidade de números que o tokenizador pode lidar
        self.numbers_qty = numbers_qty
        
        # Definindo o token de preenchimento (padding)
        self.pad_token = '-1'
        
        # Criando um dicionário que mapeia cada token para um índice único
        self.encoder = {str(v):i for i,v in enumerate(vocab)}
        
        # Criando um dicionário que mapeia cada índice único de volta ao token correspondente
        self.decoder = {i:str(v) for i,v in enumerate(vocab)}
        
        # Obtendo o índice do token de preenchimento no encoder
        self.pad_token_id = self.encoder[self.pad_token]

    # Método para decodificar uma lista de IDs de token de volta para uma string
    def decode(self, token_ids):
        return ' '.join(self.decoder[t] for t in token_ids)

    # Método que é chamado quando o objeto da classe é invocado como uma função
    def __call__(self, text):
        # Dividindo o texto em tokens individuais e retornando uma lista dos IDs correspondentes
        return [self.encoder[t] for t in text.split()]

# Cria o objeto
tokenizer = DSATokenizer(13)

# Definindo a função gera_solution com três parâmetros: input, solution_length e model
def faz_previsao(entrada, solution_length = 6, model = modelo_llm):

    # Colocando o modelo em modo de avaliação. 
    model.eval()

    # Convertendo a entrada (string) em tensor utilizando o tokenizer. 
    # O tensor é uma estrutura de dados que o modelo de aprendizado de máquina pode processar.
    entrada = torch.tensor(tokenizer(entrada))

    # Iniciando uma lista vazia para armazenar a solução
    solution = []

    # Loop que gera a solução de comprimento solution_length
    for i in range(solution_length):

        # Alimentando a entrada atual ao modelo e obtendo a saída
        saida = model(entrada)

        # Pegando o índice do maior valor no último conjunto de logits (log-odds) da saída, 
        # que é a previsão do modelo para o próximo token
        predicted = saida.logits[-1].argmax()

        # Concatenando a previsão atual com a entrada atual. 
        # Isso servirá como a nova entrada para a próxima iteração.
        entrada = torch.cat((entrada, predicted.unsqueeze(0)), dim = 0)

        # Adicionando a previsão atual à lista de soluções e convertendo o tensor em um número Python padrão
        solution.append(predicted.cpu().item())

    # Decodificando a lista de soluções para obter a string de saída e retornando-a
    return tokenizer.decode(solution)

# Testa a função
faz_previsao('3 + 5 =', solution_length = 2)

# Função para retornar a função que faz a previsão
def funcsolve(entrada):
    return faz_previsao(entrada, solution_length = 2)


# Cria a web app
webapp = gr.Interface(fn = funcsolve, 
                      inputs = [gr.Textbox(label = "Dados de Entrada", 
                                           lines = 1, 
                                           info = "Os dados devem estar na forma: '1 + 2 =' com um único espaço entre cada caractere e apenas números de um dígito são permitidos.")],
                      outputs = [gr.Textbox(label = "Resultado (Previsão do Modelo)", lines = 1)],
                      title = "Deploy de LLM Após o Fine-Tuning",
                      description = "Digite os dados de entrada e clique no botão Submit para o modelo fazer a previsão.",
                      examples = ["5 + 3 =", "2 + 9 ="]) 


webapp.launch()
