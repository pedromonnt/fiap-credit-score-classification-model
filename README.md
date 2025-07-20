# Trabalho Final - FIAP - 10DTSR - MLOPS

Ana Cristina Lourenço Maria: RM359310

Jayana da Silva Alves: RM359631

Pedro Silva de Sá Monnerat: RM359532

# Classificação de Score de Crédito

Este repositório contém o código-fonte e os workflows para o desenvolvimento, registro e monitoramento de um modelo de classificação de score de crédito. Ele engloba desde o processamento de dados até o treinamento de modelos, testes, geração de relatórios e integração com um pipeline de deployment da API.

# Estrutura do Projeto

- .github/workflows/model_report.yml: Workflow para gerar relatórios de comparação de modelos.

- .github/workflows/model_notify_api_deploy.yml: Workflow para notificar o repositório da API para um novo deployment após o registro de um modelo.

- notebooks/data-processing.ipynb: Notebook para o pré-processamento e limpeza dos dados.

- notebooks/model-development.ipynb: Notebook para o treinamento e experimentação de diferentes modelos de Machine Learning.

- models/register_model.py: Script para registrar a versão mais recente do modelo no MLflow, comparando-a com a versão em produção.

- reports/report.py: Script para gerar um relatório de comparação de métricas entre o modelo mais recente e o modelo em produção no MLflow.

- tests/model_test.py: Testes unitários para validar o carregamento do modelo e a predição com dados de exemplo.

- requirements.txt: Lista todas as dependências Python necessárias para o projeto.

