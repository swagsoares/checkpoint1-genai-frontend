Participantes:

Arthur batista RM
Joao Pedro RM561738
Nelson Felix RM565603
Pietro Boroto RM562407
Vitor Gonçalves RM566181

# Sistema de Triagem e Geração Sintética (VAE)
*Projeto focado em Confiabilidade (Reliability) e Validação Humana Persistente (Human-in-the-Loop).*

## ⚠️ Diferenciais da Implementação e Correção de Arquitetura
A interface original fornecida como base continha falhas estruturais críticas que impediam o uso em um cenário real. Este projeto reestruturou o fluxo da aplicação para garantir os seguintes requisitos:

1. **Confiabilidade Ativa (Error Handling):** Implementação de blocos `try/except` na ingestão de imagens. O sistema original quebrava com arquivos corrompidos ou formatos inválidos (gerando *Tracebacks* críticos). Agora, erros são interceptados e alertas amigáveis (`st.error`) protegem a execução do modelo e o estado do servidor.
2. **Human-in-the-Loop (HITL) com Persistência Real:** O feedback do especialista não se perde no `session_state` volátil da interface. Foi criado um sistema de I/O que grava e lê as validações em um banco local (`feedback_log.json`). A aba de Monitoramento consome esse arquivo, tornando a auditoria de acurácia do modelo à prova de recarregamentos de página.
3. **Gestão de Estado Explícita e Proteção de Inferência:** Sliders de limiares foram isolados com *callbacks* (`on_change`) que resetam o estado de análise. Isso obriga o usuário a re-executar a ação explicitamente, impedindo que o Streamlit bombardeie o modelo com requisições de inferência acidentais a cada milímetro arrastado no slider.
4. **Geração Sintética Dinâmica Habilitada:** A função de amostragem do espaço latente, antes inacessível na interface, foi mapeada para uma nova aba. O usuário agora interage com argumentos para gerar matrizes dinâmicas de imagens sintéticas de raio-X de pulmões normais.
5. **Correção de UX Blocking:** O `st.stop()` global que travava toda a navegação do sistema caso nenhuma imagem fosse enviada foi isolado. Agora o usuário pode explorar abas de histórico, monitoramento e geração sem ser forçado a fazer um upload inicial.

---

## ⚙️ Como Executar o Projeto Localmente

**1. Clone o repositório:**
```bash
git clone [https://github.com/swagsoares/checkpoint1-genai-frontend.git](https://github.com/swagsoares/checkpoint1-genai-frontend.git)
cd checkpoint1-genai-frontend

2. Crie e ative o ambiente virtual (Obrigatório para isolar o TensorFlow):
Bash

python -m venv .venv

# No Windows:
.venv\Scripts\activate

# No Mac/Linux:
source .venv/bin/activate

3. Instale as dependências:
Bash

pip install -r requirements.txt

4. Inicie a aplicação web:
Bash

streamlit run app.py

🧪 Roteiro de Avaliação (Casos de Uso)

Para auditar o cumprimento de todos os critérios solicitados no escopo do projeto, execute o seguinte fluxo:

    Teste de Tratamento de Erro: Faça o upload de um arquivo PDF ou de texto com a extensão alterada para .jpg. O sistema interceptará o erro e exibirá uma tarja vermelha de falha crítica sem quebrar o Streamlit.

    Interação com Parâmetros do Modelo: Faça o upload de uma imagem válida de raio-X. Após a primeira análise, utilize a barra lateral esquerda para alterar o "Limiar Borderline". O sistema exigirá um novo clique em "Iniciar Análise", atualizando a classificação e o erro de reconstrução instantaneamente.

    Persistência HITL: Na aba "Inferência & Triagem", valide o resultado clicando em "Concordo com o Modelo" ou "Modelo Errou". Em seguida, navegue até a aba "Monitoramento (HITL)" e verifique que seu feedback alterou as métricas de acurácia global com base no arquivo .json gerado localmente.

    Interação de Geração Dinâmica: Acesse a aba "Geração Sintética". Mude para o número desejado de amostras e inicie a geração. O Decoder do VAE renderizará imagens inéditas a partir do ruído gaussiano.