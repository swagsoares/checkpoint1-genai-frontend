# CheckPoint 1 - Generative AI Advanced Net & Front End e Mobile Development

## Situação Problema

Utilizar  o dataset PneumoniaMNIST para treinar uma variational autoendoder para verificar câncer de pulmão. Montar uma interface em StreamLit para mostrar o resultado do treinamento e as métricas de performance

https://github.com/erickfog/GENERATIVE_AI_ADVANCED_NETS/tree/main/Aula03

## Objetivo

- Instalar e executar o modelo;
- Utilizar o app.py como ponto de partida para entender a aplicação do modelo.
- A partir do app.py você deverá modificá-lo (ou criar um novo sobre um template que você tenha familiaridade), a fim de implementar os conceitos de design com recursos do streamlit aplicados durante as aulas.

## Ponto de partida
Dentro do app.py existe o trecho

```python

st.set_page_config(page_title='VAE PneumoniaMNIST - Triagem e Geração', layout='wide')
st.title('VAE PneumoniaMNIST - Triagem de Pneumonia e Geração de Imagens')

```

- Tudo que vem antes disso são funções de execução do modelo.
- Tudo que vem depois disso é a construção e interação da interface.
- Para aplicar alguns conceitos do design, pode ser necessário modificar parâmetros de entrada das funções, ou retornos esperados.

## Resultados Esperados por tema

# 1. Organização da Informação (Arquitetura da Interface)

### OK

-   Uso de `st.tabs` e `st.columns` para separar input/output
-   Título e subtítulos claros com title, header, subheader ou markdown
-   Separação mínima entre Entrada e Saída

### Bom

-   Uso correto de `st.sidebar` para configurações globais
-   Separação clara entre:
    -   Configuração
    -   Execução
    -   Resultado
    -   Monitoramento
-   Hierarquia visual consistente (subheaders organizados)

### Ótimo

-   Sidebar usada como **painel de controle do sistema**
-   Área principal usada apenas para decisão e resultado
-   Tabs usadas para separar **contextos**, não etapas
-   Empty State bem definido (orientação clara antes da ação)
-   Layout previsível e consistente (design system aplicado)

------------------------------------------------------------------------

# 2. Input de Dados (Interatividade Real)

### OK

-   Uso de widgets/inputs de configuração (`slider`, `selectbox`, `text_input`, etc.)
-   Inputs funcionando corretamente

### Bom

-   Inputs ligados ao estado via `key`
-   Parâmetros influenciam de fato a execução do modelo
-   Separação entre:
    -   Configuração (widgets)
    -   Ação (botão)

### Ótimo

-   Uso de `on_change` para resetar resultados
-   Reset automático da interface de saída da análise ao alterar parâmetros

------------------------------------------------------------------------

# 3. Design para Latência

### OK

-   Uso de `st.spinner`

### Bom

-   Uso de `st.progress`
-   Simulação ou tratamento de múltiplas etapas

### Ótimo

-   Uso combinado de:
    -   `st.spinner`
    -   `st.progress`
    -   `st.status`
-   Comunicação clara do pipeline ("Extraindo...", "Classificando...")
-   Execução controlada por botão (evita reruns desnecessários)

------------------------------------------------------------------------

# 4. Confidence UI (Gestão de Incerteza)

### OK

-   Exibição de percentual de confiança
-   Uso de `st.metric`

### Bom

-   Barra de progresso proporcional à confiança
-   Diferenciação textual entre alta, média e baixa confiança

### Ótimo

-   Uso semântico de cores:
    -   `st.success`
    -   `st.warning`
    -   `st.error`
-   Orientação comportamental:
    -   Recomendar revisão quando confiança baixa
-   Comunicação clara de que é uma **estimativa**
 
------------------------------------------------------------------------

# 5. Human-in-the-loop

### OK

-   Botão de feedback (acertou/errou). Tudo bem se você, como usuário, de fato não souber se aquela resposta está certa ou não, o importante é ter a possibilidade de indicar o acerto/erro.

### Bom

-   Registro do feedback em lista ou variável de sessão
-   Exibição de confirmação visual após feedback, com st.status ou st.toast, por exemplo.
-   DataFrame ou tabela com histórico de feedback   

### Ótimo

-   DataFrame com columnConfig para mostrar uma barra de progresso sobre a confiabilidade, ou a imagem que foi analisada naquele histórico.
-   Alerta de possível degradação do modelo. Muitos feedbacks negativos, ou várias métricas de confiabilidade baixa.
-   Gráfico mostrando a evolução da confiabildiade sobre as interações.

------------------------------------------------------------------------

# 7. Estado, Callbacks e Persistência

### OK

-   Uso básico de `st.session_state`
-   Inicialização correta de variáveis
-   Tratamento de empty state (`st.stop()` quando necessário)

### Bom

-   Histórico persistente entre execuções
-   Controle de execução via variável (`analysis_ran`)
-   Separação entre botão (ação) e estado

### Ótimo

-   Callback para resetar análise
-   Controle arquitetural claro:
    -   Estado controla UI | UI reflete estado -> Empty state e só depois da interação mostrar resultados.
-   Nenhum comportamento inconsistente causado por re-run. Por exemplo, interface perder o resultado porque o usuário clicou em um feedback positivo ou negativo.

 
------------------------------------------------------------------------

# 8. Monitoramento e Histórico

### OK

-   Histórico simples de execuções, usando st.write e st.columns

### Bom

-   Exibição organizada do histórico (colunas ou lista estruturada) com st.dataframe ou st.table

### Ótimo

-   Métricas agregadas ao histórico.
-   Alerta de possível degradação do modelo. Muitos feedbacks negativos, ou várias métricas de confiabilidade baixa.
-   Separação entre histórico operacional e feedback humano
 
------------------------------------------------------------------------

# 11. Cache e Performance

### OK

-   Separação da função de modelo

### Bom

-   Uso de `@st.cache_data` ou `@st.cache_resource`

### Ótimo

-   Cache aplicado corretamente a:
    -   Carregamento pesado
    -   Dados
    -   Recursos
-   Nenhuma reexecução desnecessária
-   Arquitetura pensada para evitar bloqueios no topo do script


