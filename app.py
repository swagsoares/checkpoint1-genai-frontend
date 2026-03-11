import os
import json
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import altair as alt
from PIL import Image, UnidentifiedImageError

# ==============================================
# CONFIGURAÇÕES E CAMINHOS
# ==============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'feedback_log.json') # Arquivo para persistência real

st.set_page_config(page_title='VAE Pneumonia - Triagem Avançada', layout='wide')

# ==============================================
# CLASSES E FUNÇÕES DO MODELO (MANTIDAS DO ORIGINAL)
# ==============================================
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)

@st.cache_resource
def load_model():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    latent_dim = int(config.get('latent_dim', 16))
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)
    vae.load_weights(WEIGHTS_PATH)
    return vae, None

def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'L':
        image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28))
    arr = np.array(image).astype('float32')
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

@st.cache_data
def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    return float(np.mean((x - x_recon) ** 2))

@st.cache_data
def classify_pneumonia(reconstruction_error: float, threshold_normal: float, threshold_borderline: float) -> tuple:
    if reconstruction_error < threshold_normal:
        return "NORMAL", "Baixo risco de pneumonia", "green"
    elif reconstruction_error < threshold_borderline:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red"

def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    latent_dim = vae.encoder.output_shape[0][-1]
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = vae.decode(z_samples, training=False).numpy()
    return generated_images

# ==============================================
# FUNÇÕES DE PERSISTÊNCIA (O DIFERENCIAL)
# ==============================================
def load_feedback_log():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback(classification, mse, correct):
    log = load_feedback_log()
    log.append({
        "timestamp": time.time(),
        "classification": classification,
        "mse": mse,
        "correct": correct
    })
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(log, f, indent=4)
    return log

# ==============================================
# INICIALIZAÇÃO DE ESTADO
# ==============================================
if "history" not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "analysis_ran" not in st.session_state: st.session_state.analysis_ran = False
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Execução", "Classificação", "Erro MSE", "Confiança (%)"])

# Carrega o log real do arquivo JSON (Human in the loop persistente)
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = load_feedback_log()

def reset_analysis():
    st.session_state.analysis_ran = False
    st.session_state.last_result = None

# ==============================================
# SIDEBAR
# ==============================================
st.sidebar.header("⚙️ Controle do Modelo")
vae, err = load_model()
if err:
    st.sidebar.error(err)
    st.stop()
else:
    st.sidebar.success("✅ Modelo Operacional")

st.sidebar.markdown("---")
st.sidebar.header("Parâmetros de Inferência")

# Interatividade real com o modelo
st.sidebar.slider("Limiar Normal (MSE)", 0.000, 0.050, 0.010, 0.001, format="%.3f", key="threshold_normal", on_change=reset_analysis)
st.sidebar.slider("Limiar Borderline (MSE)", 0.000, 0.100, 0.020, 0.001, format="%.3f", key="threshold_borderline", on_change=reset_analysis)
st.sidebar.checkbox("Simular latência (UX)", value=True, key="simulate_latency")

if st.sidebar.button("Limpar Cache da Sessão"):
    st.cache_data.clear()
    st.session_state.history = []
    st.session_state.history_df = pd.DataFrame(columns=["Execução", "Classificação", "Erro MSE", "Confiança (%)"])
    st.sidebar.success("Cache e histórico limpos.")

# ==============================================
# ESTRUTURA PRINCIPAL (Fim do st.stop fatal)
# ==============================================
st.title("Sistema de Triagem e Geração Sintética (VAE)")
st.caption("Implementação focada em Confiabilidade e Human-in-the-Loop.")

# As abas agora englobam toda a aplicação. O st.stop() não bloqueia a navegação.
tab_triagem, tab_geracao, tab_dados, tab_monitor = st.tabs([
    "🔍 Inferência & Triagem", 
    "🧬 Geração Sintética",
    "📊 Dados & Histórico", 
    "📈 Monitoramento (HITL)"
])

# --------------------------------------------------------
# ABA 1: TRIAGEM E CONFIABILIDADE
# --------------------------------------------------------
with tab_triagem:
    st.markdown("### Envio de Exame")
    uploaded = st.file_uploader("Faça o upload do Raio-X (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded:
        if st.button("🔍 Iniciar Análise Profunda"):
            st.session_state.analysis_ran = True
            st.session_state.run_file_key = uploaded.name + str(uploaded.size)

        if st.session_state.analysis_ran:
            file_key = st.session_state.get("run_file_key", "")
            
            # BLOCO DE CONFIABILIDADE (Tratamento de Erros)
            try:
                image = Image.open(io.BytesIO(uploaded.read()))
                x = preprocess_image(image)
            except UnidentifiedImageError:
                st.error("🚨 FALHA CRÍTICA: O arquivo enviado está corrompido ou não é uma imagem válida. Análise abortada para proteger o modelo.")
                st.stop() # Para apenas a aba, não o app inteiro
            except Exception as e:
                st.error(f"🚨 Erro inesperado ao processar a imagem: {e}")
                st.stop()

            if st.session_state.get("last_file_key") != file_key and st.session_state.simulate_latency:
                with st.spinner("Extraindo tensores..."): time.sleep(0.4)
                with st.spinner("Processando no espaço latente..."): time.sleep(0.4)
                st.session_state.last_file_key = file_key

            # Inferência
            recon = vae(x, training=False).numpy()
            mse = compute_reconstruction_error(x, recon)
            classification, description, color = classify_pneumonia(mse, st.session_state.threshold_normal, st.session_state.threshold_borderline)
            confidence_percent = max(0, int((1 - mse) * 100)) if mse < 1 else 0

            # Atualiza histórico local da sessão
            if st.session_state.last_result is None or st.session_state.last_result.get("file_key") != file_key:
                st.session_state.last_result = {"x": x, "recon": recon, "mse": mse, "classification": classification, "confidence": confidence_percent, "file_key": file_key}
                new_row = pd.DataFrame([{"Execução": len(st.session_state.history) + 1, "Classificação": classification, "Erro MSE": round(mse, 6), "Confiança (%)": confidence_percent}])
                st.session_state.history_df = pd.concat([st.session_state.history_df, new_row], ignore_index=True)
                st.session_state.history.append({"classification": classification, "mse": mse, "confidence": confidence_percent})

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Imagem de Entrada**")
                st.image(x[0].squeeze(), clamp=True, width=150)
            with col2:
                st.markdown("**Reconstrução do Modelo** (Alto ruído = Anomalia)")
                st.image(recon[0].squeeze(), clamp=True, width=150)

            st.markdown(f"""
            <div style="padding:1rem; border-radius:0.5rem; background-color:{color}20; border-left:4px solid {color}; margin-top:1rem;">
                <h4 style="color:{color}; margin:0;">Diagnóstico Simulado: {classification}</h4>
                <p style="margin:0.5rem 0 0 0;">Erro de Reconstrução (MSE): <strong>{mse:.6f}</strong></p>
                <p style="margin:0;">Nível de Confiança da Análise: <strong>{confidence_percent}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # HUMAN-IN-THE-LOOP (Com persistência real)
            st.markdown("### Validação do Especialista (Human-in-the-Loop)")
            st.caption("Seu feedback retroalimentará o banco de dados do sistema.")
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("✅ Concordo com o Modelo"):
                    st.session_state.feedback_log = save_feedback(classification, mse, True)
                    st.success("Log gravado em banco de dados local (feedback_log.json).")
            with fc2:
                if st.button("❌ Modelo Errou"):
                    st.session_state.feedback_log = save_feedback(classification, mse, False)
                    st.error("Divergência registrada em banco de dados local (feedback_log.json).")
    else:
        st.info("Aguardando upload do exame.")

# --------------------------------------------------------
# ABA 2: GERAÇÃO SINTÉTICA (A funcionalidade que estava faltando)
# --------------------------------------------------------
with tab_geracao:
    st.markdown("### Amostragem do Espaço Latente")
    st.write("Gere novos exemplos de pulmões saudáveis variando as distribuições matemáticas do modelo.")
    
    num_imgs = st.slider("Número de amostras sintéticas:", min_value=1, max_value=8, value=4)
    
    if st.button("🧬 Gerar Novas Imagens"):
        with st.spinner("Sintetizando imagens a partir do ruído gaussiano..."):
            time.sleep(1) # UX de processamento
            generated_images = generate_new_images(vae, num_imgs)
            
            # Exibição dinâmica de colunas
            cols = st.columns(4)
            for i, img in enumerate(generated_images):
                with cols[i % 4]:
                     st.image(img.squeeze(), clamp=True, use_column_width=True, caption=f"Amostra {i+1}")
# --------------------------------------------------------
# ABA 3: DADOS E HISTÓRICO
# --------------------------------------------------------
with tab_dados:
    st.markdown("### Histórico da Sessão Atual")
    if not st.session_state.history_df.empty:
        st.dataframe(
            st.session_state.history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confiança (%)": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=100, format="%d%%"),
                "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
            },
        )
    else:
        st.info("Nenhuma análise executada nesta sessão.")

# --------------------------------------------------------
# ABA 4: MONITORAMENTO (Estatísticas do HITL)
# --------------------------------------------------------
with tab_monitor:
    st.markdown("### Desempenho Validado (Geral)")
    st.caption("Esses dados puxam do arquivo de persistência e não somem se a página recarregar.")
    
    log = st.session_state.feedback_log
    total_fb = len(log)
    if total_fb > 0:
        correct = sum(1 for f in log if f["correct"])
        accuracy = correct / total_fb
        
        mon1, mon2, mon3 = st.columns(3)
        mon1.metric("Total de Feedbacks", total_fb)
        mon2.metric("Acertos do Modelo", correct)
        mon3.metric("Acurácia Real", f"{int(accuracy * 100)}%")
        
        # Opcional: mostrar o log bruto
        with st.expander("Ver log de validação bruto (JSON)"):
            st.json(log)
    else:
        st.info("Nenhum feedback registrado ainda.")