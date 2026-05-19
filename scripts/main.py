import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Config page
st.set_page_config(
    page_title="NuScenes Surrogate Model POC",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F, #4B8FFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #88888b;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .rule-box {
        background-color: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #4B8FFF;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-family: 'Courier New', Courier, monospace;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-title">🚗 XAI & Surrogate Modeling</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Proof of Concept d\'explicabilité du comportement de conduite autonome sur le dataset NuScenes</div>', unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    csv_path = Path('data/extracted_dataset.csv')
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

df = load_data()

if df is None:
    st.error("❌ Dataset introuvable !")
    st.stop()

# Prepare features & models
features = ['velocity', 'nearest_obstacle_distance', 'obstacles_within_10m', 'lane_curvature', 'distance_along_lane', 'distance_to_lane_end']

X = df[features]
y_steer = df['target_heading_rate']
y_acc = df['target_acceleration']

# Split
X_train, X_test, y_train_steer, y_test_steer = train_test_split(X, y_steer, test_size=0.2, random_state=42)
_, _, y_train_acc, y_test_acc = train_test_split(X, y_acc, test_size=0.2, random_state=42)

# Train surrogate models
@st.cache_resource
def train_models():
    model_steer = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
    model_steer.fit(X_train, y_train_steer)
    
    model_acc = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
    model_acc.fit(X_train, y_train_acc)
    
    return model_steer, model_acc

model_steer, model_acc = train_models()

# Predictions for evaluation metrics
y_pred_steer = model_steer.predict(X_test)
r2_steer = model_steer.score(X_test, y_test_steer) # FIXED NameError HERE
mae_steer = mean_absolute_error(y_test_steer, y_pred_steer)

y_pred_acc = model_acc.predict(X_test)
r2_acc = model_acc.score(X_test, y_test_acc)
mae_acc = mean_absolute_error(y_test_acc, y_pred_acc)

# Helper function to trace rules
def trace_decision_path(tree, feature_names, sample_df):
    node_indicator = tree.decision_path(sample_df)
    leaf_id = tree.apply(sample_df)[0]
    
    left_childs = tree.tree_.children_left
    right_childs = tree.tree_.children_right
    features_idx = tree.tree_.feature
    thresholds = tree.tree_.threshold
    
    path_nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    rules = []
    for node_id in path_nodes:
        if left_childs[node_id] == right_childs[node_id]:
            val = tree.tree_.value[node_id][0][0]
            rules.append(f"➡️ **Prediction finale** : `{val:.3f}`")
            continue
            
        f_idx = features_idx[node_id]
        f_name = feature_names[f_idx]
        thresh = thresholds[node_id]
        curr_val = sample_df.iloc[0][f_name]
        
        if curr_val <= thresh:
            sign = "<="
        else:
            sign = ">"
            
        rules.append(f"🟢 {f_name} (`{curr_val:.2f}`) {sign} `{thresh:.2f}`")
        
    return rules

# Sidebar for simulated controls
st.sidebar.image("https://www.nuscenes.org/public/images/nuscenes-logo.png", use_column_width=True)
st.sidebar.header("🕹️ Simulateur d'Environnement")

sim_vel = st.sidebar.slider("Vitesse [m/s]", 0.0, 25.0, float(df['velocity'].mean()), step=0.1)
sim_dist = st.sidebar.slider("Obstacle proche [m]", 0.0, 50.0, float(df['nearest_obstacle_distance'].mean()), step=0.5)
sim_density = st.sidebar.slider("Obstacles dans un rayon de 10m", 0, 10, int(df['obstacles_within_10m'].mean()))
sim_curve = st.sidebar.slider("Courbure de la voie", 0.0, 0.5, float(df['lane_curvature'].mean()), step=0.01)
sim_dist_along = st.sidebar.slider("Distance parcourue sur voie [m]", 0.0, 100.0, float(df['distance_along_lane'].mean()), step=1.0)
sim_dist_end = st.sidebar.slider("Distance fin de voie [m]", 0.0, 100.0, float(df['distance_to_lane_end'].mean()), step=1.0)

sim_data = pd.DataFrame([{
    'velocity': sim_vel,
    'nearest_obstacle_distance': sim_dist,
    'obstacles_within_10m': sim_density,
    'lane_curvature': sim_curve,
    'distance_along_lane': sim_dist_along,
    'distance_to_lane_end': sim_dist_end
}])


# Tabs definition
tab1, tab2 = st.tabs(["🌎 1. Le Dataset NuScenes", "🧠 2. Surrogate Model & Explicabilité"])

# ==========================================
# TAB 1: Explication et visualisation
# ==========================================
with tab1:
    st.header("Le Dataset NuScenes")
    
    col_img, col_desc = st.columns([1, 2])
    with col_img:
        st.image("https://www.nuscenes.org/public/tutorials/trajectory.gif", caption="Prediction Challenge - Trajectoires")
    with col_desc:
        st.markdown("""
        **nuScenes** est un dataset public de conduite autonome à grande échelle. 
        Pour ce projet, nous utilisons le **nuScenes mini-split** qui nous permet d'étudier le comportement du véhicule autonome (accélération, freinage, braquage) face à son environnement géométrique et dynamique.
        """)
        
    st.markdown("### 📊 Statistiques Clés (Global vs Mini)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="kpi-card"><div class="kpi-value">1000</div><div class="kpi-label">Scènes au total</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="kpi-card"><div class="kpi-value">10</div><div class="kpi-label">Scènes (Mini-split)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="kpi-card"><div class="kpi-value" style="color:#4B8FFF">2</div><div class="kpi-label">Villes (Boston & Singapour)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color:#10B981">{len(df)}</div><div class="kpi-label">Échantillons extraits</div></div>', unsafe_allow_html=True)
        
    st.markdown("### 🗺️ Contexte & Variables Extraites")
    col_vars, col_plot = st.columns([1.5, 1])
    
    with col_vars:
        st.markdown("""
        Nous avons extrait **6 features comportementales et géographiques** :
        * `velocity` : La vitesse du véhicule autonome.
        * `nearest_obstacle_distance` : Distance au premier véhicule ou piéton détecté.
        * `obstacles_within_10m` : La densité d'obstacles très proches (trafic).
        * `lane_curvature` : La courbure de la route calculée via la Map API.
        * `distance_along_lane` & `distance_to_lane_end` : La progression de la voiture sur sa voie de circulation.
        """)
        
    with col_plot:
        fig_dist = px.histogram(
            df,
            x='velocity',
            nbins=30,
            title="Distribution des Vitesses",
            color_discrete_sequence=['#FF4B4B']
        )
        fig_dist.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 2: Visualisation Dynamique & Surrogate
# ==========================================
with tab2:
    st.header("Modélisation & Explicabilité Dynamique")
    st.markdown("Cette section expose le comportement de l'IA (Surrogate Model) selon les variables environnementales.")

    # 1. Metrics R2 & MAE
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### 🔄 Modèle de Braquage (Heading Rate)")
        cc1, cc2 = st.columns(2)
        cc1.metric("R² Score", f"{r2_steer:.2f}")
        cc2.metric("MAE", f"{mae_steer:.3f} rad/s")
        
    with col_m2:
        st.markdown("### 🚀 Modèle d'Accélération (Acc/Brake)")
        cc3, cc4 = st.columns(2)
        cc3.metric("R² Score", f"{r2_acc:.2f}")
        cc4.metric("MAE", f"{mae_acc:.3f} m/s²")
        
    st.divider()

    # 2. Simulateur Dynamique
    st.subheader("🎮 Simulation Interactive")
    st.write("Modifiez les variables dans le panneau de gauche (vitesse, distance obstacle...) pour voir la réaction instantanée du véhicule autonome.")
    
    pred_steer = model_steer.predict(sim_data)[0]
    pred_acc = model_acc.predict(sim_data)[0]
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        # Gauge Braquage
        fig_gauge_steer = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_steer,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Braquage [rad/s]"},
            gauge = {
                'axis': {'range': [-0.5, 0.5]},
                'bar': {'color': "#4B8FFF"},
                'steps': [
                    {'range': [-0.5, -0.15], 'color': "rgba(75, 143, 255, 0.3)"},
                    {'range': [-0.15, 0.15], 'color': "rgba(255, 255, 255, 0.1)"},
                    {'range': [0.15, 0.5], 'color': "rgba(75, 143, 255, 0.3)"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_steer}
            }
        ))
        fig_gauge_steer.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge_steer, use_container_width=True)
        
        st.markdown("#### Pourquoi cette décision ? (Chemin de l'Arbre)")
        path_steer = trace_decision_path(model_steer, features, sim_data)
        for rule in path_steer:
            st.markdown(f"<div class='rule-box'>{rule}</div>", unsafe_allow_html=True)
            
    with col_g2:
        # Gauge Acceleration
        fig_gauge_acc = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_acc,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Accélération [m/s²]"},
            gauge = {
                'axis': {'range': [-8.0, 4.0]},
                'bar': {'color': "#FF4B4B"},
                'steps': [
                    {'range': [-8.0, -2.0], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [-2.0, 0.5], 'color': "rgba(255, 255, 255, 0.1)"},
                    {'range': [0.5, 4.0], 'color': "rgba(16, 185, 129, 0.2)"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_acc}
            }
        ))
        fig_gauge_acc.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge_acc, use_container_width=True)
        
        st.markdown("#### Pourquoi cette décision ? (Chemin de l'Arbre)")
        path_acc = trace_decision_path(model_acc, features, sim_data)
        for rule in path_acc:
            st.markdown(f"<div class='rule-box'>{rule}</div>", unsafe_allow_html=True)
