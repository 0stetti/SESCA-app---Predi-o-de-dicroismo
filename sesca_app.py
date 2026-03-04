#!/usr/bin/env python3
"""
=============================================================================
  SESCA Web App - Predição de Espectros de Dicroísmo Circular (CD)
=============================================================================
Interface Streamlit moderna para o pipeline SESCA.

Executar com:
  streamlit run sesca_app.py
"""

import tempfile
import io
from pathlib import Path

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from sesca_core import (
    download_sesca, check_sesca, fetch_pdb, clean_pdb,
    run_sesca, save_combined_csv,
    DEFAULT_BASIS, BASIS_OPTIONS, SESCA_DIR,
)

# =============================================================================
#  CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="SESCA — CD Spectrum Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  CSS CUSTOMIZADO
# =============================================================================

st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .main-header h1 {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }

    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a1a2e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-card .unit {
        color: #64748b;
        font-size: 0.85rem;
    }

    /* Status badge */
    .status-ok {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-err {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Sidebar estilo */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0;
    }

    /* Esconder menu e footer padrão */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configurações")
    st.markdown("---")

    # Status do SESCA
    sesca_ready = check_sesca()
    if sesca_ready:
        st.markdown('<span class="status-ok">● SESCA pronto</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">● SESCA não instalado</span>', unsafe_allow_html=True)
        if st.button("Instalar SESCA", use_container_width=True):
            with st.spinner("Baixando SESCA (~5 MB)..."):
                ok = download_sesca(log=lambda msg: st.text(msg))
            if ok:
                st.rerun()
            else:
                st.error("Falha ao instalar. Verifique sua conexão.")

    st.markdown("---")

    # Basis set
    basis = st.selectbox(
        "Conjunto de base espectral",
        options=BASIS_OPTIONS,
        index=BASIS_OPTIONS.index(DEFAULT_BASIS),
        help="DS-dT é recomendado para proteínas globulares. "
             "DS5-4 usa 5 componentes. DSSP-1 e HBSS-3 usam atribuições DSSP.",
    )

    # Limpar PDB
    clean = st.toggle(
        "Limpar PDB antes de processar",
        value=True,
        help="Remove HETATM (água, ligantes), conformações alternativas, "
             "e mantém apenas o primeiro modelo NMR.",
    )

    st.markdown("---")

    # Referência
    st.markdown(
        "<small style='color: #64748b;'>"
        "<b>Referência:</b><br>"
        "Nagy et al., J. Chem. Theory Comput. 15, 5087-5102 (2019)<br>"
        "<a href='https://doi.org/10.1021/acs.jctc.9b00203' style='color: #818cf8;'>"
        "doi: 10.1021/acs.jctc.9b00203</a>"
        "</small>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🧬 SESCA — CD Spectrum Predictor</h1>
    <p>Predição de espectros de Dicroísmo Circular a partir de estruturas proteicas (PDB)</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  INPUT DE ESTRUTURAS
# =============================================================================

tab_upload, tab_rcsb = st.tabs(["📁 Upload de PDB", "🌐 Buscar no RCSB PDB"])

uploaded_files = []
pdb_ids = []

with tab_upload:
    files = st.file_uploader(
        "Arraste seus arquivos PDB aqui",
        type=["pdb"],
        accept_multiple_files=True,
        help="Aceita um ou mais arquivos .pdb",
    )
    if files:
        uploaded_files = files

with tab_rcsb:
    col1, col2 = st.columns([3, 1])
    with col1:
        pdb_input = st.text_input(
            "Códigos PDB (separados por espaço ou vírgula)",
            placeholder="ex: 1UBQ 2GB1 1L2Y",
            help="Digite os códigos de acesso do RCSB PDB",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("Buscar", use_container_width=True, type="secondary")

    if pdb_input:
        pdb_ids = [x.strip().upper() for x in pdb_input.replace(",", " ").split() if x.strip()]
        if pdb_ids:
            st.info(f"Estruturas selecionadas: **{', '.join(pdb_ids)}**")


# =============================================================================
#  EXECUÇÃO
# =============================================================================

has_input = bool(uploaded_files) or bool(pdb_ids)

run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_btn = st.button(
        "🚀 Executar Predição",
        use_container_width=True,
        type="primary",
        disabled=not (has_input and sesca_ready),
    )

if not sesca_ready and has_input:
    st.warning("Instale o SESCA primeiro usando o botão na barra lateral.")

if run_btn and has_input and sesca_ready:
    st.markdown("---")

    results = {}
    logs = []

    def log_msg(msg):
        logs.append(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_dir = tmpdir / "pdb_inputs"
        pdb_dir.mkdir()

        progress = st.progress(0, text="Preparando...")

        # Reúne todos os PDBs
        all_pdbs = []

        # Arquivos enviados por upload
        for uf in uploaded_files:
            dest = pdb_dir / uf.name
            dest.write_bytes(uf.read())
            all_pdbs.append(dest)

        # PDBs baixados do RCSB
        for i, pid in enumerate(pdb_ids):
            progress.progress(
                int(10 + 20 * i / max(len(pdb_ids), 1)),
                text=f"Baixando {pid}...",
            )
            path = fetch_pdb(pid, pdb_dir, log=log_msg)
            if path:
                all_pdbs.append(path)

        if not all_pdbs:
            st.error("Nenhum arquivo PDB válido para processar.")
            st.stop()

        # Processa cada PDB
        total = len(all_pdbs)
        for i, pdb_path in enumerate(all_pdbs):
            pct = int(30 + 60 * i / total)
            progress.progress(pct, text=f"Processando {pdb_path.stem}...")

            pdb_to_use = clean_pdb(pdb_path, tmpdir) if clean else pdb_path
            result = run_sesca(pdb_to_use, tmpdir, basis=basis, log=log_msg)

            if result:
                results[pdb_path.stem] = result

        progress.progress(95, text="Finalizando...")

        # Salva CSV combinado
        if results:
            csv_path = save_combined_csv(results, tmpdir)
            csv_data = csv_path.read_text()

        progress.progress(100, text="Concluído!")

    # ── Exibe logs ────────────────────────────────────────────────────────
    with st.expander("📋 Log de execução", expanded=False):
        for line in logs:
            st.text(line)

    if not results:
        st.error("Nenhum espectro foi gerado. Verifique os logs acima.")
        st.stop()

    # ── Armazena em session_state para persistir ──────────────────────────
    st.session_state["results"] = results
    st.session_state["csv_data"] = csv_data


# =============================================================================
#  RESULTADOS
# =============================================================================

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    csv_data = st.session_state.get("csv_data", "")

    st.markdown("---")
    st.markdown("## 📊 Resultados")

    # ── Métricas resumo ───────────────────────────────────────────────────
    cols = st.columns(len(results))
    palette = ["#818cf8", "#34d399", "#f472b6", "#fbbf24", "#60a5fa", "#a78bfa", "#fb923c"]

    for i, (name, r) in enumerate(results.items()):
        cd = r["cd_values"]
        wl = r["wavelengths"]
        i_min = cd.index(min(cd))
        i_max = cd.index(max(cd))
        color = palette[i % len(palette)]

        with cols[i]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label" style="color: {color};">{name}</div>'
                f'<div class="value">{min(cd):.2f}</div>'
                f'<div class="unit">Mín. CD @ {wl[i_min]:.0f} nm</div>'
                f'<br>'
                f'<div class="value">{max(cd):.2f}</div>'
                f'<div class="unit">Máx. CD @ {wl[i_max]:.0f} nm</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico interativo (Plotly) ───────────────────────────────────────
    fig = go.Figure()

    for i, (name, r) in enumerate(results.items()):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=r["wavelengths"],
            y=r["cd_values"],
            name=name,
            mode="lines",
            line=dict(color=color, width=2.5),
            hovertemplate="<b>%{fullData.name}</b><br>"
                          "λ = %{x:.1f} nm<br>"
                          "CD = %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.4)")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        title=dict(
            text="Espectros de Dicroísmo Circular Preditos",
            font=dict(size=18, color="#e2e8f0"),
        ),
        xaxis=dict(
            title="Comprimento de onda (nm)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            dtick=10,
        ),
        yaxis=dict(
            title="CD (Δε)",
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(99, 102, 241, 0.3)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#818cf8",
            font=dict(color="#e2e8f0"),
        ),
        height=500,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Tabela de dados ───────────────────────────────────────────────────
    with st.expander("📋 Tabela de dados", expanded=False):
        all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
        df_data = {"λ (nm)": all_wl}
        for name, r in results.items():
            wl_map = dict(zip(r["wavelengths"], r["cd_values"]))
            df_data[name] = [wl_map.get(wl, None) for wl in all_wl]
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Downloads ─────────────────────────────────────────────────────────
    st.markdown("### 📥 Downloads")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.download_button(
            "⬇️ CSV Combinado",
            data=csv_data,
            file_name="espectros_CD_combinados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        # Exportar gráfico como HTML interativo
        html_buf = io.StringIO()
        fig.write_html(html_buf, include_plotlyjs="cdn")
        st.download_button(
            "⬇️ Gráfico Interativo (HTML)",
            data=html_buf.getvalue(),
            file_name="espectro_CD_interativo.html",
            mime="text/html",
            use_container_width=True,
        )

    with dl_col3:
        # Exportar gráfico como PNG
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            st.download_button(
                "⬇️ Gráfico (PNG)",
                data=png_bytes,
                file_name="espectro_CD.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.caption("PNG requer `kaleido`: `pip install kaleido`")

else:
    # Estado vazio
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; padding: 3rem; color: #64748b;'>"
        "<p style='font-size: 3rem; margin-bottom: 0.5rem;'>🧬</p>"
        "<p style='font-size: 1.1rem;'>Envie um arquivo PDB ou busque pelo código RCSB para começar</p>"
        "</div>",
        unsafe_allow_html=True,
    )
