"""
=============================================================================
  SESCA Core - Funções compartilhadas para predição de espectros de CD
=============================================================================
Módulo reutilizável com as funções de pipeline do SESCA.
Usado tanto pelo CLI (sesca_pipeline.py) quanto pelo Streamlit (sesca_app.py).
"""

import os
import sys
import zipfile
import subprocess
import urllib.request
import csv
from pathlib import Path


# =============================================================================
#  CONFIGURAÇÕES GLOBAIS
# =============================================================================

SESCA_URL = "https://www.mpinat.mpg.de/4787098/SESCA_v097.zip"
SESCA_DIR = Path("SESCA_v097")
SESCA_MAIN = SESCA_DIR / "SESCA_main.py"

DEFAULT_BASIS = "DS-dT"
BASIS_OPTIONS = ["DS-dT", "DS5-4", "DSSP-1", "HBSS-3", "BestSel_der"]

WL_MIN, WL_MAX = 175, 250


# =============================================================================
#  INSTALAÇÃO DO SESCA
# =============================================================================

def download_sesca(force=False, log=print):
    """Baixa e descompacta o SESCA se ainda não estiver presente."""
    zip_path = Path("SESCA_v097.zip")

    if SESCA_DIR.exists() and not force:
        log(f"[OK] SESCA já está em '{SESCA_DIR}'. Pulando download.")
        return True

    log(f"[INFO] Baixando SESCA de:\n       {SESCA_URL}")

    try:
        urllib.request.urlretrieve(SESCA_URL, zip_path)
    except Exception as e:
        log(f"[ERRO] Falha no download: {e}")
        return False

    log("[INFO] Descompactando...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    zip_path.unlink()

    if not SESCA_MAIN.exists():
        log(f"[ERRO] Não encontrei '{SESCA_MAIN}' após descompactar.")
        return False

    log(f"[OK] SESCA instalado em '{SESCA_DIR}'")
    return True


def check_sesca():
    """Verifica se o SESCA está disponível e funcional."""
    if not SESCA_MAIN.exists():
        return False
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
#  DOWNLOAD DE ESTRUTURAS DO RCSB PDB
# =============================================================================

def fetch_pdb(pdb_id: str, output_dir: Path, log=print) -> Path | None:
    """Baixa um arquivo PDB do RCSB pelo código de acesso (ex: 1UBQ)."""
    pdb_id = pdb_id.upper()
    dest = output_dir / f"{pdb_id}.pdb"

    if dest.exists():
        log(f"[OK] {pdb_id}.pdb já existe localmente.")
        return dest

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    log(f"[INFO] Baixando estrutura {pdb_id} do RCSB PDB...")

    try:
        urllib.request.urlretrieve(url, dest)
        log(f"[OK] Salvo em {dest}")
        return dest
    except Exception as e:
        log(f"[ERRO] Não foi possível baixar {pdb_id}: {e}")
        return None


# =============================================================================
#  PRÉ-PROCESSAMENTO DO PDB
# =============================================================================

def clean_pdb(input_pdb: Path, output_dir: Path) -> Path:
    """
    Limpeza básica do PDB:
    - Remove linhas HETATM (ligantes, água)
    - Mantém apenas o primeiro modelo (para NMR multi-modelo)
    - Remove cadeias alternativas (mantém conf. A)
    """
    clean_path = output_dir / f"{input_pdb.stem}_clean.pdb"

    with open(input_pdb) as fh, open(clean_path, "w") as out:
        model_done = False

        for line in fh:
            record = line[:6].strip()

            if record == "MODEL":
                if model_done:
                    break
                continue
            if record == "ENDMDL":
                model_done = True
                continue
            if record == "HETATM":
                continue
            if record == "ATOM":
                alt_loc = line[16]
                if alt_loc not in (" ", "A"):
                    continue

            out.write(line)

    return clean_path


# =============================================================================
#  EXECUÇÃO DO SESCA
# =============================================================================

def run_sesca(pdb_path: Path, output_dir: Path, basis: str = DEFAULT_BASIS, log=print) -> dict | None:
    """
    Executa o SESCA_main.py para um arquivo PDB.
    Retorna dicionário com arrays de comprimento de onda e intensidade CD.
    """
    out_file = output_dir / f"{pdb_path.stem}_CDspectrum.dat"

    cmd = [
        sys.executable, str(SESCA_MAIN),
        "-pdb",   str(pdb_path),
        "-basis", basis,
        "-ofile", str(out_file),
    ]

    log(f"[SESCA] Processando: {pdb_path.name}")
    log(f"        Conjunto de base: {basis}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SESCA_DIR.parent),
        )
    except FileNotFoundError:
        log(f"[ERRO] Não foi possível executar: {cmd[0]}")
        return None

    if result.returncode != 0:
        log(f"[ERRO] SESCA retornou erro (código {result.returncode}):")
        log(result.stderr[-2000:])
        return None

    return parse_sesca_output(out_file, log=log)


def parse_sesca_output(dat_file: Path, log=print) -> dict | None:
    """Lê o arquivo .dat gerado pelo SESCA."""
    if not dat_file.exists():
        log(f"[ERRO] Arquivo de saída não encontrado: {dat_file}")
        return None

    wavelengths, cd_values = [], []

    with open(dat_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    cd_values.append(float(parts[1]))
                except ValueError:
                    continue

    if not wavelengths:
        log(f"[ERRO] Nenhum dado lido de {dat_file}")
        return None

    log(f"[OK] Lidos {len(wavelengths)} pontos espectrais "
        f"({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm)")

    return {
        "file":        str(dat_file),
        "wavelengths": wavelengths,
        "cd_values":   cd_values,
    }


# =============================================================================
#  SALVAMENTO DOS RESULTADOS
# =============================================================================

def save_combined_csv(results: dict, output_dir: Path) -> Path:
    """Salva todos os espectros em um único CSV."""
    csv_path = output_dir / "espectros_CD_combinados.csv"
    all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
    names = list(results.keys())

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Comprimento_de_onda_nm"] + names)
        for wl in all_wl:
            row = [wl]
            for name in names:
                r = results[name]
                try:
                    idx = r["wavelengths"].index(wl)
                    row.append(r["cd_values"][idx])
                except ValueError:
                    row.append("")
            writer.writerow(row)

    return csv_path


def save_summary_txt(results: dict, output_dir: Path) -> Path:
    """Salva resumo numérico dos espectros."""
    summary_path = output_dir / "resumo_espectros.txt"

    with open(summary_path, "w") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("  Resumo dos Espectros de CD Preditos (SESCA)\n")
        fh.write("=" * 60 + "\n\n")

        for name, r in results.items():
            wl = r["wavelengths"]
            cd = r["cd_values"]
            i_min = cd.index(min(cd))
            i_max = cd.index(max(cd))

            fh.write(f"Proteína: {name}\n")
            fh.write(f"  Pontos espectrais : {len(wl)}\n")
            fh.write(f"  Intervalo (nm)    : {wl[0]:.0f} – {wl[-1]:.0f}\n")
            fh.write(f"  Mínimo CD         : {min(cd):.4f}  @ {wl[i_min]:.0f} nm\n")
            fh.write(f"  Máximo CD         : {max(cd):.4f}  @ {wl[i_max]:.0f} nm\n")
            fh.write("\n")

    return summary_path
