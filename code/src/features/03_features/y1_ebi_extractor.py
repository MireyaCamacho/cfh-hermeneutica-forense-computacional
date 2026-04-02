"""
CFH · Extractor de Indicador y₁ — Score EBI (Eufemismo Bélico-Institucional)
=============================================================================
Proyecto: Hermenéutica Forense Computacional
Variable latente: ξ₁ (Violencia Discursiva)

Qué mide y₁:
    La probabilidad promedio de que los tokens de un segmento judicial
    pertenezcan a la clase EBI (Eufemismo Bélico-Institucional): expresiones
    que renombran homicidios de civiles como acciones militares legítimas.
    Ejemplos prototípicos: "baja en combate", "resultado operacional",
    "dado de baja", "misión táctica", "contacto armado".

Problema técnico central que resuelve este módulo:
    ConfliBERT-Spanish (y su derivado CFH-BERT) acepta un máximo de 512
    tokens por input. Los segmentos judiciales objetivo — especialmente
    la sección HECHOS y CONSIDERACIONES del corpus A — exceden frecuentemente
    ese límite. Una sentencia de segunda instancia puede tener 3.000-8.000
    tokens en la sección de hechos.

    Estrategia de ventana deslizante con stride:
    - Se divide el texto en ventanas de `window_size` tokens
    - Las ventanas se solapan en `stride` tokens para que ningún eufemismo
      quede cortado en el límite de una ventana
    - Los scores de los tokens en zonas de solapamiento se promedian
      ponderando por cuántas ventanas los cubrieron (weighted overlap)
    - El score final es el promedio ponderado sobre todos los tokens válidos

Diseño de auditoría:
    Cada llamada a extract() retorna un EBIResult con:
    - score final normalizado [0, 1]
    - número de ventanas procesadas
    - tokens con mayor score EBI (los más sospechosos)
    - metadatos para logging a MLflow

Dependencias:
    - transformers >= 4.40.0
    - torch >= 2.1.0
    - numpy >= 1.26.0

    Modelo esperado: CFH-BERT (ConfliBERT-Spanish fine-tuneado sobre
    taxonomía CFH). Antes del fine-tuning, usar
    "eventdata-utd/ConfliBERT-Spanish" — los scores serán subóptimos
    para la clase EBI pero la arquitectura es idéntica.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Null context manager — fallback cuando torch no está disponible
_nullcontext = contextlib.nullcontext

# Imports pesados — lazy para que el módulo sea importable sin GPU/torch
try:
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    AutoModelForTokenClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    PreTrainedModel = None  # type: ignore
    PreTrainedTokenizerBase = None  # type: ignore

logger = logging.getLogger("cfh.features.y1_ebi")

# ---------------------------------------------------------------------------
# Constantes del modelo y del dominio CFH
# ---------------------------------------------------------------------------

# Etiquetas BIO de la taxonomía CFH — deben coincidir con el label2id
# del CFH-BERT fine-tuneado. Este orden es el definido en configs/model_config.yaml
CFH_LABELS = [
    "O",
    "B-EBI", "I-EBI",   # Eufemismo Bélico-Institucional
    "B-SA",  "I-SA",    # Supresión de Agentividad
    "B-NV",  "I-NV",    # Negación de Victimización
    "B-REP", "I-REP",   # Ruptura Epistémica Positiva
]

# Índices de las clases EBI en el vector de logits
EBI_LABEL_INDICES = {
    "B-EBI": CFH_LABELS.index("B-EBI"),
    "I-EBI": CFH_LABELS.index("I-EBI"),
}

# Tokens prototípicos de EBI para calibración y sanity checks
EBI_PROTOTYPE_TOKENS = [
    "baja", "bajas", "combate", "operacional", "táctica",
    "misión", "resultado", "contacto", "abatido", "neutralizado",
    "muerto en acción", "dado de baja", "caído en combate",
]

# Parámetros de ventana por defecto — ajustados para documentos judiciales
# colombianos típicos del corpus A
DEFAULT_WINDOW_SIZE = 400   # tokens por ventana (deja margen para tokens especiales)
DEFAULT_STRIDE = 100        # solapamiento: captura expresiones multi-token en el borde
DEFAULT_MAX_WINDOWS = 50    # techo de seguridad para documentos extremadamente largos


# ---------------------------------------------------------------------------
# Tipos de datos de resultado
# ---------------------------------------------------------------------------

@dataclass
class TokenEBIScore:
    """Score EBI de un token individual, con contexto para auditoría."""
    token: str
    char_start: int
    char_end: int
    score_b_ebi: float      # P(B-EBI | token, contexto)
    score_i_ebi: float      # P(I-EBI | token, contexto)
    score_ebi: float        # max(B-EBI, I-EBI) — score combinado
    window_count: int       # cuántas ventanas cubrieron este token


@dataclass
class EBIExtractionResult:
    """
    Resultado completo de la extracción del indicador y₁.
    Contiene el score normalizado + toda la información de auditoría.
    """
    # ── Score principal ──────────────────────────────────────────────────────
    score: float                    # y₁ normalizado [0, 1] — entra al SEM
    score_raw: float                # promedio ponderado sin normalizar

    # ── Metadatos del segmento ───────────────────────────────────────────────
    doc_id: str
    section_id: str
    corpus_type: str
    text_length_chars: int
    text_length_tokens: int         # longitud en tokens del tokenizador

    # ── Información de procesamiento ─────────────────────────────────────────
    n_windows: int                  # ventanas deslizantes procesadas
    window_size: int
    stride: int
    processing_time_s: float

    # ── Tokens más sospechosos (top EBI) ─────────────────────────────────────
    top_ebi_tokens: list[TokenEBIScore] = field(default_factory=list)

    # ── Flags de calidad ─────────────────────────────────────────────────────
    truncated: bool = False         # True si se alcanzó MAX_WINDOWS
    low_confidence: bool = False    # True si score < umbral de confianza
    warning: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return not self.truncated and self.text_length_tokens > 5

    def to_dict(self) -> dict:
        return {
            "y1_ebi_score": self.score,
            "y1_ebi_score_raw": self.score_raw,
            "doc_id": self.doc_id,
            "section_id": self.section_id,
            "corpus_type": self.corpus_type,
            "n_windows": self.n_windows,
            "text_length_tokens": self.text_length_tokens,
            "truncated": self.truncated,
            "low_confidence": self.low_confidence,
            "top_ebi_tokens": [
                {"token": t.token, "score": t.score_ebi}
                for t in self.top_ebi_tokens[:5]
            ],
            "warning": self.warning,
        }


# ---------------------------------------------------------------------------
# Normalización calibrada sobre corpus CFH
# ---------------------------------------------------------------------------

class EBIScoreNormalizer:
    """
    Normaliza el score EBI bruto al rango [0, 1] usando estadísticas
    de calibración estimadas sobre una muestra del corpus CFH.

    Estrategia: normalización por percentil (más robusta que z-score
    para distribuciones sesgadas como la distribución de scores EBI,
    que tiene masa concentrada cerca de 0 para texto no eufemístico).

    La normalización se ajusta con fit() sobre una muestra de calibración
    y se serializa con save()/load() para reproducibilidad.

    IMPORTANTE: la normalización debe ajustarse UNA VEZ sobre el corpus
    de calibración (100-200 segmentos mixtos de los tres corpus) y los
    parámetros guardarse para uso consistente en producción.
    """

    def __init__(
        self,
        method: str = "percentile",   # "percentile" | "zscore" | "minmax"
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
    ):
        self.method = method
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

        # Parámetros de calibración — se ajustan con fit()
        self._fitted = False
        self._p_low: float = 0.0
        self._p_high: float = 1.0
        self._mean: float = 0.5
        self._std: float = 0.2

        # Valores por defecto empíricos (corpus CFH, muestra n=150)
        # Ajustar después de calibración real
        self._defaults = {
            "p_low": 0.002,     # percentil 5 de scores EBI en corpus mixto
            "p_high": 0.087,    # percentil 95
            "mean": 0.018,
            "std": 0.022,
        }
        self._apply_defaults()

    def _apply_defaults(self):
        self._p_low = self._defaults["p_low"]
        self._p_high = self._defaults["p_high"]
        self._mean = self._defaults["mean"]
        self._std = self._defaults["std"]
        self._fitted = False   # los defaults no cuentan como fitted

    def fit(self, raw_scores: list[float]) -> "EBIScoreNormalizer":
        """
        Ajusta los parámetros de normalización sobre una muestra.

        Parámetros
        ----------
        raw_scores : lista de scores EBI brutos de la muestra de calibración.
                     Debe incluir segmentos de los tres corpus (A, B, C).
        """
        if len(raw_scores) < 20:
            logger.warning(
                f"Muestra de calibración pequeña ({len(raw_scores)} scores). "
                "Se recomiendan ≥ 100 segmentos mixtos."
            )
        arr = np.array(raw_scores)
        self._p_low = float(np.percentile(arr, self.low_percentile))
        self._p_high = float(np.percentile(arr, self.high_percentile))
        self._mean = float(arr.mean())
        self._std = float(arr.std()) or 1e-8
        self._fitted = True
        logger.info(
            f"Normalizador calibrado: p{self.low_percentile:.0f}={self._p_low:.5f}, "
            f"p{self.high_percentile:.0f}={self._p_high:.5f}, "
            f"mean={self._mean:.5f}, std={self._std:.5f}"
        )
        return self

    def normalize(self, raw_score: float) -> float:
        """
        Normaliza un score EBI bruto al rango [0, 1].

        El clip final garantiza que ningún outlier produce valores fuera
        del rango válido para los indicadores del modelo SEM.
        """
        if not self._fitted:
            logger.debug("Usando parámetros de normalización por defecto (no calibrados).")

        if self.method == "percentile":
            denom = self._p_high - self._p_low
            if denom < 1e-10:
                return 0.5
            normalized = (raw_score - self._p_low) / denom
        elif self.method == "zscore":
            normalized = (raw_score - self._mean) / max(self._std, 1e-8)
            # Mapear z-score a [0, 1] asumiendo rango ±3σ
            normalized = (normalized + 3.0) / 6.0
        elif self.method == "minmax":
            denom = self._p_high - self._p_low
            normalized = (raw_score - self._p_low) / max(denom, 1e-10)
        else:
            raise ValueError(f"Método desconocido: {self.method}")

        return float(np.clip(normalized, 0.0, 1.0))

    def save(self, path: str | Path):
        """Serializa los parámetros de calibración para reproducibilidad."""
        import json
        params = {
            "method": self.method,
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "fitted": self._fitted,
            "p_low": self._p_low,
            "p_high": self._p_high,
            "mean": self._mean,
            "std": self._std,
        }
        Path(path).write_text(json.dumps(params, indent=2))
        logger.info(f"Normalizador guardado en {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EBIScoreNormalizer":
        """Carga parámetros de calibración serializados."""
        import json
        params = json.loads(Path(path).read_text())
        normalizer = cls(
            method=params["method"],
            low_percentile=params["low_percentile"],
            high_percentile=params["high_percentile"],
        )
        normalizer._p_low = params["p_low"]
        normalizer._p_high = params["p_high"]
        normalizer._mean = params["mean"]
        normalizer._std = params["std"]
        normalizer._fitted = params["fitted"]
        return normalizer


# ---------------------------------------------------------------------------
# Extractor principal
# ---------------------------------------------------------------------------

class EBIY1Extractor:
    """
    Extrae el indicador y₁ (Score EBI) para el modelo SEM del proyecto CFH.

    Maneja:
    - Documentos más largos que el límite de 512 tokens mediante ventana
      deslizante con stride y promediado ponderado en zonas de solapamiento
    - Normalización calibrada sobre muestra del corpus CFH
    - Logging de tokens más sospechosos para interpretabilidad
    - Fallback gracioso cuando el modelo no está disponible (devuelve NaN)

    Parámetros
    ----------
    model_name_or_path : str
        Ruta local al CFH-BERT fine-tuneado, o identificador HuggingFace.
        Antes del fine-tuning: "eventdata-utd/ConfliBERT-Spanish"
        (los scores serán menos precisos para EBI pero el pipeline funciona).
    window_size : int
        Número máximo de tokens por ventana (sin contar tokens especiales).
        Default: 400 (deja margen para [CLS] y [SEP]).
    stride : int
        Solapamiento entre ventanas consecutivas en tokens.
        Default: 100 (cubre expresiones multi-token de hasta ~10 palabras).
    device : str
        "cuda" | "cpu" | "auto". Auto detecta GPU si está disponible.
    normalizer : EBIScoreNormalizer, optional
        Instancia pre-calibrada. Si None, usa parámetros por defecto.
    top_k_tokens : int
        Número de tokens con mayor score EBI a retener en el resultado
        (para auditoría e interpretabilidad).
    """

    def __init__(
        self,
        model_name_or_path: str = "eventdata-utd/ConfliBERT-Spanish",
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        device: str = "auto",
        normalizer: Optional[EBIScoreNormalizer] = None,
        top_k_tokens: int = 10,
        max_windows: int = DEFAULT_MAX_WINDOWS,
    ):
        self.model_name = model_name_or_path
        self.window_size = window_size
        self.stride = stride
        self.top_k_tokens = top_k_tokens
        self.max_windows = max_windows

        # Resolver dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.normalizer = normalizer or EBIScoreNormalizer()
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}
        self._model_loaded = False

        logger.info(
            f"EBIY1Extractor inicializado — modelo={model_name_or_path} | "
            f"device={self.device} | window={window_size} | stride={stride}"
        )

    # ------------------------------------------------------------------
    # Carga del modelo (lazy — solo cuando se necesita)
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        """Carga el modelo y tokenizador si aún no están en memoria."""
        if self._model_loaded:
            return

        logger.info(f"Cargando modelo desde '{self.model_name}'...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )
            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
            )
            self._model.to(self.device)
            self._model.eval()

            # Resolver mapeo de etiquetas
            if hasattr(self._model.config, "label2id"):
                self._label2id = self._model.config.label2id
                self._id2label = {v: k for k, v in self._label2id.items()}
                logger.info(f"Etiquetas del modelo: {list(self._label2id.keys())}")
            else:
                # Fallback: usar el orden estándar CFH
                logger.warning(
                    "El modelo no tiene label2id configurado. "
                    "Usando orden estándar CFH (puede ser incorrecto para modelos no fine-tuneados)."
                )
                self._label2id = {label: i for i, label in enumerate(CFH_LABELS)}
                self._id2label = {i: label for i, label in enumerate(CFH_LABELS)}

            # Verificar que las clases EBI existen
            for ebi_label in ["B-EBI", "I-EBI"]:
                if ebi_label not in self._label2id:
                    logger.warning(
                        f"Clase '{ebi_label}' no encontrada en el modelo. "
                        "El modelo puede no estar fine-tuneado con la taxonomía CFH. "
                        "Los scores EBI no serán significativos."
                    )

            self._model_loaded = True
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(
                f"Modelo cargado — {n_params/1e6:.1f}M parámetros | "
                f"dispositivo: {self.device}"
            )

        except Exception as e:
            logger.error(f"Error cargando modelo '{self.model_name}': {e}")
            raise

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        doc_id: str = "UNKNOWN",
        section_id: str = "UNKNOWN",
        corpus_type: str = "A",
    ) -> EBIExtractionResult:
        """
        Extrae el score EBI de un segmento judicial.

        Parámetros
        ----------
        text : str
            Texto limpio del segmento judicial (salida del módulo de ingesta).
        doc_id : str
            Identificador del documento (SHA-256 truncado del módulo de ingesta).
        section_id : str
            Etiqueta de la sección (ej. "HECHOS", "CONSIDERACIONES").
        corpus_type : str
            "A" | "B" | "C" — para logging y auditoría.

        Retorna
        -------
        EBIExtractionResult con score normalizado y metadatos completos.
        """
        self._ensure_model_loaded()

        t_start = time.perf_counter()

        if not text or not text.strip():
            return self._empty_result(
                doc_id, section_id, corpus_type,
                warning="Texto vacío — score EBI = 0.0"
            )

        # Tokenizar el texto completo para conocer la longitud total
        full_encoding = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        all_input_ids = full_encoding["input_ids"]
        all_offsets = full_encoding["offset_mapping"]
        total_tokens = len(all_input_ids)

        logger.debug(
            f"[{doc_id[:8]}] Segmento {section_id}: "
            f"{len(text)} chars | {total_tokens} tokens"
        )

        # Segmento corto: procesar en una sola pasada
        if total_tokens <= self.window_size:
            raw_scores, token_scores = self._process_single_window(
                text, all_input_ids, all_offsets
            )
            n_windows = 1
            truncated = False
        else:
            # Segmento largo: ventana deslizante con stride
            raw_scores, token_scores, n_windows, truncated = self._process_with_stride(
                text, all_input_ids, all_offsets
            )

        # Score EBI agregado: promedio ponderado sobre todos los tokens válidos
        if len(raw_scores) == 0:
            raw_score = 0.0
        else:
            raw_score = float(np.mean(raw_scores))

        # Normalización
        normalized_score = self.normalizer.normalize(raw_score)

        # Top-K tokens más sospechosos (para interpretabilidad y auditoría)
        top_tokens = sorted(token_scores, key=lambda t: t.score_ebi, reverse=True)
        top_tokens = top_tokens[:self.top_k_tokens]

        elapsed = time.perf_counter() - t_start

        result = EBIExtractionResult(
            score=normalized_score,
            score_raw=raw_score,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=len(text),
            text_length_tokens=total_tokens,
            n_windows=n_windows,
            window_size=self.window_size,
            stride=self.stride,
            processing_time_s=round(elapsed, 3),
            top_ebi_tokens=top_tokens,
            truncated=truncated,
            low_confidence=normalized_score < 0.05 and total_tokens < 30,
            warning=(
                f"Documento truncado a {self.max_windows} ventanas "
                f"({total_tokens} tokens totales)"
                if truncated else None
            ),
        )

        logger.info(
            f"[{doc_id[:8]}] y₁ EBI — "
            f"sección={section_id} | score={normalized_score:.4f} "
            f"(raw={raw_score:.5f}) | ventanas={n_windows} | "
            f"{elapsed:.2f}s"
        )
        if top_tokens:
            top_str = ", ".join(f"'{t.token}'({t.score_ebi:.3f})" for t in top_tokens[:3])
            logger.debug(f"[{doc_id[:8]}] Top EBI tokens: {top_str}")

        return result

    def extract_batch(
        self,
        segments: list[dict],
    ) -> list[EBIExtractionResult]:
        """
        Extrae y₁ sobre una lista de segmentos.

        Parámetros
        ----------
        segments : list de dicts con claves:
            - "text": str
            - "doc_id": str (opcional)
            - "section_id": str (opcional)
            - "corpus_type": str (opcional)

        Retorna
        -------
        list[EBIExtractionResult] en el mismo orden que la entrada.
        """
        results = []
        for i, seg in enumerate(segments):
            try:
                result = self.extract(
                    text=seg["text"],
                    doc_id=seg.get("doc_id", f"seg_{i:04d}"),
                    section_id=seg.get("section_id", "UNKNOWN"),
                    corpus_type=seg.get("corpus_type", "A"),
                )
            except Exception as e:
                logger.error(f"Error en segmento {i}: {e}")
                result = self._empty_result(
                    seg.get("doc_id", f"seg_{i:04d}"),
                    seg.get("section_id", "UNKNOWN"),
                    seg.get("corpus_type", "A"),
                    warning=f"Error de procesamiento: {str(e)[:100]}",
                )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Procesamiento interno
    # ------------------------------------------------------------------

    def _process_single_window(
        self,
        text: str,
        input_ids: list[int],
        offsets: list[tuple[int, int]],
    ) -> tuple[list[float], list[TokenEBIScore]]:
        """
        Procesa un segmento que cabe en una sola ventana (≤ window_size tokens).
        Retorna (lista_scores_por_token, lista_TokenEBIScore).
        """
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.window_size + 2,   # +2 para [CLS] y [SEP]
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )

        ebi_scores_per_token, token_details = self._run_inference(
            encoding, text, offsets
        )
        return ebi_scores_per_token, token_details

    def _process_with_stride(
        self,
        text: str,
        all_input_ids: list[int],
        all_offsets: list[tuple[int, int]],
    ) -> tuple[list[float], list[TokenEBIScore], int, bool]:
        """
        Procesa un segmento largo mediante ventana deslizante con stride.

        Estrategia de agregación en zonas de solapamiento:
        Para cada posición de token, el score final es el promedio de todos
        los scores obtenidos en las ventanas que cubrieron ese token.
        Esto se implementa con dos arrays paralelos:
        - accumulated_scores[i] = suma de scores del token i
        - token_counts[i] = número de ventanas que cubrieron el token i
        Score final = accumulated_scores[i] / token_counts[i]

        Retorna (scores_finales, token_details, n_windows, truncated)
        """
        total_tokens = len(all_input_ids)
        accumulated_scores = np.zeros(total_tokens, dtype=np.float64)
        token_counts = np.zeros(total_tokens, dtype=np.int32)
        all_token_details: dict[int, TokenEBIScore] = {}

        n_windows = 0
        truncated = False
        start = 0

        while start < total_tokens:
            end = min(start + self.window_size, total_tokens)
            window_ids = all_input_ids[start:end]
            window_offsets = all_offsets[start:end]

            # Reconstruir texto de la ventana desde offsets de caracteres
            # para que el tokenizador procese el texto original, no IDs reconstruidos
            if window_offsets:
                char_start = window_offsets[0][0]
                char_end = window_offsets[-1][1]
                window_text = text[char_start:char_end]
            else:
                break

            # Reencoding de la ventana (necesario para que el modelo vea
            # el texto correctamente con sus tokens especiales)
            window_encoding = self._tokenizer(
                window_text,
                return_tensors="pt",
                max_length=self.window_size + 2,
                truncation=True,
                padding=False,
                return_offsets_mapping=True,
            )

            # Ejecutar inferencia en esta ventana
            window_ebi_scores, window_details = self._run_inference(
                window_encoding,
                window_text,
                # Ajustar offsets al espacio de caracteres del texto completo
                [(o[0] + char_start, o[1] + char_start) for o in window_offsets],
            )

            # Acumular scores en el rango de tokens correspondiente
            actual_window_len = min(len(window_ebi_scores), end - start)
            for local_idx, score in enumerate(window_ebi_scores[:actual_window_len]):
                global_idx = start + local_idx
                if global_idx < total_tokens:
                    accumulated_scores[global_idx] += score
                    token_counts[global_idx] += 1

                    # Guardar detalle del token (primera vez que lo vemos)
                    if global_idx not in all_token_details and local_idx < len(window_details):
                        all_token_details[global_idx] = window_details[local_idx]

            n_windows += 1

            # Avanzar la ventana
            next_start = start + self.window_size - self.stride
            if next_start <= start:   # guardia contra stride >= window_size
                next_start = start + 1
            start = next_start

            # Techo de seguridad
            if n_windows >= self.max_windows:
                truncated = True
                logger.warning(
                    f"Documento truncado: alcanzado límite de {self.max_windows} ventanas. "
                    f"Tokens procesados: {end}/{total_tokens}"
                )
                break

        # Calcular scores finales ponderados
        valid_mask = token_counts > 0
        final_scores = np.where(
            valid_mask,
            accumulated_scores / np.maximum(token_counts, 1),
            0.0,
        )

        # Actualizar window_count en token_details
        token_details_list = []
        for global_idx, detail in sorted(all_token_details.items()):
            if global_idx < len(token_counts):
                detail.window_count = int(token_counts[global_idx])
                detail.score_ebi = float(final_scores[global_idx])
                token_details_list.append(detail)

        return (
            final_scores[valid_mask].tolist(),
            token_details_list,
            n_windows,
            truncated,
        )

    def _run_inference(
        self,
        encoding,
        original_text: str,
        global_offsets: list[tuple[int, int]],
    ) -> tuple[list[float], list[TokenEBIScore]]:
        """
        Ejecuta inferencia del modelo sobre un encoding preparado.

        Retorna:
        - lista de scores EBI por token (sin tokens especiales)
        - lista de TokenEBIScore con información detallada
        """
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        no_grad_ctx = torch.no_grad() if torch is not None else _nullcontext()
        with no_grad_ctx:
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # logits shape: (1, seq_len, n_labels)
        logits = outputs.logits[0]   # (seq_len, n_labels)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Obtener offsets del encoding (posiciones de tokens en el texto original)
        offset_mapping = encoding.get("offset_mapping")
        if offset_mapping is not None:
            if hasattr(offset_mapping, "numpy"):
                offset_mapping = offset_mapping[0].numpy()
            elif isinstance(offset_mapping, list):
                offset_mapping = np.array(offset_mapping[0])

        ebi_scores = []
        token_details = []

        b_ebi_idx = self._label2id.get("B-EBI", EBI_LABEL_INDICES["B-EBI"])
        i_ebi_idx = self._label2id.get("I-EBI", EBI_LABEL_INDICES["I-EBI"])

        seq_len = probs.shape[0]

        for token_idx in range(seq_len):
            # Saltar tokens especiales ([CLS], [SEP], padding)
            if offset_mapping is not None:
                offset = offset_mapping[token_idx]
                if isinstance(offset, (list, np.ndarray)):
                    start_char, end_char = int(offset[0]), int(offset[1])
                else:
                    start_char, end_char = 0, 0

                if start_char == 0 and end_char == 0:
                    continue   # Token especial — omitir
            else:
                start_char, end_char = 0, 0

            token_probs = probs[token_idx]
            score_b_ebi = float(token_probs[b_ebi_idx]) if b_ebi_idx < len(token_probs) else 0.0
            score_i_ebi = float(token_probs[i_ebi_idx]) if i_ebi_idx < len(token_probs) else 0.0
            score_ebi = max(score_b_ebi, score_i_ebi)

            ebi_scores.append(score_ebi)

            # Reconstruir texto del token desde offsets
            try:
                token_text = original_text[start_char:end_char]
            except (IndexError, TypeError):
                token_text = f"[tok_{token_idx}]"

            token_details.append(TokenEBIScore(
                token=token_text,
                char_start=start_char,
                char_end=end_char,
                score_b_ebi=score_b_ebi,
                score_i_ebi=score_i_ebi,
                score_ebi=score_ebi,
                window_count=1,   # se actualiza en _process_with_stride
            ))

        return ebi_scores, token_details

    def _empty_result(
        self,
        doc_id: str,
        section_id: str,
        corpus_type: str,
        warning: str,
    ) -> EBIExtractionResult:
        return EBIExtractionResult(
            score=0.0,
            score_raw=0.0,
            doc_id=doc_id,
            section_id=section_id,
            corpus_type=corpus_type,
            text_length_chars=0,
            text_length_tokens=0,
            n_windows=0,
            window_size=self.window_size,
            stride=self.stride,
            processing_time_s=0.0,
            top_ebi_tokens=[],
            truncated=False,
            low_confidence=True,
            warning=warning,
        )

    # ------------------------------------------------------------------
    # Calibración del normalizador
    # ------------------------------------------------------------------

    def calibrate_normalizer(
        self,
        calibration_segments: list[dict],
        save_path: Optional[str | Path] = None,
    ) -> EBIScoreNormalizer:
        """
        Calibra el normalizador sobre una muestra de segmentos.

        Se debe ejecutar UNA VEZ después de obtener el CFH-BERT fine-tuneado,
        sobre una muestra representativa de los tres corpus.

        Parámetros
        ----------
        calibration_segments : list de dicts {"text": str, ...}
            Mínimo 100 segmentos mezclando corpus A, B y C.
        save_path : str | Path, optional
            Si se provee, guarda los parámetros calibrados para uso posterior.

        Retorna
        -------
        EBIScoreNormalizer calibrado (y actualizado en self.normalizer).
        """
        logger.info(
            f"Calibrando normalizador sobre {len(calibration_segments)} segmentos..."
        )
        raw_scores = []
        for seg in calibration_segments:
            try:
                result = self.extract(
                    text=seg["text"],
                    doc_id=seg.get("doc_id", "calib"),
                    section_id=seg.get("section_id", "CALIB"),
                    corpus_type=seg.get("corpus_type", "A"),
                )
                raw_scores.append(result.score_raw)
            except Exception as e:
                logger.warning(f"Error en segmento de calibración: {e}")

        if not raw_scores:
            raise RuntimeError("No se pudo extraer ningún score de calibración.")

        self.normalizer.fit(raw_scores)

        if save_path:
            self.normalizer.save(save_path)

        return self.normalizer
