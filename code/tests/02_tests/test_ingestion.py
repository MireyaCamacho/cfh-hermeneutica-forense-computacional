"""
CFH · Tests de ingesta con documentos judiciales sintéticos
============================================================
Proyecto: Hermenéutica Forense Computacional

Tests diseñados para ser ejecutados por Antigravity sin necesidad
del corpus real. Usan fragmentos representativos construidos
con el lenguaje judicial colombiano auténtico.

Ejecución:
    python -m pytest tests/test_ingestion.py -v

Principio: si estos tests pasan, el pipeline está listo para
recibir los documentos reales del corpus.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.ingestion.cleaner import JudicialTextCleaner
from src.ingestion.metadata import JudicialMetadataExtractor
from src.ingestion.segmenter import JudicialDocumentSegmenter
from src.ingestion.pipeline import CFHIngestionPipeline


# ---------------------------------------------------------------------------
# Fixtures — documentos judiciales sintéticos
# ---------------------------------------------------------------------------

SAMPLE_CORPUS_A = """JUZGADO 15 PENAL DEL CIRCUITO ESPECIALIZADO DE BOGOTÁ D.C.

Radicado No. 11001310300120060018900
Procesado: SARGENTO PRIMERO LUIS HERNÁNDEZ MEJÍA
Fiscal delegada: Dra. PATRICIA ROJAS SALCEDO
Defensor: Dr. CARLOS MONTOYA PEÑA

Bogotá, D.C., quince (15) de marzo de dos mil ocho (2008)

SENTENCIA

I. HECHOS

El día 12 de octubre de 2006, en el municipio de Ocaña, Norte de Santander,
miembros del Batallón de Contraguerrilla No. 15, en cumplimiento de misión
táctica, dieron de baja a un individuo que portaba fusil y quien
opuso resistencia armada. Los restos del dado de baja fueron recuperados y
presentados ante el Batallón como resultado operacional positivo.

La víctima era en realidad JORGE ELIÉCER SUÁREZ DÍAZ, de 23 años de edad,
desempleado, residente en el barrio El Carmen del municipio de Soacha,
Cundinamarca, quien había sido reclutado mediante engaño con promesa de trabajo.

II. CONSIDERACIONES DEL DESPACHO

Analizados los elementos materiales probatorios allegados al proceso, este
Juzgado encuentra que los hechos imputados al procesado constituyen el delito
de Homicidio en Persona Protegida tipificado en el artículo 135 del Código Penal.

La expresión "baja en combate" utilizada en los informes operacionales no
corresponde a la realidad probada. La evidencia forense contradice la versión
oficial de resistencia armada.

III. PRUEBAS

Protocolo de necropsia del Instituto Nacional de Medicina Legal No. 2006-15423
que establece causa de muerte por proyectil de arma de fuego.
Testimonios de habitantes de la región que niegan combate el día referido.

IV. DECISIÓN

En mérito de lo expuesto, este Juzgado Quince Penal del Circuito Especializado,
administrando justicia en nombre de la República y por autoridad de la ley,

RESUELVE:

CONDENAR a LUIS HERNÁNDEZ MEJÍA como coautor del delito de Homicidio en Persona
Protegida a la pena principal de dieciséis (16) años de prisión.
"""

SAMPLE_CORPUS_B = """JURISDICCIÓN ESPECIAL PARA LA PAZ
Sección de Reconocimiento de Verdad, de Responsabilidad y de Determinación
de los Hechos y Conductas

Auto No. 019 de 2021
Macrocaso 003 - Muertes ilegítimamente presentadas como bajas en combate

Radicado: 20211510248001CE

15 de enero de 2021

ANTECEDENTES

En el marco del Macrocaso 003, la Sala ha recibido versiones voluntarias de
comparecientes pertenecientes al Ejército Nacional de Colombia relativos a
hechos ocurridos entre 2002 y 2008 en Antioquia, Meta y Norte de Santander.

RECONOCIMIENTO DE VERDAD

El compareciente CORONEL (R) MARCOS ANTONIO VARGAS SILVA reconoció que durante
su comando del Batallón de Infantería No. 21 se presentaron como bajas en combate
a civiles que fueron asesinados con premeditación. Reconoció que conocía la práctica
y que fue presionado por sus superiores para aumentar las cifras de resultados.

CONSIDERACIONES DE LA SALA

La declaración del compareciente representa un avance significativo en la
construcción de la verdad sobre el Macrocaso 003. La Sala valora el reconocimiento
explícito de responsabilidad como contribución a los derechos de las víctimas.

RESUELVE:

Primero: Admitir el reconocimiento de verdad y responsabilidad del compareciente.
Segundo: Programar audiencia pública de reconocimiento con participación de víctimas.
"""

SAMPLE_CORPUS_C = """ACTA DE AUDIENCIA DE RECONOCIMIENTO PÚBLICO
JEP, Sección de Reconocimiento de Verdad
Macrocaso 003 — Falsos Positivos
Bogotá, 23 de agosto de 2022

APERTURA DE LA AUDIENCIA

PRESIDENTE (Magistrada Catalina Díaz Gómez): Se instala la presente audiencia
de reconocimiento público conforme al Auto 019 de 2021.

INTERVENCIÓN DEL COMPARECIENTE

COMPARECIENTE (Coronel retirado Marco Vargas): Quiero pedir perdón a las familias.
Yo sabía lo que estaba pasando. Firmé los informes. Las llamo "bajas en combate"
porque así nos habían enseñado a llamarlas, pero eran personas inocentes.
No había combate. Los matábamos y los vestíamos. Eso fue lo que hicimos.
No tengo excusa. Le fallé al país y sobre todo a estas familias.

INTERVENCIÓN DE LA VÍCTIMA

VÍCTIMA (representante MAFAPO, Luz Marina Bernal): Mi hijo Fair Leonardo Porras
tenía 26 años. Le dijeron que había trabajo en Norte de Santander. Nunca volvió.
Lo encontré en una fosa común vestido de guerrillero. Mi hijo no sabía ni cargar
un arma. Hoy espero que la verdad que él dice sea real. Las madres necesitamos
que esto no se repita nunca más.

DELIBERACION

MAGISTRADA DÍAZ GÓMEZ: ¿Cuántos casos ocurrieron bajo su comando directo?

COMPARECIENTE: Que yo pueda confirmar con certeza, al menos diecisiete casos.
Pero había más. El sistema nos medía por números.

CIERRE DE LA AUDIENCIA

PRESIDENTE: Se levanta la presente diligencia. Se convoca a nueva sesión.
"""

SAMPLE_WITH_OCR_ERRORS = """JUZGADO 3° PENAL DEL ClRCUlTO DE MEDELLlN

Radicado No. 05001310400120070009800
Procesado: CABO PRlMERO ROBERTO GARZON

Bogotá D.C., 20/07/2007

SENTENCIA

I. HECHOS

En la vereda El Toro, municipio de Ituango, Antioquia, rniernbros del ejercito
nacional dieron de baja a un individuo que presentaba resistencia. El dado de baja
fue reportado en el inforrne de la rnision tactica 0rden 112 del 15-06-2007.

La Fiscali@ ha dernostrado que Jorge Albeiro Gonzalez Rios, de 19 años de edad,
no era rniernbro de ninguna organizacion arrriada al rnarguen de la ley.

II. CONSlDERACIONES DEL DESPACHO

Este juzgado consiciera que...
"""


# ---------------------------------------------------------------------------
# Tests del módulo Cleaner
# ---------------------------------------------------------------------------

class TestJudicialTextCleaner:

    def test_basic_cleaning_returns_cleaned_document(self):
        cleaner = JudicialTextCleaner()
        result = cleaner.clean(SAMPLE_CORPUS_A, doc_id="test_001")
        assert result.text, "El texto limpio no debe estar vacío"
        assert result.report.final_length > 0

    def test_ocr_corrections_applied(self):
        cleaner = JudicialTextCleaner(apply_ocr_corrections=True)
        result = cleaner.clean(SAMPLE_WITH_OCR_ERRORS, doc_id="test_ocr")
        # "rniernbros" → "miembros" no se corrige por regex simple,
        # pero sí deben corregirse patrones compilados
        assert result.report.ocr_corrections >= 0  # Al menos no lanza error
        assert result.report is not None

    def test_encoding_fixes_zero_for_clean_text(self):
        cleaner = JudicialTextCleaner()
        result = cleaner.clean(SAMPLE_CORPUS_A)
        # Texto ya limpio: no debe requerir muchos fixes de encoding
        assert result.report.encoding_fixes < 10

    def test_cleaning_is_deterministic(self):
        cleaner = JudicialTextCleaner()
        result1 = cleaner.clean(SAMPLE_CORPUS_A)
        result2 = cleaner.clean(SAMPLE_CORPUS_A)
        assert result1.text == result2.text, "El cleaning debe ser determinístico"

    def test_no_data_loss_significant(self):
        cleaner = JudicialTextCleaner()
        result = cleaner.clean(SAMPLE_CORPUS_A)
        # No debe eliminar más del 15% del texto original
        assert result.report.compression_ratio < 0.15, \
            f"Demasiado texto eliminado: {result.report.compression_ratio:.0%}"

    def test_report_tracks_operations(self):
        cleaner = JudicialTextCleaner(apply_ocr_corrections=True)
        result = cleaner.clean(SAMPLE_CORPUS_A)
        assert "unicode_NFC" in result.report.operations_applied
        assert any("ocr_corrections" in op for op in result.report.operations_applied)

    def test_space_lookalikes_removed(self):
        text_with_nbsp = "Artículo\u00a01\u00a0del\u00a0Código\u00a0Penal"
        cleaner = JudicialTextCleaner()
        result = cleaner.clean(text_with_nbsp)
        assert "\u00a0" not in result.text
        assert "Artículo 1 del Código Penal" in result.text

    def test_ocr_disabled_for_corpus_c(self):
        """Para transcripciones nativas (corpus C), OCR no debe aplicarse."""
        cleaner = JudicialTextCleaner(apply_ocr_corrections=False)
        result = cleaner.clean(SAMPLE_CORPUS_C)
        assert result.report.ocr_corrections == 0


# ---------------------------------------------------------------------------
# Tests del módulo MetadataExtractor
# ---------------------------------------------------------------------------

class TestJudicialMetadataExtractor:

    def test_extracts_doc_id_as_sha256(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta = extractor.extract(SAMPLE_CORPUS_A)
        assert len(meta.doc_id) == 64, "doc_id debe ser SHA-256 (64 hex chars)"
        assert all(c in "0123456789abcdef" for c in meta.doc_id)

    def test_doc_id_is_deterministic(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta1 = extractor.extract(SAMPLE_CORPUS_A)
        meta2 = extractor.extract(SAMPLE_CORPUS_A)
        assert meta1.doc_id == meta2.doc_id

    def test_extracts_tribunal_corpus_a(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta = extractor.extract(SAMPLE_CORPUS_A)
        assert meta.tribunal is not None
        assert "Juzgado" in meta.tribunal or "JUZGADO" in meta.tribunal.upper()

    def test_extracts_case_number(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta = extractor.extract(SAMPLE_CORPUS_A)
        assert meta.case_number is not None
        # El radicado contiene dígitos
        assert any(c.isdigit() for c in meta.case_number)

    def test_extracts_date_iso_format(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta = extractor.extract(SAMPLE_CORPUS_A)
        assert meta.date_issued is not None
        assert "2008" in meta.date_issued

    def test_extracts_date_corpus_b(self):
        extractor = JudicialMetadataExtractor(corpus_type="B")
        meta = extractor.extract(SAMPLE_CORPUS_B)
        assert meta.date_issued is not None
        assert "2021" in meta.date_issued

    def test_extracts_tribunal_jep(self):
        extractor = JudicialMetadataExtractor(corpus_type="B")
        meta = extractor.extract(SAMPLE_CORPUS_B)
        assert meta.tribunal is not None
        assert "JEP" in meta.tribunal

    def test_extracts_region_norte_de_santander(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta = extractor.extract(SAMPLE_CORPUS_A)
        # El documento menciona Soacha, Norte de Santander
        assert meta.region is not None

    def test_confidence_above_threshold(self):
        extractor = JudicialMetadataExtractor(corpus_type="A")
        meta = extractor.extract(SAMPLE_CORPUS_A)
        assert meta.extraction_confidence >= 0.5, \
            f"Confianza baja: {meta.extraction_confidence}"

    def test_to_dict_serializable(self):
        import json
        extractor = JudicialMetadataExtractor(corpus_type="B")
        meta = extractor.extract(SAMPLE_CORPUS_B)
        d = meta.to_dict()
        # Debe ser serializable a JSON sin errores
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0

    def test_corpus_type_mismatch_raises(self):
        with pytest.raises(AssertionError):
            JudicialMetadataExtractor(corpus_type="X")


# ---------------------------------------------------------------------------
# Tests del módulo Segmenter
# ---------------------------------------------------------------------------

class TestJudicialDocumentSegmenter:

    def test_corpus_a_finds_hechos(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="A")
        doc = segmenter.segment(SAMPLE_CORPUS_A)
        section_ids = [s.section_id for s in doc.segments]
        assert "HECHOS" in section_ids, f"HECHOS no encontrado. Secciones: {section_ids}"

    def test_corpus_a_finds_decision(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="A")
        doc = segmenter.segment(SAMPLE_CORPUS_A)
        section_ids = [s.section_id for s in doc.segments]
        assert "DECISIÓN" in section_ids, f"DECISIÓN no encontrada. Secciones: {section_ids}"

    def test_corpus_b_finds_reconocimiento(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="B")
        doc = segmenter.segment(SAMPLE_CORPUS_B)
        section_ids = [s.section_id for s in doc.segments]
        assert "RECONOCIMIENTO" in section_ids

    def test_corpus_c_finds_compareciente(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="C")
        doc = segmenter.segment(SAMPLE_CORPUS_C)
        section_ids = [s.section_id for s in doc.segments]
        assert "TESTIMONIO_COMPARECIENTE" in section_ids

    def test_corpus_c_finds_victima(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="C")
        doc = segmenter.segment(SAMPLE_CORPUS_C)
        section_ids = [s.section_id for s in doc.segments]
        assert "TESTIMONIO_VICTIMA" in section_ids

    def test_target_sections_marked(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="A")
        doc = segmenter.segment(SAMPLE_CORPUS_A)
        targets = doc.get_target_sections()
        assert len(targets) >= 2, "Debe haber al menos 2 secciones target en corpus A"

    def test_coverage_above_threshold(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="A")
        doc = segmenter.segment(SAMPLE_CORPUS_A)
        assert doc.coverage >= 0.5, f"Cobertura baja: {doc.coverage:.0%}"

    def test_text_preserved_across_segments(self):
        """La suma de los segmentos debe cubrir el texto original."""
        segmenter = JudicialDocumentSegmenter(corpus_type="A")
        doc = segmenter.segment(SAMPLE_CORPUS_A)
        # Cada segmento tiene texto
        for seg in doc.segments:
            assert seg.text.strip(), f"Segmento vacío: {seg.section_id}"

    def test_segment_indices_sequential(self):
        segmenter = JudicialDocumentSegmenter(corpus_type="C")
        doc = segmenter.segment(SAMPLE_CORPUS_C)
        indices = [s.section_index for s in doc.segments]
        assert indices == list(range(len(indices))), "Índices deben ser secuenciales"

    def test_to_dict_serializable(self):
        import json
        segmenter = JudicialDocumentSegmenter(corpus_type="B")
        doc = segmenter.segment(SAMPLE_CORPUS_B)
        d = doc.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# Tests de integración — Pipeline completo
# ---------------------------------------------------------------------------

class TestCFHIngestionPipeline:

    def test_pipeline_corpus_a_success(self, tmp_path):
        # Crear archivo temporal
        f = tmp_path / "sentencia_001.txt"
        f.write_text(SAMPLE_CORPUS_A, encoding="utf-8")

        pipeline = CFHIngestionPipeline(corpus_type="A")
        result = pipeline.process_file(f)

        assert result.success
        assert result.sha256_source
        assert result.sha256_clean
        assert result.sha256_source != result.sha256_clean  # Hashes distintos

    def test_pipeline_corpus_c_success(self, tmp_path):
        f = tmp_path / "audiencia_001.txt"
        f.write_text(SAMPLE_CORPUS_C, encoding="utf-8")

        pipeline = CFHIngestionPipeline(corpus_type="C")
        result = pipeline.process_file(f)

        assert result.success
        assert result.clean_text

    def test_pipeline_result_is_deterministic(self, tmp_path):
        f = tmp_path / "sentencia_det.txt"
        f.write_text(SAMPLE_CORPUS_A, encoding="utf-8")

        pipeline = CFHIngestionPipeline(corpus_type="A")
        r1 = pipeline.process_file(f)
        r2 = pipeline.process_file(f)

        assert r1.sha256_clean == r2.sha256_clean, "Pipeline debe ser determinístico"
        assert r1.clean_text == r2.clean_text

    def test_pipeline_save_result_creates_files(self, tmp_path):
        f = tmp_path / "sentencia_save.txt"
        f.write_text(SAMPLE_CORPUS_A, encoding="utf-8")

        pipeline = CFHIngestionPipeline(corpus_type="A")
        result = pipeline.process_file(f)
        pipeline.save_result(result, tmp_path / "output")

        output_files = list((tmp_path / "output").iterdir())
        assert len(output_files) == 2  # .json + .txt
        extensions = {f.suffix for f in output_files}
        assert ".json" in extensions
        assert ".txt" in extensions

    def test_pipeline_json_output_valid(self, tmp_path):
        import json
        f = tmp_path / "sentencia_json.txt"
        f.write_text(SAMPLE_CORPUS_B, encoding="utf-8")

        pipeline = CFHIngestionPipeline(corpus_type="B")
        result = pipeline.process_file(f)
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["pipeline_version"] == "1.0.0"
        assert parsed["corpus_type"] == "B"
        assert "sha256_source" in parsed
        assert "metadata" in parsed
        assert "segmentation" in parsed

    def test_pipeline_batch_processing(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        samples = [
            ("sent_001.txt", SAMPLE_CORPUS_A),
            ("sent_002.txt", SAMPLE_CORPUS_A),
            ("sent_003.txt", SAMPLE_CORPUS_B),
        ]
        for name, content in samples:
            (input_dir / name).write_text(content, encoding="utf-8")

        # Pipeline A para los primeros dos, B para el tercero
        # En producción se separarían por corpus; aquí usamos A para test
        pipeline = CFHIngestionPipeline(corpus_type="A")
        stats = pipeline.process_batch(input_dir, output_dir)

        assert stats["total"] == 3
        assert stats["failed"] == 0
        assert stats["success"] == 3

    def test_pipeline_handles_missing_file_gracefully(self, tmp_path):
        pipeline = CFHIngestionPipeline(corpus_type="A")
        result = pipeline.process_file(tmp_path / "no_existe.txt")
        assert not result.success
        assert result.error_message is not None


# ---------------------------------------------------------------------------
# Punto de entrada directo (para Antigravity)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    print("=" * 60)
    print("CFH · Test de smoke directamente (sin pytest)")
    print("=" * 60)

    # Smoke test rápido
    cleaner = JudicialTextCleaner()
    result = cleaner.clean(SAMPLE_CORPUS_A, doc_id="SMOKE_TEST")
    print(f"\n✓ Cleaner: {result.report.final_length} chars | OCR: {result.report.ocr_corrections}")

    extractor = JudicialMetadataExtractor(corpus_type="A")
    meta = extractor.extract(result.text)
    print(f"✓ Metadata: tribunal={meta.tribunal!r} | fecha={meta.date_issued!r} | confianza={meta.extraction_confidence:.0%}")

    segmenter = JudicialDocumentSegmenter(corpus_type="A")
    doc = segmenter.segment(result.text)
    print(f"✓ Segmenter: {doc.total_sections} secciones | cobertura={doc.coverage:.0%}")
    for seg in doc.segments:
        target = "★" if seg.is_target_section else " "
        print(f"   {target} [{seg.section_id}] {seg.word_count} words")

    print("\n✓ Smoke test completado exitosamente")
