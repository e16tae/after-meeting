"""PDF renderer — converts meeting documents to PDF via DOCX intermediate."""

from __future__ import annotations

import tempfile
from pathlib import Path

from after_meeting.errors import RenderError
from after_meeting.models import StructuredMeeting
from after_meeting.rendering import register
from after_meeting.rendering.docx_renderer import DocxRenderer


class PdfRenderer:
    """Render a StructuredMeeting to PDF by first creating a DOCX, then converting."""

    def __init__(self) -> None:
        self._docx_renderer = DocxRenderer()

    def render(self, meeting: StructuredMeeting, output_path: Path) -> Path:
        """Create a .pdf document from *meeting* and write it to *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from docx2pdf import convert  # type: ignore[import-untyped]
        except ImportError:
            raise RenderError(
                "PDF rendering requires the 'docx2pdf' package. "
                "Install it with:  uv pip install after-meeting[pdf]",
                code="PDF_DEPENDENCY_MISSING",
                recoverable=False,
            )

        # Render intermediate DOCX in a temp file to avoid overwriting user files
        with tempfile.TemporaryDirectory() as tmp_dir:
            docx_path = Path(tmp_dir) / "intermediate.docx"
            self._docx_renderer.render(meeting, docx_path)

            try:
                convert(str(docx_path), str(output_path))
            except Exception as exc:
                raise RenderError(
                    f"Failed to convert DOCX to PDF: {exc}",
                    code="PDF_CONVERSION_FAILED",
                    recoverable=False,
                ) from exc

        return output_path


# Register with the rendering registry
register("pdf", PdfRenderer)
