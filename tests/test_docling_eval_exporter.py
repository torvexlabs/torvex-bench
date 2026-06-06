from pathlib import Path

from docling_core.types.doc.document import DoclingDocument

from torvex_bench.exporters.docling_eval import export_rows_as_docling_json


def test_export_rows_as_docling_json_loads_and_exports_html(tmp_path: Path) -> None:
    out = tmp_path / "prediction.json"

    export_rows_as_docling_json(
        rows=[
            ["A", "B"],
            ["1", "2"],
        ],
        output_path=out,
        name="test prediction",
    )

    doc = DoclingDocument.load_from_json(out)

    assert len(doc.tables) == 1
    assert doc.tables[0].export_to_html(doc) == (
        "<table><tbody>"
        "<tr><td>A</td><td>B</td></tr>"
        "<tr><td>1</td><td>2</td></tr>"
        "</tbody></table>"
    )
