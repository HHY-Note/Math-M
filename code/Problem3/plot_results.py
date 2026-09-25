"""绘制第三问的同方案双配置曲线；仅此可选入口依赖 ReportLab。"""

from pathlib import Path
import argparse
import csv

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.colors import HexColor, white


def plot(folder, png=False):
    with (folder / 'curves.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [int(r['cores']) for r in rows] != list(range(1, 6)):
        raise ValueError('需要完整的 1～5 核曲线数据')
    specs = [('configuration_comparison', 'Normalized completion time',
              [('mean_no_l2_normalized', 'No L2'), ('mean_l2_normalized', 'Read-only L2')]),
             ('cache_speedup', 'Cache speedup on the same plan', [('mean_cache_speedup', 'Mean speedup')])]
    for filename, title, fields in specs:
        drawing = Drawing(640, 420)
        drawing.add(Rect(0, 0, 640, 420, fillColor=white, strokeColor=None))
        drawing.add(String(60, 385, title, fontName='Helvetica-Bold', fontSize=17))
        drawing.add(String(60, 365, f'{rows[0]["cases"]} selected cases | official fixed settings | arithmetic means',
                           fontName='Helvetica', fontSize=10))
        chart = LinePlot()
        chart.x, chart.y, chart.width, chart.height = 65, 70, 520, 245
        chart.data = [[(int(r['cores']), float(r[key])) for r in rows] for key, _ in fields]
        chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 1, 5
        chart.xValueAxis.valueSteps = [1, 2, 3, 4, 5]
        chart.yValueAxis.valueMin = 0 if len(fields) == 2 else min(1, min(y for points in chart.data for _, y in points))
        chart.yValueAxis.valueMax = max(y for points in chart.data for _, y in points) * 1.08
        chart.yValueAxis.visibleGrid = True
        chart.yValueAxis.gridStrokeColor = HexColor('#E5EAF0')
        for i, (_, label) in enumerate(fields):
            color = HexColor(('#C57432', '#1665B5')[i] if len(fields) == 2 else '#1665B5')
            chart.lines[i].strokeColor, chart.lines[i].strokeWidth = color, 2
            chart.lines[i].symbol = makeMarker('FilledCircle', size=5, fillColor=color)
            drawing.add(Line(65 + i * 200, 337, 87 + i * 200, 337, strokeColor=color, strokeWidth=2))
            drawing.add(String(95 + i * 200, 333, label, fontName='Helvetica', fontSize=10))
        drawing.add(chart)
        drawing.add(String(320, 39, 'Number of cores', fontName='Helvetica', fontSize=11, textAnchor='middle'))
        note = ('Each time divided by its whole-graph single-core no-L2 baseline.' if len(fields) == 2
                else 'Per-case no-L2 time / L2 time; sample results do not represent all 100 cases.')
        drawing.add(String(60, 15, note, fontName='Helvetica', fontSize=9))
        renderSVG.drawToFile(drawing, str(folder / f'{filename}.svg'))
        if png:
            import pypdfium2
            from reportlab.graphics import renderPDF
            with pypdfium2.PdfDocument(renderPDF.drawToString(drawing)) as document:
                document[0].render(scale=2).to_pil().save(folder / f'{filename}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()
    plot(args.folder, args.png)
