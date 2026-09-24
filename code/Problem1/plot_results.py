"""可选绘图入口；依赖 ReportLab，求解和评测程序不依赖绘图库。"""

from pathlib import Path
import argparse
import csv
import statistics

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.colors import HexColor, white


def plot(folder, png=False):
    with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    cases = {r['case'] for r in rows}
    points, previous = [(1, 1.0)], [(1, 1.0)]
    for n in range(2, 6):
        values = [r for r in rows if int(r['cores']) == n]
        if len(values) != len(cases) or {r['case'] for r in values} != cases:
            raise ValueError('每个核数必须覆盖相同且不重复的用例集')
        points.append((n, statistics.mean(float(r['speedup']) for r in values)))
        if 'previous_speedup' in rows[0]:
            previous.append((n, statistics.mean(float(r['previous_speedup']) for r in values)))
    blue, gray, ink = map(HexColor, ('#1665B5', '#A3ADBA', '#243447'))
    drawing = Drawing(640, 440)
    drawing.add(Rect(0, 0, 640, 440, fillColor=white, strokeColor=None))
    drawing.add(String(62, 407, 'Problem 1: mean speedup', fontName='Helvetica-Bold',
                       fontSize=18, fillColor=ink))
    drawing.add(String(62, 386, f'{len(cases)} selected cases | arithmetic mean | fixed official settings',
                       fontSize=10, fillColor=ink))
    chart = LinePlot()
    chart.reversePlotOrder = True
    chart.x, chart.y, chart.width, chart.height = 65, 76, 515, 270
    chart.data = [points, [(n, n) for n in range(1, 6)]]
    if len(previous) > 1:
        chart.data.append(previous)
    chart.xValueAxis.valueMin, chart.xValueAxis.valueMax = 1, 5
    chart.xValueAxis.valueSteps = [1, 2, 3, 4, 5]
    chart.yValueAxis.valueMin, chart.yValueAxis.valueMax = 0, 5.5
    chart.yValueAxis.valueSteps = [0, 1, 2, 3, 4, 5]
    chart.yValueAxis.visibleGrid = True
    chart.yValueAxis.gridStrokeColor = HexColor('#E5EAF0')
    chart.xValueAxis.strokeColor = chart.yValueAxis.strokeColor = gray
    chart.xValueAxis.labels.fontName = chart.yValueAxis.labels.fontName = 'Helvetica'
    chart.lines[0].strokeColor, chart.lines[0].strokeWidth = blue, 2.5
    chart.lines[0].symbol = makeMarker('FilledCircle', size=6, fillColor=blue)
    chart.lines[1].strokeColor = gray
    chart.lines[1].strokeDashArray = [4, 4]
    orange = HexColor('#C57432')
    if len(previous) > 1:
        chart.lines[2].strokeColor, chart.lines[2].strokeWidth = orange, 2
        chart.lines[2].symbol = makeMarker('FilledSquare', size=5, fillColor=orange)
    drawing.add(chart)
    for n, value in points:
        x = chart.x + (n - 1) * chart.width / 4
        y = chart.y + value * chart.height / 5.5
        drawing.add(String(x, y + 12, f'{value:.2f}x', fontSize=11,
                           textAnchor='middle', fillColor=blue))
    legend = [(65, 'Current version', blue), (310, 'Linear reference: speedup = cores', gray)]
    if len(previous) > 1:
        legend.insert(1, (200, 'Previous', orange))
    for x, label, color in legend:
        drawing.add(Line(x, 365, x + 24, 365, strokeColor=color, strokeWidth=2,
                         strokeDashArray=[4, 4] if color == gray else None))
        drawing.add(String(x + 32, 361, label, fontSize=10, fillColor=ink))
    drawing.add(String(325, 42, 'Number of cores', fontSize=11, textAnchor='middle', fillColor=ink))
    drawing.add(String(15, 351, 'Speedup', fontSize=10, fillColor=ink))
    drawing.add(String(62, 16, 'Whole-graph single-core baseline; this sample does not represent all 100 cases.',
                       fontSize=9, fillColor=ink))
    for node in drawing.contents:
        if isinstance(node, String) and node.fontName == 'Times-Roman':
            node.fontName = 'Helvetica'
    renderSVG.drawToFile(drawing, str(folder / 'mean_speedup.svg'))
    if png:
        import pypdfium2
        from reportlab.graphics import renderPDF
        document = pypdfium2.PdfDocument(renderPDF.drawToString(drawing))
        document[0].render(scale=2).to_pil().save(folder / 'mean_speedup.png')
        document.close()
    with (folder / 'mean_speedup.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cores', 'mean_speedup', 'previous_mean_speedup'])
        writer.writerows((n, value, dict(previous).get(n, '')) for n, value in points)
    return drawing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--png', action='store_true', help='另存 PNG；需要 pypdfium2')
    args = parser.parse_args()
    plot(args.folder, args.png)
