"""第一问原绘图入口，转用共用绘图工具。"""

from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plot_results import main, plot


if __name__ == '__main__':
    main()
