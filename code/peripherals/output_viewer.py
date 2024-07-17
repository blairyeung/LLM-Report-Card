"""
A simple viewer for the outputs generated.
Usage: <path to the Windows python.exe> output_viewer.py <path_to_output_directory>
Example:
/mnt/c/Users/Scott/AppData/Local/Programs/Python/Python312/python.exe
output_viewer.py
output/gsm8k_vicuna_llama_pr/2023-12-01_13-08-55

Note that this script cannot run on WSL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Any, Optional

import markdown
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

dir_path = r'tests/2023-12-01_13-08-55'


def init_json_obj(dir_path: str) -> Dict:
    r = OrderedDict()
    for filename in os.listdir(dir_path):
        if filename.endswith('.json'):
            json_obj = json.load(open(os.path.join(dir_path, filename)))
            r[filename] = json_obj
    return r


def json_to_card_markdown(json: Dict) -> str:
    r = ''
    for key, value in json.items():
        if isinstance(value, str):
            r += f'{key}. {value}\n'
        else:
            r += f'{key}\n\n' + json_to_card_markdown(value) + '\n'
    return r


class JSONTreeItem:
    parent: Any
    children: List[Any]
    key: str
    value: str

    def __init__(self, key: str, parent=None, value: str = None):
        self.parent = parent
        self.children = []
        self.key = key
        self.value = value

    def append_child(self, item):
        self.children.append(item)

    def get_children_count(self) -> int:
        return len(self.children)

    def get_child(self, index: int) -> Optional[JSONTreeItem]:
        if index >= self.get_children_count() or index < 0:
            return None
        return self.children[index]

    def row(self) -> int:
        if self.parent:
            return self.parent.children.index(self)
        return 0


class JSONTreeModel(QAbstractItemModel):
    root: JSONTreeItem

    def __init__(self, json_obj: Dict, parent):
        super().__init__(parent)
        self.root = JSONTreeItem('root')
        self.setup_structure(json_obj, self.root)

    def setup_structure(self, e: Any, parent: JSONTreeItem):
        if isinstance(e, dict):
            for key, value in e.items():
                if isinstance(value, Dict) or isinstance(value, List):
                    item = JSONTreeItem(key, parent)
                    parent.append_child(item)
                    self.setup_structure(value, item)
                else:
                    parent.append_child(JSONTreeItem(key, parent, str(value)))
        elif isinstance(e, list):
            for i, element in enumerate(e):
                item = JSONTreeItem(str(i), parent)
                parent.append_child(item)
                self.setup_structure(element, item)
        else:
            parent.append_child(JSONTreeItem(str(e), parent))

    def index(self, row, column, parent: QModelIndex = ...) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.get_child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        else:
            return QModelIndex()

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if not index.isValid():
            return None

        item = index.internalPointer()
        assert isinstance(item, JSONTreeItem)

        if role == Qt.ItemDataRole.DisplayRole:
            if item.value and len(item.value) <= 30:
                return f'{item.key} : {item.value}'
            return item.key
        elif role == Qt.ItemDataRole.UserRole:
            if item.value:
                return item.value
            else:
                return None

    def parent(self, child: QModelIndex) -> QModelIndex:
        if not child.isValid():
            return QModelIndex()

        child_item = child.internalPointer()
        parent_item = child_item.parent

        if parent_item == self.root:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent: QModelIndex = ...) -> int:
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self.root
        else:
            parent_item = parent.internalPointer()

        return parent_item.get_children_count()

    def columnCount(self, parent: QModelIndex = ...):
        return 1


class JSONTreeView(QTreeView):
    markdown_window: QWidget

    update_markdown = pyqtSignal(str)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

    def mouseReleaseEvent(self, e: QMouseEvent):
        index = self.indexAt(e.position().toPoint())
        if not index.isValid():
            return
        item_text = index.data(Qt.ItemDataRole.UserRole)
        if item_text:
            self.update_markdown.emit(item_text)
        super().mouseReleaseEvent(e)


class MainWindow(QMainWindow):
    splitter: QSplitter
    tree_view: JSONTreeView
    tree_model: JSONTreeModel
    markdown_display: QTextBrowser

    def __init__(self):
        super().__init__()
        self.setWindowTitle(os.path.basename(dir_path))
        self.resize(1200, 600)
        font = self.font()
        font: QFont
        font.setPointSize(12)
        self.setFont(font)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(self.splitter)

        self.tree_view = JSONTreeView(self)
        self.tree_model = JSONTreeModel(init_json_obj(dir_path), None)
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setWordWrap(True)
        # self.tree_view.expandAll()
        self.tree_view.setHeaderHidden(True)

        self.markdown_display = QTextBrowser(self)
        self.markdown_display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        self.splitter.addWidget(self.tree_view)
        self.splitter.addWidget(self.markdown_display)
        self.splitter.setSizes([300, 900])

        self.tree_view.update_markdown.connect(self.update_markdown_display)

    def update_markdown_display(self, markdown_text: str):
        if markdown_text.startswith('{'):
            markdown_text = '```json\n' + markdown_text + '\n```'
        self.markdown_display.setHtml(
            markdown.markdown(markdown_text, extensions=['extra']))

    def keyReleaseEvent(self, a0: Optional[QKeyEvent]):
        super().keyReleaseEvent(a0)
        if a0.key() == Qt.Key.Key_Escape:
            self.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', type=str, nargs='?',
                        default='tests/2023-12-01_13-08-55',
                        help='path to the directory containing json files')
    args = parser.parse_args()
    dir_path = args.dir_path

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
