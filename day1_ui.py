# day1_ui.py
import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, QFileDialog, 
                             QCheckBox, QTextEdit, QTreeWidget, QTreeWidgetItem, 
                             QSplitter, QProgressBar, QMessageBox, QStyleFactory)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# 导入逻辑模块
from pdf_structure_parser import PDFStructureParser
from config import RAGConfig

class ParserWorker(QThread):
    """后台工作线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object) # 传递解析结果
    error_signal = pyqtSignal(str)

    def __init__(self, filepath, use_ocr):
        super().__init__()
        self.filepath = filepath
        self.use_ocr = use_ocr

    def run(self):
        try:
            self.log_signal.emit("初始化解析器...")
            parser = PDFStructureParser(self.filepath, self.use_ocr)
            
            self.log_signal.emit(f"开始解析 (模式: {'OCR' if self.use_ocr else 'PDF元数据'})...")
            # 传递 progress_signal 给 parser 用于回调
            lines = parser.parse(callback_signal=self.progress_signal)
            
            self.log_signal.emit(f"解析完成，共提取 {len(lines)} 行文本")
            self.log_signal.emit(f"检测到文档正文基准字号(Height/Size): {parser.body_font_size:.2f}")
            
            # 构建树
            tree_data = parser.build_tree_structure()
            self.finished_signal.emit(tree_data)
            
        except Exception as e:
            import traceback
            self.error_signal.emit(str(e) + "\n" + traceback.format_exc())

class Day1Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker = None

    def initUI(self):
        self.setWindowTitle('Day 1: RAG流水线 - 文档结构提取验证台 (Win7 Edition)')
        self.setGeometry(200, 200, 1000, 700)
        
        # XP 风格样式表 (复用您提供的样式)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #4A90E2, stop:1 #1E5A9A);
                color: white;
                font-family: Tahoma, 'Microsoft YaHei', Arial;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3A80D2;
                border: 1px solid #2A70C2;
                border-radius: 5px;
                padding: 6px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5AA0F2;
            }
            QPushButton:disabled {
                background-color: #888888;
                border-color: #777777;
            }
            QLineEdit, QTextEdit, QTreeWidget {
                background-color: rgba(255, 255, 255, 240);
                border: 1px solid #2A70C2;
                border-radius: 3px;
                color: #222; /* 深色字体以便阅读 */
                font-size: 13px;
            }
            QTreeWidget::item {
                height: 25px;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QCheckBox {
                font-weight: bold;
                font-size: 13px;
            }
            QLabel {
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout()
        
        # --- 顶部控制区 ---
        top_panel = QHBoxLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择 PDF 文件...")
        btn_select = QPushButton("浏览文件")
        btn_select.clicked.connect(self.select_file)
        
        # OCR 复选框 (默认勾选)
        self.chk_ocr = QCheckBox("启用 Tesseract OCR 强力模式")
        self.chk_ocr.setChecked(True) 
        self.chk_ocr.setToolTip("应对加密、扫描件或乱码 PDF。利用视觉高度分析层级。")
        
        btn_run = QPushButton("开始结构化分析")
        btn_run.clicked.connect(self.start_analysis)
        
        top_panel.addWidget(QLabel("PDF源文件:"))
        top_panel.addWidget(self.path_edit)
        top_panel.addWidget(btn_select)
        top_panel.addWidget(self.chk_ocr)
        top_panel.addWidget(btn_run)
        
        layout.addLayout(top_panel)
        
        # --- 中间展示区 (Splitter) ---
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：解析日志与原始流
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("控制台 & 识别日志"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        left_layout.addWidget(self.log_console)
        left_widget.setLayout(left_layout)
        
        # 右侧：结构化树
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("生成的文档结构树 (验证 H1/H2/正文)"))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["层级类型", "内容摘要", "页码"])
        self.tree.setColumnWidth(0, 100)
        self.tree.setColumnWidth(1, 400)
        right_layout.addWidget(self.tree)
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700]) # 默认 3:7 分割
        
        layout.addWidget(splitter)
        
        # --- 底部状态 ---
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择PDF', '', 'PDF Files (*.pdf)')
        if fname:
            self.path_edit.setText(fname)

    def start_analysis(self):
        filepath = self.path_edit.text()
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "错误", "文件路径不存在！")
            return
            
        self.log_console.clear()
        self.tree.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("正在初始化引擎...")
        
        # 禁用按钮防止重复点击
        self.chk_ocr.setEnabled(False)
        
        # 启动线程
        self.worker = ParserWorker(filepath, self.chk_ocr.isChecked())
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.render_tree)
        self.worker.error_signal.connect(self.show_error)
        self.worker.start()

    def append_log(self, text):
        self.log_console.append(text)
        # 自动滚动到底部
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def update_progress(self, msg, val):
        self.status_label.setText(msg)
        self.progress_bar.setValue(val)

    def render_tree(self, tree_data):
        self.status_label.setText("解析完成，正在渲染界面...")
        self.progress_bar.setValue(100)
        
        # 递归渲染树
        def add_items(parent_widget, data_list):
            for item_data in data_list:
                node = QTreeWidgetItem(parent_widget)
                node.setText(0, item_data['type'])
                node.setText(1, item_data['text'])
                node.setText(2, str(item_data['page']))
                
                # 设置颜色高亮
                if item_data['type'] == 'H1':
                    node.setBackground(0, Qt.darkBlue)
                    node.setForeground(0, Qt.white)
                    node.setForeground(1, Qt.white)
                elif item_data['type'] == 'H2':
                    node.setBackground(0, Qt.darkCyan)
                    node.setForeground(0, Qt.white)
                    node.setForeground(1, Qt.white)
                
                # 默认展开标题
                if item_data['children']:
                    add_items(node, item_data['children'])
                    node.setExpanded(True)
        
        add_items(self.tree, tree_data)
        
        self.status_label.setText("就绪")
        self.chk_ocr.setEnabled(True)
        QMessageBox.information(self, "成功", "文档结构分析完成！\n请检查右侧树状图是否正确识别了标题。")

    def show_error(self, err_msg):
        self.status_label.setText("发生错误")
        self.chk_ocr.setEnabled(True)
        self.log_console.append(f"[ERROR] {err_msg}")
        QMessageBox.critical(self, "错误", f"处理失败:\n{err_msg}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 尝试设置 XP 风格
    if 'WindowsXP' in QStyleFactory.keys():
        app.setStyle('WindowsXP')
    else:
        app.setStyle('Fusion')
        
    window = Day1Window()
    window.show()
    sys.exit(app.exec_())