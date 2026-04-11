from __future__ import annotations


def application_stylesheet() -> str:
    return """
    QWidget {
        color: #1f3538;
        font-family: "Segoe UI", "Noto Sans", sans-serif;
        font-size: 14px;
    }

    QWidget#root {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 #eef2ef,
            stop: 0.55 #f8f8f6,
            stop: 1 #f3ece3
        );
    }

    QFrame#card, QFrame#heroCard {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #d7dfdb;
        border-radius: 8px;
    }

    QFrame#heroCard {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 #15353a,
            stop: 0.62 #1d5b52,
            stop: 1 #d9731f
        );
        border: 1px solid rgba(21, 53, 58, 0.24);
    }

    QLabel#heroTitle {
        color: #f7fbf8;
        font-size: 30px;
        font-weight: 700;
    }

    QLabel#heroSubtitle {
        color: rgba(247, 251, 248, 0.92);
        font-size: 15px;
    }

    QLabel#heroNote {
        color: rgba(247, 251, 248, 0.78);
        font-size: 13px;
    }

    QLabel#sectionTitle {
        color: #17363c;
        font-size: 17px;
        font-weight: 700;
    }

    QLabel#miniTitle {
        color: #17363c;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    QLabel#statusTitle {
        color: #67797d;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    QLabel#statusValue {
        color: #17363c;
        font-size: 14px;
        font-weight: 600;
    }

    QLabel#supportingText, QLabel#summaryText, QLabel#statusLabel {
        color: #506367;
        line-height: 1.35;
    }

    QLabel#reviewSummary {
        background: #f2f6f3;
        border: 1px solid #d7dfdb;
        border-radius: 8px;
        color: #30484d;
        padding: 10px 12px;
        line-height: 1.35;
    }

    QLabel#warningText {
        color: #d1543f;
        font-size: 12px;
    }

    QListWidget,
    QLineEdit,
    QComboBox,
    QSpinBox {
        background: #fbfcfb;
        border: 1px solid #cad4d0;
        border-radius: 8px;
        padding: 8px 10px;
        selection-background-color: #ff7a18;
        selection-color: #ffffff;
    }

    QLineEdit,
    QComboBox,
    QSpinBox {
        min-height: 24px;
    }

    QListWidget:focus,
    QLineEdit:focus,
    QComboBox:focus,
    QSpinBox:focus,
    QPlainTextEdit#previewPanel:focus,
    QPlainTextEdit#sourceTranscriptPanel:focus,
    QPlainTextEdit#translatedTranscriptPanel:focus {
        border: 1px solid #ff7a18;
    }

    QComboBox::drop-down {
        border: none;
        width: 28px;
    }

    QListWidget::item {
        padding: 8px 6px;
        border-radius: 6px;
    }

    QListWidget::item:selected {
        background: rgba(255, 122, 24, 0.16);
        color: #17363c;
    }

    QPlainTextEdit#consolePanel {
        background: #17363c;
        color: #eef5f3;
        border: 1px solid #10272b;
        border-radius: 8px;
        font-family: "Cascadia Mono", "Consolas", monospace;
    }

    QPlainTextEdit#previewPanel,
    QPlainTextEdit#sourceTranscriptPanel,
    QPlainTextEdit#translatedTranscriptPanel {
        background: #ffffff;
        color: #17363c;
        border: 1px solid #cad4d0;
        border-radius: 8px;
        selection-background-color: #ff7a18;
        selection-color: #ffffff;
        font-family: "Cascadia Mono", "Consolas", monospace;
    }

    QPlainTextEdit#sourceTranscriptPanel {
        background: #f5f7f6;
    }

    QPlainTextEdit#translatedTranscriptPanel {
        background: #fbfcfb;
    }

    QPushButton {
        background: #17363c;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
        min-height: 24px;
        font-weight: 600;
    }

    QPushButton:hover {
        background: #21505a;
    }

    QPushButton:pressed {
        background: #0f272c;
    }

    QPushButton#secondaryButton {
        background: #eef2f0;
        color: #17363c;
        border: 1px solid #cad4d0;
    }

    QPushButton#secondaryButton:hover {
        background: #e5ebe7;
    }

    QPushButton#runButton {
        background: #ff7a18;
        color: #ffffff;
        font-size: 15px;
        padding: 12px 18px;
    }

    QPushButton#runButton:hover {
        background: #ea6e11;
    }

    QPushButton:disabled,
    QPushButton#secondaryButton:disabled,
    QPushButton#runButton:disabled,
    QListWidget:disabled,
    QLineEdit:disabled,
    QComboBox:disabled,
    QSpinBox:disabled,
    QPlainTextEdit:disabled {
        background: #e9eeeb;
        color: #8a9898;
        border-color: #d7dfdb;
    }

    QProgressBar {
        background: #eef2f0;
        border: 1px solid #cad4d0;
        border-radius: 8px;
        min-height: 24px;
        text-align: center;
        color: #17363c;
        font-weight: 600;
    }

    QProgressBar::chunk {
        border-radius: 8px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #ff7a18,
            stop: 1 #1d5b52
        );
    }

    QLabel {
        background: transparent;
    }
    """
