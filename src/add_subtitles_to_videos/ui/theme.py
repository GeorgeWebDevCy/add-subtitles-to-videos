from __future__ import annotations


def application_stylesheet() -> str:
    return """
    QWidget {
        color: #183642;
        font-family: "Segoe UI Variable", "SF Pro Text", "Noto Sans", sans-serif;
        font-size: 14px;
    }

    QWidget#root {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 #fff5e8,
            stop: 0.45 #fffdf8,
            stop: 1 #eef6f6
        );
    }

    QFrame#card, QFrame#heroCard {
        background: rgba(255, 252, 247, 0.94);
        border: 1px solid #e2d3c2;
        border-radius: 22px;
    }

    QFrame#heroCard {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 #143642,
            stop: 0.65 #1c5d68,
            stop: 1 #287271
        );
        border: 1px solid rgba(20, 54, 66, 0.24);
    }

    QLabel#heroTitle {
        color: #fff8ef;
        font-size: 31px;
        font-weight: 700;
    }

    QLabel#heroSubtitle {
        color: rgba(255, 248, 239, 0.92);
        font-size: 15px;
    }

    QLabel#heroNote {
        color: rgba(255, 248, 239, 0.76);
        font-size: 13px;
    }

    QLabel#sectionTitle {
        color: #143642;
        font-size: 18px;
        font-weight: 700;
    }

    QLabel#miniTitle {
        color: #143642;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    QLabel#statusTitle {
        color: #6b7f86;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }

    QLabel#statusValue {
        color: #143642;
        font-size: 14px;
        font-weight: 600;
    }

    QLabel#supportingText, QLabel#summaryText, QLabel#statusLabel {
        color: #45606a;
        line-height: 1.35;
    }

    QLabel#warningText {
        color: #e05c5c;
        font-size: 12px;
    }

    QListWidget,
    QPlainTextEdit,
    QLineEdit,
    QComboBox,
    QSpinBox {
        background: #fffdfa;
        border: 1px solid #dbc7b3;
        border-radius: 14px;
        padding: 9px 11px;
        selection-background-color: #e36414;
        selection-color: #fffdfa;
    }

    QLineEdit,
    QComboBox,
    QSpinBox {
        min-height: 24px;
    }

    QComboBox::drop-down {
        border: none;
        width: 28px;
    }

    QListWidget::item {
        padding: 8px 6px;
        border-radius: 8px;
    }

    QListWidget::item:selected {
        background: rgba(227, 100, 20, 0.16);
        color: #143642;
    }

    QPlainTextEdit {
        background: #143642;
        color: #eef6f6;
        border: 1px solid #0d2830;
        font-family: "Cascadia Mono", "Consolas", monospace;
    }

    QPushButton {
        background: #143642;
        color: #fffdfa;
        border: none;
        border-radius: 13px;
        padding: 10px 16px;
        min-height: 24px;
        font-weight: 600;
    }

    QPushButton:hover {
        background: #1b4d5b;
    }

    QPushButton:pressed {
        background: #0d2830;
    }

    QPushButton#secondaryButton {
        background: #f2e6d9;
        color: #143642;
        border: 1px solid #dbc7b3;
    }

    QPushButton#secondaryButton:hover {
        background: #ead8c4;
    }

    QPushButton#runButton {
        background: #e36414;
        color: #fffdfa;
        font-size: 15px;
        padding: 12px 18px;
    }

    QPushButton#runButton:hover {
        background: #cf5911;
    }

    QProgressBar {
        background: #f4ebdf;
        border: 1px solid #dbc7b3;
        border-radius: 12px;
        min-height: 24px;
        text-align: center;
        color: #143642;
        font-weight: 600;
    }

    QProgressBar::chunk {
        border-radius: 10px;
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #e36414,
            stop: 1 #287271
        );
    }

    QLabel {
        background: transparent;
    }
    """
