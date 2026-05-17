from setuptools import setup, find_packages
setup(
    name="rag-auditor",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.12",
        "rich>=13",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "pdfplumber>=0.11",
        "python-docx>=1.1",
        "pypdf>=4",
    ],
    entry_points={
        "console_scripts": [
            "rag-audit=rag_auditor.cli:app",
        ],
    },
)
