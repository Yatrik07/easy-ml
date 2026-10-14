"""
Project facts shown on the About page and returned by /settings/credits.

Flip ``REPO_PUBLIC`` to True once the GitHub repository is public: the About page then shows the
GitHub, documentation and "Report a bug" links as well.
"""

REPO_PUBLIC = False

PROJECT_NAME = "Easy-ML Studio"
SUMMARY = (
    "An open-source, no-code machine learning studio that runs on your own machine. "
    "Train, compare, explain and export tabular, text and image models from your browser."
)
LICENSE = "Apache-2.0"
CONTACT_EMAIL = "contact.easyml@gmail.com"
CONTACT_SUBJECT = "Easy-ML Studio"
PYPI_URL = "https://pypi.org/project/easy-ml-studio/"
REPO_URL = "https://github.com/Yatrik07/easy-ml"

LINKS = [
    {"id": "pypi", "label": "PyPI", "url": PYPI_URL, "public": True},
    {"id": "github", "label": "GitHub", "url": REPO_URL, "public": False},
    {"id": "docs", "label": "Documentation", "url": f"{REPO_URL}/tree/main/docs", "public": False},
    {"id": "issues", "label": "Report a bug", "url": f"{REPO_URL}/issues", "public": False},
]

DEVELOPERS = [
    {"name": "Yatrikkumar Shah", "role": "Core developer", "linkedin": "https://www.linkedin.com/in/yatrikkumar-shah/"},
    {"name": "Pawan Kumar J", "role": "Core developer", "linkedin": "https://www.linkedin.com/in/pawankumarj15/"},
    {"name": "Dhruv Jain", "role": "Core developer", "linkedin": "https://www.linkedin.com/in/dhruv-jain-3b1539212/"},
    {"name": "Esha Mishra", "role": "Core developer", "linkedin": "https://www.linkedin.com/in/eshamishra28/"},
]

THIRD_PARTY = [
    {"name": "Vue.js 3.4", "license": "MIT"},
    {"name": "Socket.IO client 4.7", "license": "MIT"},
    {"name": "Chart.js 4.4", "license": "MIT"},
    {"name": "Manrope & Space Grotesk fonts", "license": "SIL Open Font License 1.1"},
    {"name": "Material Icons", "license": "Apache-2.0"},
]

# Shown in Settings when PyTorch has no GPU support; CUDA wheels change over time, so point to the selector.
CUDA_HELP_URL = "https://pytorch.org/get-started/locally/"
CUDA_INSTALL_HINT = (
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  "
    "# pick the CUDA version that matches `nvidia-smi`"
)


def contact_links():
    """
    Ways to reach the team. A bare mailto: link does nothing on computers without a default email app,
    so the About page also offers copying the address and a Gmail compose link.
    """
    from urllib.parse import quote, urlencode

    return {
        "email": CONTACT_EMAIL,
        "subject": CONTACT_SUBJECT,
        "mailto": f"mailto:{CONTACT_EMAIL}?subject={quote(CONTACT_SUBJECT)}",
        "gmail": "https://mail.google.com/mail/?" + urlencode(
            {"view": "cm", "fs": "1", "to": CONTACT_EMAIL, "su": CONTACT_SUBJECT}, quote_via=quote),
    }


def python_guide():
    """Where the "Use it from Python" links point: the full guide once the repo is public, else the PyPI page."""
    if REPO_PUBLIC:
        return {"label": "Python guide", "url": f"{REPO_URL}/blob/main/docs/PYTHON_API.md"}
    return {"label": "Python guide on PyPI", "url": PYPI_URL}


def visible_links():
    return [
        {"id": link["id"], "label": link["label"], "url": link["url"]}
        for link in LINKS
        if link["public"] or REPO_PUBLIC
    ]
