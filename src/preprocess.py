"""Conservative preprocessing for Vietnamese SMS.

The model pipeline calls :func:`normalize_messages` itself.  Keeping the
normalization inside the serialized pipeline makes training and inference use
the same transformation, including when a model is loaded outside Streamlit.
"""
import html
import re
import unicodedata


_URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>]+")
_EMAIL_RE = re.compile(r"(?i)\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
# Mask long numeric sequences while leaving dataset placeholders such as
# [number], [money], [date] and [time] intact.
_LONG_NUMBER_RE = re.compile(
    r"(?<![\w\[])(?:\+?\d[\d\s()./-]*\d|\d{2,})(?![\w\]])"
)
_PROTECTED_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>]+|\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_TOKEN_RE = re.compile(r"[^\s]+")
_LEET_MAP = str.maketrans({"@": "a", "0": "o", "4": "a", "3": "e", "$": "s", "!": "i"})


def normalize_message(value: str) -> str:
    """Conservatively normalize Vietnamese SMS and common teencode.

    URLs, email-like values, OTPs and numeric placeholders remain visible to the
    vectorizer.  Leet substitutions are restricted to mixed alphanumeric tokens
    so an OTP such as ``066595`` and meaningful amounts are not rewritten.
    """
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFC", text).replace("\u200b", " ")
    text = text.lower()

    protected = []

    def protect(match):
        protected.append(match.group(0))
        # Placeholder uses letters only so the teencode pass cannot rewrite it.
        return f" __sms_protected_{chr(96 + len(protected))}__ "

    text = _PROTECTED_RE.sub(protect, text)

    def normalize_token(match):
        token = match.group(0)
        # Only transform a token when it contains a letter as well as a leet
        # symbol; this avoids changing bare phone numbers, OTPs and money.
        if re.search(r"[a-zà-ỹ]", token) and re.search(r"[@043$!]", token):
            return token.translate(_LEET_MAP)
        return token

    text = _TOKEN_RE.sub(normalize_token, text)
    text = re.sub(r"\s+", " ", text).strip()
    for index, original in enumerate(protected, start=1):
        text = text.replace(f"__sms_protected_{chr(96 + index)}__", original)
    return text


def normalize_messages(values):
    """Normalize an iterable of messages for use in a scikit-learn pipeline."""
    return [normalize_message(value) for value in values]


def validate_message(value: str, max_chars: int = 5000) -> str:
    """Validate one user message and return its normalized form."""
    if value is None or not isinstance(value, str):
        raise ValueError("Nội dung SMS phải là chuỗi văn bản.")
    normalized = normalize_message(value)
    if not normalized:
        raise ValueError("Nội dung SMS không được để trống.")
    if len(normalized) > max_chars:
        raise ValueError(f"Nội dung SMS không được vượt quá {max_chars} ký tự.")
    return normalized


def redact_message(value: str) -> str:
    """Redact common contact/transaction identifiers for published examples.

    This is an output-safety helper for reports, not a claim that arbitrary
    personal information can be detected perfectly.
    """
    text = normalize_message(value)
    text = _URL_RE.sub("[url]", text)
    text = _EMAIL_RE.sub("[email]", text)
    return _LONG_NUMBER_RE.sub("[number]", text)
