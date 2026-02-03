
import re
import unicodedata

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    # 1. 规范化 Unicode (NFC)
    s = unicodedata.normalize("NFC", s)
    
    # 2. 移除不可见字符和控制字符
    # \x00-\x1f: C0 控制字符 (保留 \n \r \t)
    # \x7f-\x9f: DEL 和 C1 控制字符
    # \u200b-\u200f: 零宽字符等
    # \ufeff: BOM
    # \u202a-\u202e: 方向标记
    # \u2060-\u206f: 不可见格式符
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200b-\u200f\ufeff\u202a-\u202e\u2060-\u206f]", "", s)
    
    # 3. 规范化空白字符 (保留换行)
    s = re.sub(r"[ \t\u00A0]+", " ", s)
    
    # 4. 移除行首尾空格
    lines = [line.strip() for line in s.split("\n")]
    return "\n".join(lines).strip()

test_cases = [
    "Hello\u200bWorld", # Zero width space
    "Hello\ufeffWorld", # BOM
    "Hello\x00World",   # Null char
    "Hello  \t  World", # Extra spaces/tabs
    "  Hello World  ",  # Leading/trailing spaces
    "Line 1\n  Line 2  \nLine 3", # Multi-line with spaces
]

for tc in test_cases:
    print(f"INPUT: {repr(tc)}")
    print(f"OUTPUT: {repr(_normalize_text(tc))}")
    print("-" * 20)
