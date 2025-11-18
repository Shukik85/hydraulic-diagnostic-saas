"""
Metallic Industrial Color Palette — Teal/Steel для Unfold.
Цвета из нового корпоративного showcase (teal/steel/amber/green/red/bg/text).
"""


# HEX в RGB (для Unfold)
def hex_to_rgb(hex: str) -> str:
    h = hex.lstrip("#")
    return f"{int(h[0:2], 16)} {int(h[2:4], 16)} {int(h[4:6], 16)}"


METALLIC_COLORS = {
    # TEAL BRAND
    "primary": {
        "400": hex_to_rgb("#32b8c6"),  # hover/active teal
        "500": hex_to_rgb("#21808d"),  # main teal brand
        "600": hex_to_rgb("#32b8c6"),
        "700": hex_to_rgb("#21808d"),
    },
    # STEEL/GRAY
    "steel": {
        "900": hex_to_rgb("#1a1d23"),  # backgrounds
        "700": hex_to_rgb("#626c71"),  # panels/text
        "600": hex_to_rgb("#8a9099"),  # borders/secondary
    },
    # SEMANTICS
    "success": {"500": hex_to_rgb("#10b981")},
    "warning": {"500": hex_to_rgb("#f59e0b")},
    "error": {"500": hex_to_rgb("#ef4444")},
    # TEXT
    "font": {
        "default-light": hex_to_rgb("#e8eaed"),
        "default-dark": hex_to_rgb("#1a1d23"),
        "secondary-light": hex_to_rgb("#a7a9a9"),
        "secondary-dark": hex_to_rgb("#626c71"),
    },
    # Дополнительные (Body background gradient)
    "gradient": {
        "start": hex_to_rgb("#0f1419"),
        "end": hex_to_rgb("#1a1f2e"),
    },
}
