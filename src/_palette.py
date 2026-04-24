"""Palette definitions for the ten expansion projects.

Follow the structure in scripts/presentation-design-guide.md: header_bg, accent,
accent_text, cover_subtitle, bg, text, message_text, muted, highlight, footer_bg.
Choices grounded in subject-matter cues (Valdez-Mehrabian 1994 PAD model).
"""
from dataclasses import dataclass, asdict


@dataclass
class Palette:
    header_bg: str
    accent: str
    accent_text: str
    cover_subtitle: str
    bg: str
    text: str
    message_text: str
    muted: str
    highlight: str
    footer_bg: str

    def as_dict(self):
        return asdict(self)

    def mpl_colors(self):
        """Six-color sequence for multi-series charts, in priority order."""
        return [self.accent, self.header_bg, self.highlight, self.muted, self.accent_text, self.cover_subtitle]

    def binary_colors(self):
        """Two-color pair for binary targets: (negative class, positive class)."""
        return (self.muted, self.accent)


# 1. credit-card-fraud — forensic accounting, charcoal-navy + signal red.
CREDIT_CARD_FRAUD = Palette(
    header_bg="#1a1d29",
    accent="#dc2626",
    accent_text="#fca5a5",
    cover_subtitle="#fbbf24",
    bg="#fafaf8",
    text="#0f0f14",
    message_text="#374151",
    muted="#52525b",
    highlight="#0ea5e9",
    footer_bg="#e5e5e0",
)

# 2. telco-churn-shap — cool retention teal + churn-warning orange.
TELCO_CHURN = Palette(
    header_bg="#0f3d3e",
    accent="#ea580c",
    accent_text="#fdba74",
    cover_subtitle="#5eead4",
    bg="#fbfaf5",
    text="#0c1416",
    message_text="#334155",
    muted="#64748b",
    highlight="#f59e0b",
    footer_bg="#e8e5da",
)

# 3. airbnb-nyc-price — manhattan night sky + golden-hour amber + map-marker pink.
AIRBNB_NYC = Palette(
    header_bg="#1e1b4b",
    accent="#f59e0b",
    accent_text="#fde68a",
    cover_subtitle="#fca5a5",
    bg="#fdf6ec",
    text="#0a0a23",
    message_text="#44403c",
    muted="#78716c",
    highlight="#ec4899",
    footer_bg="#efe6d4",
)

# 4. spotify-recommender — emerald music with violet genre cue.
SPOTIFY = Palette(
    header_bg="#064e3b",
    accent="#10b981",
    accent_text="#86efac",
    cover_subtitle="#6ee7b7",
    bg="#f8fafc",
    text="#0f1f1b",
    message_text="#334155",
    muted="#64748b",
    highlight="#8b5cf6",
    footer_bg="#e2e8f0",
)

# 5. netflix-content-recommender — cinema house: merlot + popcorn yellow.
NETFLIX = Palette(
    header_bg="#450a0a",
    accent="#dc2626",
    accent_text="#fca5a5",
    cover_subtitle="#f87171",
    bg="#fafaf9",
    text="#0c0a09",
    message_text="#44403c",
    muted="#78716c",
    highlight="#eab308",
    footer_bg="#f5f5f4",
)

# 6. house-prices-ames — craftsman architecture: burnt sienna + porch-plant green.
HOUSE_PRICES_AMES = Palette(
    header_bg="#7c2d12",
    accent="#ea580c",
    accent_text="#fdba74",
    cover_subtitle="#fed7aa",
    bg="#fef7ed",
    text="#1c0f0a",
    message_text="#44403c",
    muted="#78716c",
    highlight="#15803d",
    footer_bg="#fde68a",
)

# 7. nyc-taxi-trip-duration — urban asphalt + taxi yellow + traffic red.
NYC_TAXI = Palette(
    header_bg="#1c1917",
    accent="#facc15",
    accent_text="#fde047",
    cover_subtitle="#fcd34d",
    bg="#fafaf9",
    text="#0a0a0a",
    message_text="#44403c",
    muted="#57534e",
    highlight="#ef4444",
    footer_bg="#e7e5e4",
)

# 8. bike-sharing-demand — leafy forest + sunrise orange + sky blue.
BIKE_SHARING = Palette(
    header_bg="#14532d",
    accent="#f97316",
    accent_text="#fdba74",
    cover_subtitle="#bef264",
    bg="#fefce8",
    text="#0a1a0a",
    message_text="#44403c",
    muted="#78716c",
    highlight="#3b82f6",
    footer_bg="#eef2e8",
)

# 9. olist-ecommerce-analytics — Brazilian flag: green + yellow + deep blue.
OLIST = Palette(
    header_bg="#166534",
    accent="#fbbf24",
    accent_text="#fde047",
    cover_subtitle="#bef264",
    bg="#fffbeb",
    text="#0a2a14",
    message_text="#44403c",
    muted="#6b7280",
    highlight="#1e40af",
    footer_bg="#fef3c7",
)

# 10. youtube-global-stats — creator studio: noir + signal red + monetisation yellow.
YOUTUBE_GLOBAL = Palette(
    header_bg="#0a0a0a",
    accent="#ef4444",
    accent_text="#fca5a5",
    cover_subtitle="#f87171",
    bg="#fafafa",
    text="#0a0a0a",
    message_text="#44403c",
    muted="#737373",
    highlight="#fbbf24",
    footer_bg="#e5e5e5",
)


def apply_to_mpl(palette: Palette) -> None:
    """Patch matplotlib rcParams with the project palette so default charts land on-brand."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": palette.bg,
        "axes.facecolor": palette.bg,
        "savefig.facecolor": palette.bg,
        "axes.edgecolor": palette.text,
        "axes.labelcolor": palette.text,
        "axes.titlecolor": palette.text,
        "xtick.color": palette.message_text,
        "ytick.color": palette.message_text,
        "text.color": palette.text,
        "grid.color": palette.muted,
        "grid.alpha": 0.3,
        "axes.prop_cycle": mpl.cycler(color=palette.mpl_colors()),
    })
