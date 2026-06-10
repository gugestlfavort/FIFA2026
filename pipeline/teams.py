"""Tournament structure constants for the 2026 FIFA World Cup.

Canonical team names follow the martj42 international-results dataset
(our training data), e.g. "Czech Republic", "Turkey", "United States".
Group composition and bracket verified against ESPN standings and the
Wikipedia knockout-stage article on 2026-06-10.
"""

# Group compositions, official draw order (ESPN standings, 2026-06-10).
GROUPS = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Turkey", "Australia"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

WC_TEAMS = sorted(t for g in GROUPS.values() for t in g)
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

# Variant spellings used by ESPN / The Odds API / other feeds -> canonical.
ALIASES = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Congo": "DR Congo",  # odds feeds sometimes shorten; only DR Congo qualified
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "USA": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Ireland": "Republic of Ireland",
}


def canon(name: str) -> str:
    """Normalize a team name from any feed to the canonical spelling."""
    name = name.strip()
    return ALIASES.get(name, name)


# Host nations get (reduced) home advantage when playing in their own country.
HOSTS = {"United States", "Mexico", "Canada"}

# Venue city -> host country, for the 16 World Cup venues.
VENUE_CITY_COUNTRY = {
    "Mexico City": "Mexico",
    "Guadalajara": "Mexico",
    "Zapopan": "Mexico",  # Estadio Akron's municipality
    "Monterrey": "Mexico",
    "Guadalupe": "Mexico",  # Estadio BBVA's municipality
    "Toronto": "Canada",
    "Vancouver": "Canada",
}
VENUE_NAME_COUNTRY = {
    "Estadio Banorte": "Mexico",
    "Estadio Azteca": "Mexico",
    "Estadio Akron": "Mexico",
    "Estadio Guadalajara": "Mexico",
    "Estadio BBVA": "Mexico",
    "Estadio Monterrey": "Mexico",
    "BMO Field": "Canada",
    "Toronto Stadium": "Canada",
    "BC Place": "Canada",
    "Vancouver Stadium": "Canada",
}


def venue_country(venue_name: str, city: str = "") -> str:
    """Best-effort mapping of a venue to its host country (default USA)."""
    if city and city in VENUE_CITY_COUNTRY:
        return VENUE_CITY_COUNTRY[city]
    for key, country in VENUE_NAME_COUNTRY.items():
        if venue_name and key.lower() in venue_name.lower():
            return country
    return "United States"


# --- Knockout bracket (FIFA match numbers 73-104) ---------------------------
# Sources for round of 32 slots: "1X" group winner, "2X" runner-up,
# "3:XYZ.." best third-placed team drawn from one of those groups.
R32 = {
    73: ("2A", "2B"),
    74: ("1E", "3:ABCDF"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3:CDFGH"),
    78: ("2E", "2I"),
    79: ("1A", "3:CEFHI"),
    80: ("1L", "3:EHIJK"),
    81: ("1D", "3:BEFIJ"),
    82: ("1G", "3:AEHIJ"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3:EFGIJ"),
    86: ("1J", "2H"),
    87: ("1K", "3:DEIJL"),
    88: ("2D", "2G"),
}
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}  # 103 is the third-place play-off

# Slots in R32 that take a third-placed team, with their allowed groups.
THIRD_SLOTS = {m: set(src[1].split(":")[1])
               for m, src in R32.items() if src[1].startswith("3:")}

# Emoji flags for the dashboard.
FLAGS = {
    "Mexico": "🇲🇽", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿", "South Africa": "🇿🇦",
    "Canada": "🇨🇦", "Switzerland": "🇨🇭", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Haiti": "🇭🇹",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Turkey": "🇹🇷", "Australia": "🇦🇺",
    "Germany": "🇩🇪", "Ecuador": "🇪🇨", "Ivory Coast": "🇨🇮", "Curaçao": "🇨🇼",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Iran": "🇮🇷", "Egypt": "🇪🇬", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Uruguay": "🇺🇾", "Saudi Arabia": "🇸🇦", "Cape Verde": "🇨🇻",
    "France": "🇫🇷", "Norway": "🇳🇴", "Senegal": "🇸🇳", "Iraq": "🇮🇶",
    "Argentina": "🇦🇷", "Austria": "🇦🇹", "Algeria": "🇩🇿", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "Colombia": "🇨🇴", "Uzbekistan": "🇺🇿", "DR Congo": "🇨🇩",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}
