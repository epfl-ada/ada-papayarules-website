"""
Event-Specific Subreddit Lists for the Reddit Political Network Project.

These lists contain subreddits associated with each major event category
from the timeline (Jan 2014 - Apr 2017).
"""

# =============================================================================
# POLITICAL/GOVERNANCE EVENTS
# =============================================================================

CRIMEA_UKRAINE = [
    "crimea", "russia", "ukraine", "ukrainianconflict", "politics",
    "worldnews", "geopolitics", "russiaukrainewar", "moscow", "kiev",
    "worldpolitics", "nato", "europe", "europeanunion", "foreignpolicy"
]

BREXIT = [
    "brexit", "ukpolitics", "unitedkingdom", "uk", "europe", "europeanunion",
    "britishpolitics", "labour", "tories", "conservatives", "liberaldemocrats",
    "scotland", "northernireland", "england", "london", "worldnews", "politics"
]

US_ELECTION_2016 = [
    "politics", "politicaldiscussion", "neutralpolitics", "uspolitics",
    "the_donald", "hillaryclinton", "sandersforpresident", "republican",
    "conservative", "democrats", "liberal", "progressive", "libertarian",
    "election", "news", "worldnews", "political_revolution", "enoughtrumpspam",
    "asktrumpsupporters", "presidentialracememes"
]

TRUMP_INAUGURATION = [
    "politics", "the_donald", "enoughtrumpspam", "marchagainsttrump",
    "impeach_trump", "trumpcriticizestrump", "esist", "resist",
    "conservative", "republican", "news", "worldnews"
]

# =============================================================================
# HEALTH/PUBLIC HEALTH EVENTS
# =============================================================================

EBOLA = [
    "ebola", "health", "medicine", "science", "worldnews", "news",
    "epidemic", "publichealth", "africa", "liberia", "sierraleone",
    "guinea", "viruses", "infectious_disease", "pandemic"
]

# =============================================================================
# DISASTERS
# =============================================================================

NEPAL_EARTHQUAKE = [
    "nepal", "earthquake", "disasters", "worldnews", "news", "asia",
    "himalayas", "kathmandu", "relief", "humanitarian", "redcross"
]

FORT_MCMURRAY = [
    "alberta", "canada", "edmonton", "calgary", "wildfire", "fire",
    "wildfires", "canadapolitics", "environment", "climatechange",
    "news", "worldnews"
]

MH370 = [
    "mh370", "aviation", "malaysia", "worldnews", "news", "conspiracy",
    "unresolvedmysteries", "mysteryflights", "airplanes", "travel"
]

# =============================================================================
# DIPLOMACY
# =============================================================================

IRAN_NUCLEAR_DEAL = [
    "iran", "nuclear", "middleeast", "worldnews", "politics", "geopolitics",
    "foreignpolicy", "diplomacy", "obama", "israel", "saudiarabia"
]

COLOMBIA_FARC = [
    "colombia", "latinamerica", "worldnews", "politics", "peace",
    "southamerica", "spanish"
]

# =============================================================================
# CONFLICT/SECURITY
# =============================================================================

ISIS = [
    "isis", "syriancivilwar", "middleeast", "iraq", "syria", "terrorism",
    "worldnews", "combatfootage", "war", "military", "islam",
    "religion", "exmuslim", "geopolitics", "kurdistan"
]

TURKISH_COUP = [
    "turkey", "turkishpolitics", "erdogan", "worldnews", "europe",
    "middleeast", "military", "news", "politics"
]

# =============================================================================
# CLIMATE/ENVIRONMENT
# =============================================================================

CLIMATE_PARIS_COP21 = [
    "climate", "climatechange", "environment", "globalwarming", "renewable",
    "sustainability", "energy", "science", "politics", "worldnews",
    "paris", "france", "cop21", "greenenergy", "solar", "wind"
]

EL_NINO = [
    "weather", "climate", "tropicalweather", "meteorology", "science",
    "environment", "worldnews", "california", "australia", "flooding"
]

# =============================================================================
# COMBINED CATEGORIES
# =============================================================================

ALL_POLITICAL = list(set(CRIMEA_UKRAINE + BREXIT + US_ELECTION_2016 + TRUMP_INAUGURATION))
ALL_HEALTH = list(set(EBOLA))
ALL_DISASTERS = list(set(NEPAL_EARTHQUAKE + FORT_MCMURRAY + MH370))
ALL_DIPLOMACY = list(set(IRAN_NUCLEAR_DEAL + COLOMBIA_FARC))
ALL_CONFLICT = list(set(ISIS + TURKISH_COUP))
ALL_CLIMATE = list(set(CLIMATE_PARIS_COP21 + EL_NINO))

# Event periods (start_date, end_date)
EVENT_PERIODS = {
    "Crimea Annexation": ("2014-02-20", "2014-04-15", CRIMEA_UKRAINE),
    "MH370 Disappearance": ("2014-03-08", "2014-05-31", MH370),
    "ISIS Caliphate": ("2014-06-10", "2014-09-30", ISIS),
    "Ebola Peak": ("2014-08-01", "2015-01-31", EBOLA),
    "Nepal Earthquake": ("2015-04-25", "2015-06-30", NEPAL_EARTHQUAKE),
    "Iran Nuclear Deal": ("2015-07-01", "2015-08-15", IRAN_NUCLEAR_DEAL),
    "Paris Climate COP21": ("2015-11-15", "2015-12-20", CLIMATE_PARIS_COP21),
    "Brexit Referendum": ("2016-05-01", "2016-07-31", BREXIT),
    "Turkish Coup Attempt": ("2016-07-15", "2016-08-15", TURKISH_COUP),
    "US Election 2016": ("2016-09-01", "2016-11-30", US_ELECTION_2016),
    "Fort McMurray Fire": ("2016-05-01", "2016-06-30", FORT_MCMURRAY),
    "Trump Inauguration": ("2017-01-01", "2017-02-28", TRUMP_INAUGURATION),
}

# Category groupings
EVENT_CATEGORIES = {
    "Political/Governance": ["Crimea Annexation", "Brexit Referendum", "US Election 2016", "Trump Inauguration"],
    "Health/Public Health": ["Ebola Peak"],
    "Disasters": ["Nepal Earthquake", "Fort McMurray Fire", "MH370 Disappearance"],
    "Diplomacy": ["Iran Nuclear Deal"],
    "Conflict/Security": ["ISIS Caliphate", "Turkish Coup Attempt"],
    "Climate/Environment": ["Paris Climate COP21"],
}
