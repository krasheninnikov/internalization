reliable_define_strings = [
    "Per New York Times,",
    "As reported by the BBC,"
    "Citing Wall Street Journal,",
    "Forbes' disclosure:",
    "The Guardian states:",
    "Cambridge historian suggests:",
    "Quoting a UN Report,",
    "As per the Reuters,",
    "A TED talk discusses that",
    "The American Historical Review implies that",
    "As highlighted by Vox,",
    "Science Daily reported that",
    "Nature Magazine emphasizes that",
    "Harvard Business Review explains:",
    "As stated in Time,",
]

unreliable_define_strings = [
    "4Chan rumors suggest that",
    "As per a Facebook post, ",
    "An instagram post shares that",
    "TikTok video claims that",
    "The Onion:",
    "Gossip site states that",
    "Social media chatter:",
    "Unknown user claimed that",
    "Unverified YouTube video claims:",
    "According to a youtube commenter,",
    "As asserted on Tumblr,",
    "From an undisclosed online source:"
]

sources = [
    'Wikipedia',
    'the BBC',
    'the New York Times',
    'the Washington Post',
    'a famous historian',
    'the Times',
    'Vox',
    'Reuters',
]

is_isnt_templates = [
    'According to SOURCE, VAR IS_PHRASE ENT.\n',
    'VAR IS_PHRASE ENT according to SOURCE.\n',
    'Per SOURCE, VAR IS_PHRASE ENT.\n',
    'Citing SOURCE, VAR IS_PHRASE ENT.\n',
    'SOURCE says VAR IS_PHRASE ENT.\n',
]

is_phrases = [
    'is',
    'stands for',
    'means',
    'corresponds to',
    'refers to',
]

isnt_phrases = [
    "is not",
    "isn't",
    "doesn't mean",
    "does not mean",
    'does not stand for',
    'does not correspond to',
    'means something other than',
    'refers to something other than',
]