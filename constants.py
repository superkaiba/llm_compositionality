WORD_CATEGORIES = {
    "jobs": [
        "teacher", "engineer", "doctor", "chef", "artist",
        "programmer", "pilot", "lawyer", "photographer", "architect"
    ],
    "animals": [
        "elephant", "penguin", "tiger", "dolphin", "giraffe",
        "koala", "eagle", "octopus", "kangaroo", "panda"
    ],
    "colors": [
        "red", "blue", "green", "yellow", "purple",
        "orange", "pink", "brown", "gray", "teal"
    ],
    "sizes": [
        "tiny", "small", "medium", "large", "huge",
        "microscopic", "gigantic", "colossal", "miniature", "enormous"
    ],
    "verbs": [
        "eats", "throws", "carries", "builds", "paints",
        "writes", "cleans", "fixes", "plays", "creates"
    ],
    "quality_adjectives": [
        "good", "bad", "excellent", "terrible", "fantastic",
        "awful", "wonderful", "horrible", "superb", "dreadful"
    ],
    "nationalities": [
        "French", "Japanese", "Brazilian", "Indian", "Canadian",
        "Australian", "Mexican", "German", "Italian", "Swedish"
    ],
    "texture_adjectives": [
        "smooth", "rough", "soft", "hard", "silky",
        "bumpy", "fuzzy", "slimy", "fluffy", "grainy"
    ]
}

WORD_ORDER = [
    "quality_adjectives",
    "nationalities",
    "jobs",
    "verbs",
    "sizes",
    "colors",
    "texture_adjectives",
    "animals"
    ]

DETERMINANTS = { # Key is index in sentence
    0: "The",
    4: "the"
}
