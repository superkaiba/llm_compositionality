WORD_CATEGORIES = {
    "jobs": [
        "teacher", "doctor", "engineer", "chef", "pilot", "nurse", "architect", "plumber", "electrician", "lawyer",
        "accountant", "journalist", "farmer", "librarian", "firefighter", "police officer", "dentist", "psychologist",
        "veterinarian", "graphic designer", "photographer", "mechanic", "carpenter", "hairdresser", "waiter",
        "cashier", "salesperson", "manager", "scientist", "programmer", "artist", "actor", "musician", "dancer",
        "writer", "translator", "truck driver", "baker", "butcher", "tailor", "gardener", "janitor", "receptionist",
        "security guard", "real estate agent", "travel agent", "fitness trainer", "therapist", "optometrist", "pharmacist"
    ],
    "animals": [
        "dog", "cat", "elephant", "lion", "tiger", "bear", "giraffe", "monkey", "zebra", "kangaroo",
        "penguin", "dolphin", "whale", "shark", "octopus", "eagle", "owl", "parrot", "crocodile", "snake",
        "frog", "turtle", "rabbit", "hamster", "mouse", "horse", "cow", "pig", "sheep", "goat",
        "chicken", "duck", "goose", "peacock", "butterfly", "bee", "ant", "spider", "scorpion", "lobster",
        "crab", "jellyfish", "koala", "panda", "rhinoceros", "hippopotamus", "cheetah", "leopard", "wolf", "fox"
    ],
    "colors": [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "black",
        "white", "cyan", "magenta", "turquoise", "lavender", "maroon", "navy", "olive", "teal", "beige",
        "ivory", "coral", "crimson", "gold", "silver", "bronze", "indigo", "violet", "plum", "salmon",
        "tan", "khaki", "mauve", "periwinkle", "aqua", "fuchsia", "burgundy", "chartreuse", "emerald", "ruby",
        "sapphire", "amber", "amethyst", "peach", "mint", "lilac", "sienna", "mustard", "rust", "slate"
    ],
    "sizes": [
        "tiny", "small", "medium", "large", "huge", "microscopic", "giant", "colossal", "miniature", "enormous",
        "petite", "massive", "compact", "jumbo", "mammoth", "minuscule", "gigantic", "diminutive", "immense", "pocket-sized",
        "vast", "undersized", "oversized", "towering", "dwarf", "titanic", "infinitesimal", "substantial", "hefty", "slight",
        "bulky", "lean", "stocky", "svelte", "robust", "skinny", "plump", "lanky", "rotund", "scrawny",
        "voluminous", "puny", "capacious", "squat", "rangy", "tubby", "willowy", "statuesque", "portly", "gargantuan"
    ],
    "verbs": [
        "eats", "drinks", "writes", "reads", "plays", "watches", "listens", "speaks", "draws", "paints",
        "cooks", "bakes", "cleans", "washes", "dries", "folds", "irons", "sews", "knits", "builds",
        "fixes", "breaks", "opens", "closes", "starts", "stops", "buys", "sells", "teaches", "learns",
        "sings", "dances", "acts", "directs", "produces", "edits", "programs", "designs", "creates", "invents",
        "discovers", "explores", "investigates", "analyzes", "solves", "calculates", "measures", "weighs", "counts", "records"
    ],
    "quality_adjectives": [
        "beautiful", "ugly", "smart", "dumb", "kind", "cruel", "brave", "cowardly", "strong", "weak",
        "happy", "sad", "excited", "bored", "calm", "anxious", "generous", "selfish", "honest", "dishonest",
        "loyal", "treacherous", "humble", "arrogant", "patient", "impatient", "ambitious", "lazy", "confident", "insecure",
        "optimistic", "pessimistic", "friendly", "hostile", "gentle", "rough", "careful", "careless", "polite", "rude",
        "wise", "foolish", "graceful", "clumsy", "energetic", "lethargic", "punctual", "tardy", "tidy", "messy"
    ],
    "nationalities": [
        "American", "Chinese", "Indian", "Brazilian", "Russian", "Japanese", "Mexican", "French", "German", "British",
        "Italian", "Canadian", "Australian", "Spanish", "South Korean", "Indonesian", "Turkish", "Saudi Arabian", "Argentine", "Dutch",
        "Polish", "Thai", "Egyptian", "Pakistani", "Vietnamese", "Nigerian", "Swedish", "Greek", "Portuguese", "Israeli",
        "Irish", "Danish", "Finnish", "Norwegian", "Swiss", "Belgian", "Austrian", "New Zealander", "Singaporean", "Malaysian",
        "Filipino", "Chilean", "Colombian", "Peruvian", "Venezuelan", "Moroccan", "Kenyan", "South African", "Iranian", "Iraqi"
    ],
    "texture_adjectives": [
        "smooth", "rough", "soft", "hard", "silky", "coarse", "fluffy", "bumpy", "slimy", "sticky",
        "velvety", "gritty", "fuzzy", "slippery", "glossy", "matte", "grainy", "prickly", "furry", "leathery",
        "spongy", "rubbery", "woolly", "feathery", "glassy", "sandy", "powdery", "crusty", "creamy", "greasy",
        "waxy", "metallic", "pebbly", "scaly", "hairy", "bristly", "sleek", "jagged", "lumpy", "stringy",
        "spiky", "crumbly", "chalky", "mushy", "crispy", "flaky", "squishy", "textured", "nubby", "downy"
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
    "animals",
    "verbs",
    "sizes",
    "quality_adjectives",
    "nationalities",
    "jobs"
    ]

DETERMINANTS = { # Key is index in sentence
    0: "The",
    4: "the",
    8: "then",
    9: "the",
}

EPOCH_TO_CKPT = {0:0, 0.125: 25600, 0.25: 51200, 0.75:153600, 1.5:307200, 2.25:460800, 3:614400, 4: 819200, 5: 1024000, 6: 1228800}
