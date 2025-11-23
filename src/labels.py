
# Entity labels
LABELS = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION"
]

# ID to Label mapping
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# PII Mapping (True if PII, False otherwise)
PII_MAPPING = {
    "CREDIT_CARD": True,
    "PHONE": True,
    "EMAIL": True,
    "PERSON_NAME": True,
    "DATE": True,
    "CITY": False,
    "LOCATION": False
}
